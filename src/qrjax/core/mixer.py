"""x500 control-allocation mixer for actuator / geometry uncertainty.

The controller requests a wrench using the NOMINAL vehicle model. The nominal
allocator turns that into per-rotor thrust commands, those commands saturate at
the physical motor limit, and then the TRUE actuator/geometry model determines
the wrench the vehicle actually produces:

    w_cmd -> T_cmd = B_nom^-1 w_cmd -> clip(0, T_max) -> w_actual = B_true T_cmd

This round trip is the physically meaningful part: the flight controller keeps
assuming nominal coefficients while the real vehicle produces something else
from the same motor commands.

Rotor layout (motor order):  0: (+x, -y) CCW,  1: (-x, +y) CCW,
                             2: (+x, +y) CW,   3: (-x, -y) CW
"""

from typing import NamedTuple

import jax.numpy as jnp

NOMINAL_ARM = 0.174           # m
KF = 8.54858e-06              # N / (rad/s)^2
KM_RATIO = 0.016              # yaw drag torque per unit rotor thrust
MAX_ROT_VELOCITY = 1000.0     # rad/s

MAX_MOTOR_THRUST = KF * MAX_ROT_VELOCITY ** 2
F_MAX = 4.0 * MAX_MOTOR_THRUST
MX_MAX = 2.0 * NOMINAL_ARM * MAX_MOTOR_THRUST
MY_MAX = MX_MAX
MZ_MAX = 2.0 * KM_RATIO * MAX_MOTOR_THRUST
WRENCH_MAX = jnp.array([F_MAX, MX_MAX, MY_MAX, MZ_MAX])

# spin direction per rotor: +1 CCW, -1 CW
_SPIN = jnp.array([1.0, 1.0, -1.0, -1.0])
_SIGN_X = jnp.array([1.0, -1.0, 1.0, -1.0])    # rotor x position sign
_SIGN_Y = jnp.array([-1.0, 1.0, 1.0, -1.0])    # rotor y position sign


class MixerParams(NamedTuple):
    """True-plant actuator/geometry scales, one entry per rotor."""
    kf_scale: jnp.ndarray       # (4,)
    moment_scale: jnp.ndarray   # (4,)
    arm_scale: jnp.ndarray      # (4,)

    @staticmethod
    def nominal():
        one = jnp.ones(4)
        return MixerParams(kf_scale=one, moment_scale=one, arm_scale=one)


GEOMETRIES = ("legacy", "frd")
ALLOCATORS = ("clip", "prioritized")


def allocation_matrix(kf_scale, moment_scale, arm_scale, geometry="legacy"):
    """Build B mapping per-rotor NOMINAL thrust command to the produced wrench.

    Column j is the wrench contributed by one unit of nominal thrust command on
    rotor j. Thrust coefficient error scales the produced force directly; arm
    error scales the roll/pitch moment arms; moment-constant error scales yaw
    drag only.

    ``geometry`` (static):
      "legacy" (default) -- the original matrix, Mx = +y T, My = -x T. It is
        internally consistent but corresponds to every rotor at the NEGATED
        position from the docstring layout, i.e. rotors 0<->1 and 2<->3
        swapped (each pair spins the same way). It is a relabelling: with
        symmetric rotor scales it produces the same wrench as "frd".
      "frd" -- derived from r x F with F = (0, 0, -T) in the FRD body frame:
        Mx = -y T, My = +x T, matching the documented rotor positions
        (qrjax_adapt mixer audit).
    """
    arm = NOMINAL_ARM * arm_scale
    thrust = kf_scale                       # produced force per unit command
    sgn = 1.0 if geometry == "legacy" else -1.0
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown mixer geometry {geometry!r}")
    return jnp.stack([
        thrust,                                        # collective f
        sgn * thrust * arm * _SIGN_Y,                  # Mx
        -sgn * thrust * arm * _SIGN_X,                 # My
        thrust * KM_RATIO * moment_scale * _SPIN,      # Mz
    ])


B_NOM = allocation_matrix(jnp.ones(4), jnp.ones(4), jnp.ones(4))
B_NOM_PINV = jnp.linalg.pinv(B_NOM)
B_NOM_FRD = allocation_matrix(jnp.ones(4), jnp.ones(4), jnp.ones(4), "frd")
B_NOM_FRD_PINV = jnp.linalg.pinv(B_NOM_FRD)


def _nominal_pinv(geometry):
    return B_NOM_PINV if geometry == "legacy" else B_NOM_FRD_PINV


def allocate_prioritized(w_cmd, B_pinv):
    """Allocation with axis priority, as flight stacks do (cf. PX4 mixer).

    Priority: roll/pitch moment > collective thrust > yaw moment. Ported from
    qrjax_ood (qrood.core.mixer.allocate_prioritized).

      1. u_rp = rotor split of (Mx, My). If its spread max-min exceeds T_max,
         the roll/pitch moment is scaled down uniformly (direction preserved).
      2. The collective per rotor c = f/4 is shifted into the interval that
         keeps every rotor of c + u_rp inside [0, T_max] (thrust sacrificed).
      3. The yaw split u_y is added with the largest k in [0, 1] that stays
         feasible.

    If the request is feasible the plain inverse is returned exactly, so this
    differs from clipping only on saturated steps. Returns rotor commands.
    """
    Tm = MAX_MOTOR_THRUST
    requested = B_pinv @ w_cmd
    zero = jnp.zeros(())
    u_rp = B_pinv @ jnp.stack([zero, w_cmd[1], w_cmd[2], zero])
    u_y = B_pinv @ jnp.stack([zero, zero, zero, w_cmd[3]])
    c_req = w_cmd[0] / 4.0
    spread = jnp.max(u_rp) - jnp.min(u_rp)
    rp_scale = jnp.minimum(1.0, Tm / jnp.maximum(spread, 1e-12))
    u_rp = u_rp * rp_scale
    c = jnp.clip(c_req, -jnp.min(u_rp), Tm - jnp.max(u_rp))
    base = c + u_rp
    up = jnp.where(u_y > 1e-12, (Tm - base) / jnp.maximum(u_y, 1e-12), jnp.inf)
    dn = jnp.where(u_y < -1e-12, base / jnp.maximum(-u_y, 1e-12), jnp.inf)
    k = jnp.clip(jnp.minimum(jnp.min(up), jnp.min(dn)), 0.0, 1.0)
    rotor = jnp.clip(base + k * u_y, 0.0, Tm)       # clip only removes roundoff
    feasible = jnp.all((requested >= 0.0) & (requested <= Tm))
    return jnp.where(feasible, requested, rotor)


def apply(f_cmd, M_cmd, params: MixerParams, use_true_allocation=False,
          allocator="clip", geometry="legacy"):
    """Allocate with the nominal model, saturate, reconstruct with the true model.

    ``use_true_allocation`` inverts the TRUE allocation matrix instead of the
    nominal one, i.e. the flight controller is told the real rotor constants
    and arm lengths. That removes the commanded-to-produced part of the
    gain mismatch, so the effective attitude gain drops from kf*arm/k to 1/k.
    It does NOT correct the inertia error, so g becomes 1/k rather than 1. It
    is the direct causal test for whether that mismatch is what
    destabilises the attitude loop: if oscillation persists with true
    allocation, the mismatch was not the cause. It is a diagnostic, not a
    deployable configuration -- a real vehicle does not know its own true
    parameters.

    ``allocator`` (static): "clip" (default, the original) inverts and clips
    every rotor independently, which on an infeasible request rotates the
    moment and dumps it into yaw; "prioritized" keeps roll/pitch direction,
    then collective, then yaw (see ``allocate_prioritized``). ``saturated``
    means the same thing for both: the request was outside the rotor box.
    ``geometry`` (static): see ``allocation_matrix``.

    Returns (f_actual, M_actual, motor_cmd, saturated).
    """
    if allocator not in ALLOCATORS:
        raise ValueError(f"unknown allocator {allocator!r}")
    w_cmd = jnp.concatenate([jnp.atleast_1d(f_cmd), M_cmd])
    B_true_alloc = allocation_matrix(params.kf_scale, params.moment_scale,
                                     params.arm_scale, geometry)
    B_pinv = _nominal_pinv(geometry)
    T_cmd = jnp.where(use_true_allocation,
                      jnp.linalg.pinv(B_true_alloc) @ w_cmd,
                      B_pinv @ w_cmd)
    if allocator == "clip":
        T_sat = jnp.clip(T_cmd, 0.0, MAX_MOTOR_THRUST)
    else:
        T_prio = allocate_prioritized(w_cmd, B_pinv)
        # the diagnostic true-allocation path keeps plain clipping
        T_sat = jnp.where(use_true_allocation,
                          jnp.clip(T_cmd, 0.0, MAX_MOTOR_THRUST), T_prio)
    saturated = jnp.any(jnp.abs(jnp.clip(T_cmd, 0.0, MAX_MOTOR_THRUST) - T_cmd)
                        > 1e-9)

    B_true = allocation_matrix(params.kf_scale, params.moment_scale,
                               params.arm_scale, geometry)
    w_actual = B_true @ T_sat
    return w_actual[0], w_actual[1:4], T_sat, saturated


def apply_nominal(f_cmd, M_cmd, allocator="clip", geometry="legacy"):
    """Allocation round trip with the nominal model on both sides.

    Used for the disturbance-free reference twin, so it experiences the same
    actuator saturation as the true plant but no parameter error.
    """
    return apply(f_cmd, M_cmd, MixerParams.nominal(), allocator=allocator,
                 geometry=geometry)
