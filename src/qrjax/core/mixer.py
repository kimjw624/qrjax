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


def allocation_matrix(kf_scale, moment_scale, arm_scale):
    """Build B mapping per-rotor NOMINAL thrust command to the produced wrench.

    Column j is the wrench contributed by one unit of nominal thrust command on
    rotor j. Thrust coefficient error scales the produced force directly; arm
    error scales the roll/pitch moment arms; moment-constant error scales yaw
    drag only.
    """
    arm = NOMINAL_ARM * arm_scale
    thrust = kf_scale                       # produced force per unit command
    return jnp.stack([
        thrust,                                        # collective f
        thrust * arm * _SIGN_Y,                        # Mx
        -thrust * arm * _SIGN_X,                       # My
        thrust * KM_RATIO * moment_scale * _SPIN,      # Mz
    ])


B_NOM = allocation_matrix(jnp.ones(4), jnp.ones(4), jnp.ones(4))
B_NOM_PINV = jnp.linalg.pinv(B_NOM)


def apply(f_cmd, M_cmd, params: MixerParams, use_true_allocation=False):
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

    Returns (f_actual, M_actual, motor_cmd, saturated).
    """
    w_cmd = jnp.concatenate([jnp.atleast_1d(f_cmd), M_cmd])
    B_true_alloc = allocation_matrix(params.kf_scale, params.moment_scale,
                                     params.arm_scale)
    T_cmd = jnp.where(use_true_allocation,
                      jnp.linalg.pinv(B_true_alloc) @ w_cmd,
                      B_NOM_PINV @ w_cmd)
    T_sat = jnp.clip(T_cmd, 0.0, MAX_MOTOR_THRUST)
    saturated = jnp.any(jnp.abs(T_sat - T_cmd) > 1e-9)

    B_true = allocation_matrix(params.kf_scale, params.moment_scale, params.arm_scale)
    w_actual = B_true @ T_sat
    return w_actual[0], w_actual[1:4], T_sat, saturated


def apply_nominal(f_cmd, M_cmd):
    """Allocation round trip with the nominal model on both sides.

    Used for the disturbance-free reference twin, so it experiences the same
    actuator saturation as the true plant but no parameter error.
    """
    return apply(f_cmd, M_cmd, MixerParams.nominal())
