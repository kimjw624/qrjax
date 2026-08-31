"""Geometric SE(3) tracking controller in JAX, covering both PD and PID.

One implementation serves both baselines. The PD controller is exactly the PID
controller with ``ki = kI = 0``:

    A = kx ex + kv ev + ki ei + m g e3 - m ad
    M = -kR eR - kOmega eOmega - kI eI + <Lee (2010) feedforward>

with the integral states

    ei_dot = ev + c1 * ex        (saturated elementwise to +-ei_limit)
    eI_dot = eOmega + c2 * eR    (unsaturated, matching the Simulink diagram)

That collapse is worth stating explicitly: it means PD-vs-PID is a gain
setting, not two code paths, so there is no way for the two to drift apart.
``Gains.pd()`` and ``Gains.pid()`` are the only difference between the arms.

Sign convention. This follows the NED formulation of the reference NumPy
implementation, where ``A = +kx ex + kv ev + ...``. The MATLAB/Simulink source
defines ``A`` with the opposite sign, so the integral term appears there as
``-ki ei``. Both describe the same physical controller; the moment convention
is identical in both.

The learned residual is applied downstream of this controller, as a wrench
correction before control allocation. This module has no knowledge of it.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .so3 import E3, hat, normalize, project_to_so3, rotation_error, so3_log, vee


class Gains(NamedTuple):
    """Controller gains. A pytree, so it can be closed over or passed to jit.

    kx/kv are mass-scaled to match the reference tuning. ki and kI default to
    kx/10 and kR/10, reproducing the supplied Simulink Constants block
    (ki = 3.6, kI = 0.881 at m = 2 kg).
    """
    mass: float
    gravity: float
    dt: float
    J: jnp.ndarray
    kx: float
    kv: float
    kR: float
    kOmega: float
    ki: float
    kI: float
    c1: float
    c2: float
    ei_limit: float

    @staticmethod
    def _base(mass, gravity, dt, J, ki, kI):
        return Gains(
            mass=float(mass), gravity=float(gravity), dt=float(dt),
            J=(jnp.asarray(J, dtype=jnp.float32)
               if np.ndim(J) == 2
               else jnp.diag(jnp.asarray(J, dtype=jnp.float32))),
            kx=float(mass) * 18.0, kv=float(mass) * 10.0,
            kR=8.81, kOmega=2.54,
            ki=float(ki), kI=float(kI), c1=5.0, c2=0.5, ei_limit=1.0,
        )

    @classmethod
    def pd(cls, mass=2.0, gravity=9.807, dt=0.01, J=(0.022, 0.022, 0.04)):
        """Legacy PD baseline: both integral gains zero."""
        return cls._base(mass, gravity, dt, J, ki=0.0, kI=0.0)

    @classmethod
    def pid(cls, mass=2.0, gravity=9.807, dt=0.01, J=(0.022, 0.022, 0.04),
            ki=None, kI=None):
        """PID baseline. ``None`` selects the reference tuning ki=kx/10, kI=kR/10."""
        ki = (mass * 18.0) / 10.0 if ki is None else ki
        kI = 8.81 / 10.0 if kI is None else kI
        return cls._base(mass, gravity, dt, J, ki=ki, kI=kI)

    @classmethod
    def make(cls, kind="pd", **kwargs):
        kind = str(kind).lower()
        if kind == "pd":
            return cls.pd(**kwargs)
        if kind == "pid":
            return cls.pid(**kwargs)
        raise ValueError(f"base_controller must be 'pd' or 'pid', got {kind!r}")


class CtrlState(NamedTuple):
    """Controller memory carried between steps.

    ``started`` replaces the reference implementation's ``Rd_prev is None``
    check, which cannot be a traced branch. Before the first step the desired
    angular velocity and its derivative are forced to zero so the finite
    difference does not emit an impulse out of the reset.
    """
    Rd_prev: jnp.ndarray
    omega_d_prev: jnp.ndarray
    started: jnp.ndarray
    ei: jnp.ndarray
    eI: jnp.ndarray


def ctrl_init() -> CtrlState:
    return CtrlState(
        Rd_prev=jnp.eye(3),
        omega_d_prev=jnp.zeros(3),
        started=jnp.array(False),
        ei=jnp.zeros(3),
        eI=jnp.zeros(3),
    )


class CtrlInfo(NamedTuple):
    """Diagnostics. Kept small: everything here is materialized per env per step."""
    ex: jnp.ndarray
    ev: jnp.ndarray
    eR: jnp.ndarray
    eOmega: jnp.ndarray
    A_cmd: jnp.ndarray
    ei: jnp.ndarray
    eI: jnp.ndarray


def control(cs: CtrlState, state, desired, g: Gains, hold_integral=False):
    """One controller evaluation.

    Parameters
    ----------
    cs:
        Controller memory from the previous step.
    state:
        A ``RigidState``.
    desired:
        ``(xd, vd, ad, b1d)`` as produced by :mod:`qrjax.core.trajectory`.
    hold_integral:
        Traced boolean. Freezes both accumulators for this step, used for
        anti-windup while the actuators are saturated.

    Returns
    -------
    (f, M, new_ctrl_state, info)
    """
    xd, vd, ad, b1d = desired

    ex = state.x - xd
    ev = state.v - vd

    hold = jnp.asarray(hold_integral)

    # --- translational integral ---
    ei_next = jnp.clip(cs.ei + (ev + g.c1 * ex) * g.dt, -g.ei_limit, g.ei_limit)
    ei = jnp.where(hold, cs.ei, ei_next)

    A = (g.kx * ex + g.kv * ev + g.ki * ei
         + g.mass * g.gravity * E3 - g.mass * ad)

    # Collective thrust is the projection of A onto the current body z axis.
    f = jnp.maximum(0.0, A @ (state.R @ E3))

    # --- desired attitude: body z aligns with A, heading fixed by b1d ---
    b3d = normalize(A, E3)
    b2d = normalize(jnp.cross(b3d, b1d), jnp.array([0.0, 1.0, 0.0]))
    b1d_real = jnp.cross(b2d, b3d)
    Rd = project_to_so3(jnp.stack([b1d_real, b2d, b3d], axis=1))

    # --- desired rates by finite difference of Rd ---
    omega_d_raw = vee(so3_log(cs.Rd_prev.T @ Rd)) / g.dt
    omega_d = jnp.where(cs.started, omega_d_raw, jnp.zeros(3))
    omega_d_dot = jnp.where(
        cs.started, (omega_d - cs.omega_d_prev) / g.dt, jnp.zeros(3)
    )

    eR = rotation_error(state.R, Rd)
    eOmega = state.omega - state.R.T @ Rd @ omega_d

    # --- attitude integral ---
    eI_next = cs.eI + (eOmega + g.c2 * eR) * g.dt
    eI = jnp.where(hold, cs.eI, eI_next)

    RtRd = state.R.T @ Rd
    M = (
        -g.kR * eR
        - g.kOmega * eOmega
        + jnp.cross(state.omega, g.J @ state.omega)
        - g.J @ (hat(state.omega) @ RtRd @ omega_d - RtRd @ omega_d_dot)
        - g.kI * eI
    )

    new_cs = CtrlState(
        Rd_prev=Rd, omega_d_prev=omega_d, started=jnp.array(True), ei=ei, eI=eI
    )
    info = CtrlInfo(ex=ex, ev=ev, eR=eR, eOmega=eOmega, A_cmd=A, ei=ei, eI=eI)
    return f, M, new_cs, info
