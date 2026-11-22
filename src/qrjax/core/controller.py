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
    def _base(mass, gravity, dt, J, ki, kI, ei_limit=1.0):
        return Gains(
            mass=float(mass), gravity=float(gravity), dt=float(dt),
            J=(jnp.asarray(J, dtype=jnp.float32)
               if np.ndim(J) == 2
               else jnp.diag(jnp.asarray(J, dtype=jnp.float32))),
            kx=float(mass) * 18.0, kv=float(mass) * 10.0,
            kR=8.81, kOmega=2.54,
            ki=float(ki), kI=float(kI), c1=5.0, c2=0.5, ei_limit=float(ei_limit),
        )

    @classmethod
    def pd(cls, mass=2.0, gravity=9.807, dt=0.01, J=(0.022, 0.022, 0.04)):
        """Legacy PD baseline: both integral gains zero."""
        return cls._base(mass, gravity, dt, J, ki=0.0, kI=0.0)

    @classmethod
    def pid(cls, mass=2.0, gravity=9.807, dt=0.01, J=(0.022, 0.022, 0.04),
            ki=None, kI=None, ei_limit=1.0):
        """PID baseline. ``None`` selects the reference tuning ki=kx/10, kI=kR/10.

        ``ei_limit`` clips the translational integral state (anti-windup); the
        reference tuning uses 1.0.
        """
        ki = (mass * 18.0) / 10.0 if ki is None else ki
        kI = 8.81 / 10.0 if kI is None else kI
        return cls._base(mass, gravity, dt, J, ki=ki, kI=kI, ei_limit=ei_limit)

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

    ``n_calls`` (0, 1, 2, saturating) counts how many Rd samples exist. It is
    used only with ``fd_counter_fix=True``: omega_d needs one previous Rd, but
    omega_d_dot needs one previous omega_d that was itself a real difference,
    i.e. two previous samples. Gating both on ``started`` (the original
    behaviour) makes the second call compute omega_d_dot = omega_d / dt, a
    one-step impulse after every reset.
    """
    Rd_prev: jnp.ndarray
    omega_d_prev: jnp.ndarray
    started: jnp.ndarray
    ei: jnp.ndarray
    eI: jnp.ndarray
    n_calls: jnp.ndarray


def ctrl_init() -> CtrlState:
    return CtrlState(
        Rd_prev=jnp.eye(3),
        omega_d_prev=jnp.zeros(3),
        started=jnp.array(False),
        ei=jnp.zeros(3),
        eI=jnp.zeros(3),
        n_calls=jnp.int32(0),
    )


def ctrl_warm(Rd, omega_d, omega_d_dot, dt) -> CtrlState:
    """Controller memory of a vehicle that has already been flying the reference.

    Used with ``start_on_reference``: the finite-difference memory holds the
    previous reference attitude and rate (first order in dt), integrators are
    zero, and both derivatives are valid from the first call.
    """
    Rd_prev = project_to_so3(Rd @ (jnp.eye(3) - dt * hat(omega_d)))
    return CtrlState(
        Rd_prev=Rd_prev,
        omega_d_prev=omega_d - dt * omega_d_dot,
        started=jnp.array(True),
        ei=jnp.zeros(3),
        eI=jnp.zeros(3),
        n_calls=jnp.int32(2),
    )


# --------------------------------------------------- analytic attitude rates
def _unit_derivs(u, du, ddu):
    """(n, n_dot, n_ddot) for n = u/|u| given u and its first two derivatives."""
    r = jnp.sqrt(jnp.dot(u, u))
    ud = jnp.dot(u, du)
    n = u / r
    dn = du / r - u * ud / r ** 3
    ddn = (ddu / r - 2.0 * du * ud / r ** 3
           - u * (jnp.dot(du, du) + jnp.dot(u, ddu)) / r ** 3
           + 3.0 * u * ud ** 2 / r ** 5)
    return n, dn, ddn


def attitude_from_thrust(A, dA, ddA, b1d):
    """Rd, omega_d, omega_d_dot from the thrust-vector command and its derivatives.

    Rd = [b1c, b2d, b3d] with b3d = A/|A|, b2d = unit(b3d x b1d), b1c = b2d x b3d
    (b1d constant). omega_d = vee(Rd^T Rd_dot), omega_d_dot = vee(Rd^T Rd_ddot -
    hat(omega_d)^2). Ported unchanged from qrjax_ood (qrood.core.controller).
    """
    b3, db3, ddb3 = _unit_derivs(A, dA, ddA)
    C, dC, ddC = jnp.cross(b3, b1d), jnp.cross(db3, b1d), jnp.cross(ddb3, b1d)
    b2, db2, ddb2 = _unit_derivs(C, dC, ddC)
    b1 = jnp.cross(b2, b3)
    db1 = jnp.cross(db2, b3) + jnp.cross(b2, db3)
    ddb1 = jnp.cross(ddb2, b3) + 2.0 * jnp.cross(db2, db3) + jnp.cross(b2, ddb3)
    Rd = jnp.stack([b1, b2, b3], axis=1)
    dRd = jnp.stack([db1, db2, db3], axis=1)
    ddRd = jnp.stack([ddb1, ddb2, ddb3], axis=1)
    W = Rd.T @ dRd
    omega_d = vee(0.5 * (W - W.T))
    Wh = hat(omega_d)
    X = Rd.T @ ddRd - Wh @ Wh
    omega_d_dot = vee(0.5 * (X - X.T))
    return Rd, omega_d, omega_d_dot


def reference_attitude(g, desired, derivs):
    """Attitude and exact body rates of the reference at ZERO tracking error.

    A = m (g e3 - ad), A_dot = -m jd, A_ddot = -m sd. Used only to place the
    vehicle on the reference at episode start (``start_on_reference``).
    """
    _, _, ad, b1d = desired
    jd, sd = derivs
    m = g.mass
    return attitude_from_thrust(m * g.gravity * E3 - m * ad, -m * jd, -m * sd, b1d)


class CtrlInfo(NamedTuple):
    """Diagnostics. Kept small: everything here is materialized per env per step."""
    ex: jnp.ndarray
    ev: jnp.ndarray
    eR: jnp.ndarray
    eOmega: jnp.ndarray
    A_cmd: jnp.ndarray
    ei: jnp.ndarray
    eI: jnp.ndarray
    omega_d: jnp.ndarray
    omega_d_dot: jnp.ndarray


def control(cs: CtrlState, state, desired, g: Gains, hold_integral=False,
            rate_ff="finite_difference", fd_counter_fix=False, accel=None,
            derivs=None):
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
    rate_ff:
        Static. How the desired body rate omega_d and its derivative are
        obtained.

        ``"finite_difference"`` (default, the original behaviour):
        omega_d = log(Rd_prev^T Rd)/dt, omega_d_dot = (omega_d - omega_d_prev)/dt.
        Rd contains the feedback terms, so any step-to-step change in the state
        reaches the moment command divided by dt^2.

        ``"analytic"``: exact time derivatives of Rd from A, A_dot, A_ddot
        (Lee et al. 2010), as in qrjax_ood. Needs ``derivs = (jerk, snap)`` of
        the reference and ``accel``, the measured NED acceleration (an ideal
        accelerometer); if ``accel`` is None the nominal-model prediction
        g e3 - (f/m) R e3 is used, which is biased by every unmodelled force.

        ``"none"`` (diagnostic only): omega_d = omega_d_dot = 0, i.e. the attitude
        loop tracks Rd with no rate feedforward.
    fd_counter_fix:
        Static, finite-difference mode only. Gate omega_d_dot on two previous
        samples instead of one (see ``CtrlState``).

    Returns
    -------
    (f, M, new_ctrl_state, info)
    """
    xd, vd, ad, b1d = desired

    ex = state.x - xd
    ev = state.v - vd

    hold = jnp.asarray(hold_integral)

    # --- translational integral ---
    ei_raw = cs.ei + (ev + g.c1 * ex) * g.dt
    ei_next = jnp.clip(ei_raw, -g.ei_limit, g.ei_limit)
    ei = jnp.where(hold, cs.ei, ei_next)

    A = (g.kx * ex + g.kv * ev + g.ki * ei
         + g.mass * g.gravity * E3 - g.mass * ad)

    # Collective thrust is the projection of A onto the current body z axis.
    f = jnp.maximum(0.0, A @ (state.R @ E3))

    if rate_ff == "none":
        # diagnostic: no desired-rate feedforward at all (omega_d = omega_d_dot = 0)
        b3d = normalize(A, E3)
        b2d = normalize(jnp.cross(b3d, b1d), jnp.array([0.0, 1.0, 0.0]))
        Rd = project_to_so3(jnp.stack([jnp.cross(b2d, b3d), b2d, b3d], axis=1))
        omega_d = jnp.zeros(3)
        omega_d_dot = jnp.zeros(3)
    elif rate_ff == "finite_difference":
        # --- desired attitude: body z aligns with A, heading fixed by b1d ---
        b3d = normalize(A, E3)
        b2d = normalize(jnp.cross(b3d, b1d), jnp.array([0.0, 1.0, 0.0]))
        b1d_real = jnp.cross(b2d, b3d)
        Rd = project_to_so3(jnp.stack([b1d_real, b2d, b3d], axis=1))

        # --- desired rates by finite difference of Rd ---
        omega_d_raw = vee(so3_log(cs.Rd_prev.T @ Rd)) / g.dt
        omega_d = jnp.where(cs.started, omega_d_raw, jnp.zeros(3))
        dot_ok = (cs.n_calls >= 2) if fd_counter_fix else cs.started
        omega_d_dot = jnp.where(
            dot_ok, (omega_d - cs.omega_d_prev) / g.dt, jnp.zeros(3)
        )
    elif rate_ff == "analytic":
        if derivs is None:
            raise ValueError("rate_ff='analytic' needs derivs=(jerk, snap)")
        jd, sd = derivs
        m = g.mass
        b3 = state.R @ E3
        # integrator derivatives are zero on axes that are held or clipped
        active = jnp.logical_and(jnp.logical_not(hold),
                                 jnp.abs(ei_raw) < g.ei_limit)
        a_model = g.gravity * E3 - (f / m) * b3
        a_used = a_model if accel is None else accel
        ea = a_used - ad
        dei = jnp.where(active, ev + g.c1 * ex, 0.0)
        dA = g.kx * ev + g.kv * ea + g.ki * dei - m * jd
        db3 = state.R @ hat(state.omega) @ E3              # gyro
        df = dA @ b3 + A @ db3
        # jerk is not measurable: its model part is used in both cases
        da_m = -(df / m) * b3 - (f / m) * db3
        dea = da_m - jd
        ddei = jnp.where(active, ea + g.c1 * ev, 0.0)
        ddA = g.kx * ea + g.kv * dea + g.ki * ddei - m * sd
        Rd, omega_d, omega_d_dot = attitude_from_thrust(A, dA, ddA, b1d)
    else:
        raise ValueError(f"unknown rate_ff {rate_ff!r}")

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
        Rd_prev=Rd, omega_d_prev=omega_d, started=jnp.array(True), ei=ei, eI=eI,
        n_calls=jnp.minimum(cs.n_calls + 1, 2).astype(jnp.int32),
    )
    info = CtrlInfo(ex=ex, ev=ev, eR=eR, eOmega=eOmega, A_cmd=A, ei=ei, eI=eI,
                    omega_d=omega_d, omega_d_dot=omega_d_dot)
    return f, M, new_cs, info
