"""Observation construction.

The policy never sees absolute state. It sees the DISCREPANCY between the true
(disturbed) plant and a disturbance-free nominal twin running the same
controller on its own state. That discrepancy is physically centred at zero, so
it is normalized by fixed physical scales rather than a running mean/std --
which also means the observation statistics never shift under a curriculum.

Two modes:

``history``
    H stacked error frames interleaved with the H-1 most recent normalized
    actions: ``[e_0, a_0, e_1, a_1, ..., a_{H-2}, e_{H-1}]``, oldest first.
    The current action is deliberately absent (it does not exist yet when the
    observation is built). Dimension: H*12 + (H-1)*action_dim.

``pid``
    Proportional / leaky-integral / derivative of the same 12-D error, plus the
    normalized base wrench: 36 + 4 = 40. Compact and Markov-ish without needing
    a long history.

Both keep their rolling buffers inside the env state, so they are pure
functions of that state and survive scan.
"""

from typing import NamedTuple

import jax.numpy as jnp

from ..core.so3 import rotation_error
from ..core.mixer import WRENCH_MAX


class ObsState(NamedTuple):
    """Rolling observation memory. Shapes are fixed at build time."""
    err_hist: jnp.ndarray       # (H, 12) oldest first
    act_hist: jnp.ndarray       # (H-1, action_dim) oldest first
    integral: jnp.ndarray       # (12,) leaky integral, "pid" mode
    prev_err: jnp.ndarray       # (12,) for the derivative, "pid" mode
    u_base: jnp.ndarray         # (4,) most recent base wrench, "pid" mode


def obs_init(history: int, action_dim: int) -> ObsState:
    return ObsState(
        err_hist=jnp.zeros((history, 12)),
        act_hist=jnp.zeros((max(history - 1, 1), action_dim)),
        integral=jnp.zeros(12),
        prev_err=jnp.zeros(12),
        u_base=jnp.zeros(4),
    )


def discrepancy(nom, true, cfg) -> jnp.ndarray:
    """Normalized 12-D twin discrepancy: position, velocity, attitude, rate.

    Sign convention is TRUE minus NOMINAL -- actual relative to reference --
    matching the geometric controller's own convention (e_x = x - x_d in Lee
    et al.), so the same symbol means the same direction everywhere in the
    system and in the write-up.

    Attitude is the exception, and not by choice: SO(3) is not a vector space,
    so R - R_nom is not a rotation and carries no meaning. The geometric error

        e_R = 1/2 ( R_nom^T R - R^T R_nom )^vee

    is used instead, which is the same map the controller applies, with the
    nominal attitude in the role of the reference.
    """
    return jnp.concatenate([
        (true.x - nom.x) / cfg.obs_pos_scale,
        (true.v - nom.v) / cfg.obs_vel_scale,
        rotation_error(true.R, nom.R) / cfg.obs_att_scale,
        (true.omega - nom.omega) / cfg.obs_omega_scale,
    ])


def obs_push(os: ObsState, err, action_norm, u_base, cfg) -> ObsState:
    """Shift the rolling buffers by one and insert the newest frame."""
    err_hist = jnp.roll(os.err_hist, shift=-1, axis=0).at[-1].set(err)
    act_hist = jnp.roll(os.act_hist, shift=-1, axis=0).at[-1].set(action_norm)
    integral = cfg.pid_integral_leak * os.integral + err * cfg.dt
    return ObsState(
        err_hist=err_hist,
        act_hist=act_hist,
        integral=integral,
        prev_err=err,
        u_base=u_base,
    )


def obs_vector(os: ObsState, cfg) -> jnp.ndarray:
    """Flatten the rolling state into the policy observation."""
    if cfg.obs_mode == "history":
        H = cfg.history
        # Interleave [e_0, a_0, e_1, a_1, ..., a_{H-2}, e_{H-1}].
        parts = []
        for i in range(H - 1):
            parts.append(os.err_hist[i])
            parts.append(os.act_hist[i])
        parts.append(os.err_hist[H - 1])
        return jnp.concatenate(parts)

    if cfg.obs_mode == "pid":
        err = os.err_hist[-1]
        prev = os.err_hist[-2]
        derivative = (err - prev) / cfg.dt
        return jnp.concatenate([
            err,
            jnp.clip(os.integral, -10.0, 10.0),
            jnp.clip(derivative, -50.0, 50.0),
            os.u_base / WRENCH_MAX,
        ])

    raise ValueError(f"unknown obs_mode {cfg.obs_mode!r}")
