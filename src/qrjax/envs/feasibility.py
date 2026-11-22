"""Quasi-static feasibility of an episode: an ORACLE label for failure analysis only.

Port of qrjax_ood's ``qrood.eval.feasibility`` to this repository's plant model
(mass/inertia factor k, per-rotor thrust/moment/arm scales, a constant or
oscillating external force).

For the true parameters of an episode and sample times t over one episode:

    thrust vector needed   f_vec(t) = m_true (g e3 - a_d(t)) + F_ext(t)      (NED)
    rotor commands needed  T(t) = B_true^-1 [ |f_vec(t)|, 0, 0, 0 ]

T(t) are the commands that hold the reference acceleration with zero moment.
The margin is

    margin = min_t min_j  min(T_j(t), T_max - T_j(t)) / T_max

margin < 0: no admissible rotor command can fly the reference even
quasi-statically, so any controller fails; the episode is outside what the
vehicle can do, not a controller failure. 0 <= margin < MARGINAL: flyable only
with almost no authority left. It ignores attitude dynamics and transients, so
it is a NECESSARY condition only. It uses the true parameters and is never
visible to the controller or the policy.

The fixes study found 9 of the 512 evaluation conditions failing in every arm,
all heavy with weak rotors; this is the label that identifies such episodes.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..core import mixer
from ..core.so3 import E3
from ..core.trajectory import make_trajectory

MARGINAL = 0.05
FEAS_INFEASIBLE, FEAS_MARGINAL, FEAS_OK = 0, 1, 2
FEAS_NAMES = {FEAS_INFEASIBLE: "infeasible", FEAS_MARGINAL: "marginal", FEAS_OK: "feasible"}


def _traj_kwargs(cfg):
    # same rule as residual_env.make_env
    if cfg.trajectory == "circle":
        return {"radius": cfg.traj_radius, "speed": cfg.traj_speed, "z0": cfg.traj_z0}
    if cfg.trajectory in ("figure8", "hover"):
        return {"z0": cfg.traj_z0}
    return {}


def feasibility_margin(cfg, state, n_samples=400):
    """Margin per episode for a BATCH of env states (leading axis = episodes).

    ``state`` is an ``EnvState`` from ``VecEnv.reset`` (only its true-plant
    parameters are read: inertia_true, mixer_true, external_force, force_freq,
    force_phase).
    """
    traj = make_trajectory(cfg.trajectory, **_traj_kwargs(cfg))
    ts = jnp.linspace(0.0, cfg.episode_steps * cfg.dt, n_samples)
    Tm = mixer.MAX_MOTOR_THRUST
    geometry = getattr(cfg, "mixer_geometry", "legacy")
    g = cfg.gravity

    def one(mass, kf, mom, arm, amp, freq, phase):
        B = mixer.allocation_matrix(kf, mom, arm, geometry)
        A = amp * jnp.where(freq > 0.0, jnp.sqrt(2.0), 1.0)

        def at(t):
            ad = traj(t)[2]
            F = A * jnp.cos(2.0 * jnp.pi * freq * t + phase)
            f_vec = mass * (g * E3 - ad) + F
            T = jnp.linalg.solve(B, jnp.array([jnp.linalg.norm(f_vec), 0.0, 0.0, 0.0]))
            return jnp.min(jnp.minimum(T, Tm - T)) / Tm
        return jnp.min(jax.vmap(at)(ts))

    m = state.mixer_true
    return jax.jit(jax.vmap(one))(state.inertia_true.mass, m.kf_scale, m.moment_scale,
                                  m.arm_scale, state.external_force, state.force_freq,
                                  state.force_phase)


def feasibility_class(margin):
    m = np.asarray(margin)
    return np.where(m < 0, FEAS_INFEASIBLE, np.where(m < MARGINAL, FEAS_MARGINAL, FEAS_OK))
