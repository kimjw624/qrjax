"""v2: defaults are the fixed code; the paper's code stays reproducible.

Checks that
  * EnvConfig() / TrainConfig() defaults are the fixes at the paper's task settings,
  * legacy_env() is exactly the pre-v2 default and paper_env() the paper's setting,
  * a run config saved before the fix switches existed loads with their ORIGINAL
    values (old checkpoints keep flying the code they were trained on),
  * paper_env() on this code reproduces a rollout of the submitted code (tag
    paper-submission) -- reference numbers below were produced by that commit,
  * the three switches added in v2 (smooth integrator bound, disturbance ramp-in,
    filter time constant) and the feasibility label do what they claim.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrjax.core.controller import integrator_projection
from qrjax.envs import Config, EnvConfig, TrainConfig, VecEnv, legacy_env, paper_env
from qrjax.envs.config import (ALL_DISTURBANCES, FIXED_ENV, FIXED_TRAIN, LEGACY_ENV,
                               LEGACY_TRAIN, PAPER_ENV, paper_config)
from qrjax.envs.feasibility import feasibility_margin
from qrjax.rl.curriculum import flat_ranges


# ------------------------------------------------------------------ defaults
def test_v2_defaults_are_fixed_code_at_paper_settings():
    e, t = EnvConfig(), TrainConfig()
    assert all(getattr(e, k) == v for k, v in FIXED_ENV.items())
    assert all(getattr(t, k) == v for k, v in FIXED_TRAIN.items())
    assert (e.dt, e.episode_steps, e.base_controller, tuple(e.disturbances)) == \
        (0.005, 2000, "pid", ALL_DISTURBANCES)
    assert t.use_curriculum is False
    # the switches added in v2 are off by default
    assert (e.pid_ei_mode, e.disturbance_ramp_s, e.thrust_filter_tau,
            e.moment_filter_tau) == ("clip", 0.0, 0.0, 0.0)


def test_legacy_env_is_the_pre_v2_default():
    d = legacy_env()
    assert (d.dt, d.episode_steps, d.base_controller, tuple(d.disturbances)) == \
        (0.01, 1000, "pd", ("massmoi",))
    assert (d.rate_ff, d.fd_counter_fix, d.allocator, d.mixer_geometry,
            d.start_on_reference, d.term_ref_error, d.term_heading_deg) == \
        ("finite_difference", False, "clip", "legacy", False, 0.0, 0.0)
    p = paper_env()
    assert (p.dt, p.episode_steps, p.base_controller) == (0.005, 2000, "pid")
    assert p.rate_ff == "finite_difference" and p.allocator == "clip"
    assert paper_env(base_controller="pd").base_controller == "pd"
    c = paper_config()
    assert all(getattr(c.train, k) == v for k, v in {**LEGACY_TRAIN,
                                                     "use_curriculum": False}.items())
    with pytest.raises(KeyError):
        legacy_env(not_a_field=1)


def test_old_saved_config_loads_with_original_switch_values():
    # a config written before the switches existed: no switch keys at all
    old = Config(env=legacy_env()).to_dict()
    for k in LEGACY_ENV:
        if k not in ("dt", "episode_steps", "base_controller", "disturbances"):
            old["env"].pop(k)
    for k in ("eval_fixed_key", "best_window"):
        old["train"].pop(k)
    old["env"]["dt"], old["env"]["base_controller"] = 0.005, "pid"
    cfg = Config.from_dict(old)
    for k, v in LEGACY_ENV.items():
        if k not in ("dt", "episode_steps", "base_controller", "disturbances"):
            assert getattr(cfg.env, k) == v, k
    assert cfg.env.dt == 0.005 and cfg.env.base_controller == "pid"
    assert cfg.train.eval_fixed_key is False and cfg.train.best_window == 1
    # a v2 config round-trips unchanged
    new = Config()
    assert Config.from_dict(new.to_dict()).to_dict() == new.to_dict()


# Produced by the submitted code (git tag paper-submission, c9d957f) with
# EnvConfig(); dt = 0.005; episode_steps = 2000; base_controller = "pid", and
# by this code with paper_env(): identical bit for bit on CPU.
PAPER_POS_ERR = [0.0054426924, 0.040003225, 0.08541429, 0.12718734, 0.13213515]
PAPER_RETURN = [101.559135, 230.12474, 244.75623, 184.21388]


def test_paper_env_reproduces_paper_code_fingerprint():
    cfg = paper_env()
    env = VecEnv(cfg, 16)
    ranges = env.broadcast_ranges(flat_ranges(cfg))
    state, obs = env.reset(jax.random.PRNGKey(7), ranges, stagger=False)

    def body(c, _):
        s, o = c
        a = 0.3 * jnp.tanh(o[:, :4] * 5.0)          # deterministic, state-dependent
        s, o, r, d, info = env.step(s, a, ranges)
        return (s, o), (info["pos_err_desired"], r)
    _, (pe, r) = jax.lax.scan(body, (state, obs), None, length=300)
    pe, r = np.asarray(pe), np.asarray(r)
    np.testing.assert_allclose(pe[[0, 49, 99, 199, 299]].mean(axis=1), PAPER_POS_ERR,
                               rtol=1e-4)
    np.testing.assert_allclose(r.sum(axis=0)[:4], PAPER_RETURN, rtol=1e-4)


# ------------------------------------------------------------------ v2 switches
def test_integrator_projection_is_continuous_and_bounded():
    L, eps = 1.0, 0.2
    # continuous across the start of the projection zone and across y = 0
    e0 = L / np.sqrt(1 + eps)
    for ei in (e0 - 1e-6, e0 + 1e-6):
        r, _, _ = integrator_projection(jnp.float32(ei), jnp.float32(1.0), L, eps)
        assert abs(float(r) - 1.0) < 1e-4
    r_out, _, _ = integrator_projection(jnp.float32(0.99 * L), jnp.float32(1e-6), L, eps)
    assert abs(float(r_out)) < 1e-6
    # inward motion is never slowed; outward motion stops at the bound
    r_in, _, _ = integrator_projection(jnp.float32(L), jnp.float32(-1.0), L, eps)
    r_edge, _, _ = integrator_projection(jnp.float32(L), jnp.float32(1.0), L, eps)
    assert float(r_in) == -1.0 and abs(float(r_edge)) < 1e-6
    # a long push stays (essentially) inside the bound
    ei = jnp.float32(0.0)
    for _ in range(5000):
        r, _, _ = integrator_projection(ei, jnp.float32(3.0), L, eps)
        ei = ei + 0.005 * r
    assert 0.95 * L < float(ei) <= L * (1 + 1e-3)


def _windup_rollout(**kw):
    """Heavy plant + large force: the translational integrator reaches its bound."""
    cfg = EnvConfig(**kw)
    cfg.k_min, cfg.k_max = 1.25, 1.3
    cfg.external_force_max = 5.0
    cfg.episode_steps = 800
    cfg = cfg.validate()
    n = 16
    env = VecEnv(cfg, n)
    ranges = env.broadcast_ranges(flat_ranges(cfg))
    state, obs = env.reset(jax.random.PRNGKey(5), ranges, stagger=False)

    def body(c, _):
        s, o = c
        s, o, r, d, info = env.step(s, jnp.zeros((n, 4)), ranges)
        return (s, o), (info["u_base"], info["ctrl_ei"], info["saturated"])
    _, (u, ei, sat) = jax.lax.scan(body, (state, obs), None, length=cfg.episode_steps)
    return cfg, np.asarray(u), np.asarray(ei), np.asarray(sat)


def test_smooth_integrator_removes_the_onset_kick():
    worst = {}
    for mode in ("clip", "smooth"):
        cfg, u, ei, _ = _windup_rollout(pid_ei_mode=mode)
        dM = np.linalg.norm(np.diff(u[:, :, 1:], axis=0), axis=-1)
        at = (np.abs(ei) >= 0.999 * cfg.pid_ei_limit).any(-1)
        onset = at[1:] & ~at[:-1]
        onset[: int(0.3 / cfg.dt)] = False                  # skip the start transient
        assert onset.any(), mode
        worst[mode] = float(dM[onset].max())
        assert np.abs(ei).max() <= cfg.pid_ei_limit * (1 + 1e-3)
    assert worst["smooth"] < 0.1 * worst["clip"], worst


def test_disturbance_ramp_removes_start_saturation():
    _, _, _, sat0 = _windup_rollout()
    cfg, _, _, sat1 = _windup_rollout(disturbance_ramp_s=0.5)
    n0 = int(0.1 / cfg.dt)
    assert sat0[:n0].sum() > 0 and sat1[:n0].sum() == 0


def test_disturbance_ramp_reaches_the_true_plant():
    # after the ramp the dynamics are the sampled plant's: same accelerations at
    # the same state as with no ramp
    from qrjax.envs.residual_env import make_env
    cfg0 = EnvConfig(disturbance_ramp_s=0.0)
    cfg1 = EnvConfig(disturbance_ramp_s=0.2)
    ranges = flat_ranges(cfg0)
    (reset0, _, step0), (_, _, step1) = make_env(cfg0), make_env(cfg1)
    s = reset0(jax.random.PRNGKey(2), ranges)
    s = s._replace(ep_length=jnp.int32(1000))                # well past the ramp
    a = jnp.zeros(4)
    n0, *_ = step0(s, a, ranges)
    n1, *_ = step1(s, a, ranges)
    np.testing.assert_allclose(np.asarray(n0.true.v), np.asarray(n1.true.v), atol=1e-6)


def test_filter_time_constant_equals_the_beta_it_implies():
    cfg_b = EnvConfig(thrust_filter_beta=0.2)
    tau = -cfg_b.dt / np.log(1.0 - 0.2)                      # 22.4 ms at 200 Hz
    cfg_t = EnvConfig(thrust_filter_beta=0.9, thrust_filter_tau=float(tau))
    outs = []
    for cfg in (cfg_b, cfg_t):
        env = VecEnv(cfg, 4)
        ranges = env.broadcast_ranges(flat_ranges(cfg))
        state, obs = env.reset(jax.random.PRNGKey(1), ranges, stagger=False)
        acts = jnp.tanh(jax.random.normal(jax.random.PRNGKey(9), (50, 4, 4)))
        for a in acts:
            state, obs, *_ = env.step(state, a, ranges)
        outs.append(np.asarray(state.prev_action_norm))
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-5)


def test_feasibility_margin_flags_heavy_weak_rotor_episodes():
    cfg = EnvConfig()
    env = VecEnv(cfg, 2)
    ranges = env.broadcast_ranges(flat_ranges(cfg))
    state, _ = env.reset(jax.random.PRNGKey(0), ranges, stagger=False)
    nominal = state._replace(
        inertia_true=state.inertia_true._replace(mass=jnp.full(2, cfg.mass_nom)),
        mixer_true=jax.tree.map(jnp.ones_like, state.mixer_true),
        external_force=jnp.zeros((2, 3)))
    heavy_weak = nominal._replace(
        inertia_true=nominal.inertia_true._replace(mass=jnp.full(2, 1.3 * cfg.mass_nom)),
        mixer_true=nominal.mixer_true._replace(kf_scale=jnp.full((2, 4), 0.7)))
    assert (np.asarray(feasibility_margin(cfg, nominal)) > 0.2).all()
    assert (np.asarray(feasibility_margin(cfg, heavy_weak)) < 0.0).all()
