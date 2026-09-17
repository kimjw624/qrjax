"""Test suite.

Run with:  PYTHONPATH=src python -m pytest tests/ -v

The parity tests against the reference NumPy implementation are skipped
automatically when that package is not importable, so this suite works in a
standalone checkout.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from qrjax.core import so3, dynamics, mixer
from qrjax.core.controller import Gains, ctrl_init, control
from qrjax.core.trajectory import make_trajectory
from qrjax.envs import Config, EnvConfig, VecEnv, DisturbRanges
from qrjax.rl import SAC, buffer as buffer_mod
from qrjax.rl.curriculum import flat_ranges


# ------------------------------------------------------------------- SO(3)

def test_hat_vee_roundtrip():
    w = jnp.array([0.3, -1.2, 0.7])
    assert np.allclose(so3.vee(so3.hat(w)), w, atol=1e-6)


def test_hat_matches_cross_product():
    w, v = jnp.array([0.4, 0.1, -0.9]), jnp.array([1.0, -2.0, 0.5])
    assert np.allclose(so3.hat(w) @ v, jnp.cross(w, v), atol=1e-6)


def test_project_to_so3_gives_valid_rotation():
    key = jax.random.PRNGKey(0)
    R = so3.project_to_so3(jax.random.normal(key, (3, 3)))
    assert np.allclose(R.T @ R, jnp.eye(3), atol=1e-5)
    assert float(jnp.linalg.det(R)) == pytest.approx(1.0, abs=1e-5)


def test_so3_log_is_nan_safe_at_identity():
    """theta -> 0 is a removable singularity; a naive formula returns NaN."""
    out = so3.so3_log(jnp.eye(3))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0, atol=1e-7)


def test_so3_log_nan_safe_under_jit_and_vmap():
    Rs = jnp.stack([jnp.eye(3)] * 4)
    out = jax.jit(jax.vmap(so3.so3_log))(Rs)
    assert np.all(np.isfinite(out))


def test_normalize_falls_back_on_zero_vector():
    fb = jnp.array([0.0, 0.0, 1.0])
    assert np.allclose(so3.normalize(jnp.zeros(3), fb), fb)
    assert np.all(np.isfinite(jax.grad(lambda v: so3.normalize(v, fb).sum())(jnp.zeros(3))))


# -------------------------------------------------------------- controller

def test_pd_is_pid_with_zero_integral_gains():
    """The two baselines differ ONLY by ki and kI, by construction."""
    pd = Gains.pd()
    assert pd.ki == 0.0 and pd.kI == 0.0
    pid = Gains.pid()
    assert pid.ki == pytest.approx(3.6)      # kx / 10 at m = 2 kg
    assert pid.kI == pytest.approx(0.881)    # kR / 10


def test_controller_first_step_emits_no_feedforward_impulse():
    """Rd_prev has no history at reset; omega_d must be zero, not a spike."""
    g = Gains.pd()
    traj = make_trajectory("figure8")
    x0, v0, _, _ = traj(0.0)
    st = dynamics.initial_state(x0, v0)
    f, M, cs, _ = control(ctrl_init(), st, traj(0.0), g)
    assert np.all(np.isfinite(M))
    assert np.linalg.norm(M) < 10.0
    assert bool(cs.started)


def test_pid_integral_saturates():
    g = Gains.pid()
    traj = make_trajectory("hover")
    st = dynamics.initial_state(jnp.array([5.0, -4.0, 3.0]), jnp.zeros(3))
    cs = ctrl_init()
    for _ in range(500):
        _, _, cs, _ = control(cs, st, traj(0.0), g)
    assert np.all(np.abs(np.asarray(cs.ei)) <= g.ei_limit + 1e-6)


def test_hold_integral_freezes_accumulator():
    g = Gains.pid()
    traj = make_trajectory("hover")
    st = dynamics.initial_state(jnp.array([1.0, 0.0, 0.0]), jnp.zeros(3))
    _, _, cs, _ = control(ctrl_init(), st, traj(0.0), g)
    frozen = np.asarray(cs.ei)
    for _ in range(20):
        _, _, cs, _ = control(cs, st, traj(0.0), g, hold_integral=True)
    assert np.allclose(np.asarray(cs.ei), frozen)


# ------------------------------------------------------------------ mixer

def test_mixer_nominal_is_identity_roundtrip():
    f, M, _, sat = mixer.apply(20.0, jnp.array([0.1, -0.05, 0.02]),
                               mixer.MixerParams.nominal())
    assert float(f) == pytest.approx(20.0, abs=1e-3)
    assert np.allclose(M, [0.1, -0.05, 0.02], atol=1e-4)
    assert not bool(sat)


def test_mixer_reports_saturation_beyond_envelope():
    _, _, _, sat = mixer.apply(1e4, jnp.zeros(3), mixer.MixerParams.nominal())
    assert bool(sat)


def test_mixer_thrust_scale_changes_produced_force():
    p = mixer.MixerParams.nominal()._replace(kf_scale=jnp.full(4, 1.2))
    f, _, _, _ = mixer.apply(20.0, jnp.zeros(3), p)
    assert float(f) == pytest.approx(24.0, rel=1e-3)


# -------------------------------------------------------------------- env

def test_obs_dim_matches_config():
    cfg = EnvConfig()
    env = VecEnv(cfg, 4)
    _, obs = env.reset(jax.random.PRNGKey(0), env.broadcast_ranges(
        DisturbRanges.from_config(cfg)))
    assert obs.shape == (4, cfg.obs_dim())
    assert cfg.obs_dim() == 156          # matches the PyTorch reference


def test_env_step_is_finite_and_jittable():
    cfg = EnvConfig()
    env = VecEnv(cfg, 8)
    r = env.broadcast_ranges(flat_ranges(cfg))
    state, obs = env.reset(jax.random.PRNGKey(0), r)
    step = jax.jit(env.step)
    for _ in range(20):
        state, obs, rew, done, info = step(state, jnp.zeros((8, 4)), r)
    assert np.all(np.isfinite(obs))
    assert np.all(np.isfinite(rew))


def test_disabled_disturbance_is_degenerate_range():
    cfg = EnvConfig()
    cfg.disturbances = ("none",)
    r = DisturbRanges.from_config(cfg)
    assert float(r.k_min) == float(r.k_max) == 1.0
    assert float(r.force_max) == 0.0


def test_env_auto_resets_on_episode_end():
    cfg = EnvConfig()
    cfg.episode_steps = 5
    env = VecEnv(cfg, 2)
    r = env.broadcast_ranges(DisturbRanges.from_config(cfg))
    # stagger=False pins every env to phase 0 so the boundary is deterministic.
    state, obs = env.reset(jax.random.PRNGKey(0), r, stagger=False)
    for _ in range(5):
        state, obs, rew, done, info = env.step(state, jnp.zeros((2, 4)), r)
    assert bool(np.all(np.asarray(done)))
    assert int(state.step_idx[0]) == 0       # already reset for the next episode


def test_reset_staggering_desynchronizes_episodes():
    """Without staggering all N envs share a trajectory phase, so every
    transition in an iteration comes from the same point on the path."""
    cfg = EnvConfig()
    env = VecEnv(cfg, 64)
    r = env.broadcast_ranges(DisturbRanges.from_config(cfg))

    state, _ = env.reset(jax.random.PRNGKey(0), r, stagger=False)
    assert len(np.unique(np.asarray(state.step_idx))) == 1

    state, _ = env.reset(jax.random.PRNGKey(0), r, stagger=True)
    assert len(np.unique(np.asarray(state.step_idx))) > 20


def test_same_seed_gives_same_disturbance_across_controllers():
    """Required for the paired PD-vs-PID evaluation to be valid."""
    draws = []
    for controller in ("pd", "pid"):
        cfg = EnvConfig()
        cfg.base_controller = controller
        env = VecEnv(cfg, 4)
        r = env.broadcast_ranges(flat_ranges(cfg))
        state, _ = env.reset(jax.random.PRNGKey(3), r)
        draws.append((np.asarray(state.k), np.asarray(state.external_force)))
    assert np.allclose(draws[0][0], draws[1][0])
    assert np.allclose(draws[0][1], draws[1][1])


def test_pid_beats_pd_under_constant_force():
    """Integral action should remove the steady-state offset a constant
    disturbance leaves under PD. This is the premise of the whole comparison."""
    def steady(controller):
        cfg = EnvConfig()
        cfg.base_controller = controller
        cfg.trajectory = "hover"
        cfg.episode_steps = 1500
        cfg.disturbances = ("force",)
        cfg.external_force_max = 2.0
        env = VecEnv(cfg, 4)
        r = env.broadcast_ranges(DisturbRanges.from_config(cfg))
        state, obs = env.reset(jax.random.PRNGKey(11), r)
        errs = []
        step = jax.jit(env.step)
        for _ in range(1500):
            state, obs, rew, done, info = step(state, jnp.zeros((4, 4)), r)
            errs.append(np.asarray(info["pos_err_desired"]))
        return np.mean(np.array(errs)[-300:])

    e_pd, e_pid = steady("pd"), steady("pid")
    assert e_pid < 0.5 * e_pd, f"PID {e_pid:.4f} not better than PD {e_pd:.4f}"


# ------------------------------------------------------------------- SAC

def test_buffer_wraps_and_tracks_size():
    buf = buffer_mod.init(10, 3, 2)
    for _ in range(3):
        buf = buffer_mod.add_batch(
            buf, jnp.ones((4, 3)), jnp.ones((4, 2)), jnp.ones(4),
            jnp.ones((4, 3)), jnp.zeros(4))
    assert int(buf.size) == 10          # capped at capacity
    assert int(buf.ptr) == 2            # 12 mod 10


def test_sac_update_runs_and_is_finite():
    cfg = Config()
    agent = SAC(156, 4, cfg.sac)
    st = agent.init(jax.random.PRNGKey(0))
    k = jax.random.PRNGKey(1)
    buf = buffer_mod.init(2000, 156, 4)
    buf = buffer_mod.add_batch(
        buf, jax.random.normal(k, (500, 156)),
        jax.random.uniform(k, (500, 4), minval=-1, maxval=1),
        jax.random.normal(k, (500,)), jax.random.normal(k, (500, 156)),
        jnp.zeros(500))
    upd = jax.jit(agent.update)
    for _ in range(5):
        st, m = upd(st, buf)
    assert all(np.isfinite(float(v)) for v in m.values())


def test_actor_starts_at_zero_residual():
    """Zero-init mean head means training begins as the untouched baseline."""
    cfg = Config()
    agent = SAC(156, 4, cfg.sac)
    st = agent.init(jax.random.PRNGKey(0))
    mean, _ = agent.actor.apply(st.actor_params, jnp.ones((1, 156)))
    assert np.allclose(np.asarray(mean), 0.0, atol=1e-6)


# ---------------------------------------------- parity with NumPy reference

def test_matches_numpy_reference_env():
    """Cross-check against the original NumPy implementation, when available.

    importorskip must be called INSIDE the test: at decorator level it runs at
    collection time and silently skips the entire module.
    """
    pytest.importorskip("robust_safe_rl",
                        reason="reference NumPy package not installed")
    from robust_safe_rl.rl.config import EnvConfig as NPEnvConfig
    from robust_safe_rl.rl.residual_env import ResidualTwinEnv

    K, STEPS = 1.2, 300
    npc = NPEnvConfig()
    npc.disturbances = ("massmoi",)
    npe = ResidualTwinEnv(npc, seed=0)
    npe.reset(k=K)
    np_pos = []
    for _ in range(STEPS):
        npe.step(np.zeros(4))
        np_pos.append(np.linalg.norm(
            npe.dyn_true.state()["x"] - npe.traj.desired(npe.t - npc.dt)["x"]))

    jc = EnvConfig()
    jc.disturbances = ("massmoi",)
    jc.k_min = jc.k_max = K
    env = VecEnv(jc, 1)
    r = env.broadcast_ranges(DisturbRanges.from_config(jc))
    state, _ = env.reset(jax.random.PRNGKey(0), r)
    jx_pos = []
    for _ in range(STEPS):
        state, _, _, _, info = env.step(state, jnp.zeros((1, 4)), r)
        jx_pos.append(float(info["pos_err_desired"][0]))

    # float32 JAX vs float64 NumPy over 300 closed-loop steps
    assert np.max(np.abs(np.array(np_pos) - np.array(jx_pos))) < 1e-5


# ------------------------------------------------- replay-buffer correctness

def test_step_exposes_true_successor_observation():
    """On a done step, next_obs belongs to a NEW episode; final_obs does not.

    Storing next_obs in the replay buffer makes the critic bootstrap from a
    state under a different disturbance draw.
    """
    cfg = EnvConfig()
    cfg.episode_steps = 4
    env = VecEnv(cfg, 4)
    r = env.broadcast_ranges(DisturbRanges.from_config(cfg))
    state, obs = env.reset(jax.random.PRNGKey(0), r, stagger=False)
    for i in range(4):
        state, obs, rew, done, info = env.step(state, jnp.zeros((4, 4)), r)
        if i < 3:
            assert not np.any(np.asarray(done))
            assert np.allclose(np.asarray(obs), np.asarray(info["final_obs"]))
    # final step truncates: the two observations must now diverge
    assert np.all(np.asarray(done))
    assert not np.allclose(np.asarray(obs), np.asarray(info["final_obs"]))


def test_truncation_is_not_marked_terminal():
    """Running out of clock is not an absorbing state.

    Bootstrapping must continue through truncation, so the stored mask has to
    be `terminated`, never `done`.
    """
    cfg = EnvConfig()
    cfg.episode_steps = 4
    cfg.disturbances = ("none",)
    env = VecEnv(cfg, 4)
    r = env.broadcast_ranges(DisturbRanges.from_config(cfg))
    state, obs = env.reset(jax.random.PRNGKey(0), r, stagger=False)
    for _ in range(4):
        state, obs, rew, done, info = env.step(state, jnp.zeros((4, 4)), r)
    assert np.all(np.asarray(info["truncated"]))
    assert not np.any(np.asarray(info["terminated"]))
    assert np.all(np.asarray(done))


# ------------------------------------------------- time-varying disturbances

def test_zero_frequency_reduces_to_constant_force():
    """f=0 must be EXACTLY the constant-disturbance case.

    The whole frequency sweep is anchored on this: if the f=0 column does not
    reproduce the constant-disturbance results, the sweep is uninterpretable.
    """
    cfg = EnvConfig()
    cfg.disturbances = ("force",)
    env = VecEnv(cfg, 8)
    r = env.broadcast_ranges(DisturbRanges.from_config(cfg, force_freq=0.0))
    state, _ = env.reset(jax.random.PRNGKey(0), r, stagger=False)
    assert np.allclose(np.asarray(state.force_phase), 0.0)
    step = jax.jit(env.step)
    forces = []
    for _ in range(50):
        state, _, _, _, info = step(state, jnp.zeros((8, 4)), r)
        forces.append(np.asarray(info["force_norm"]))
    forces = np.array(forces)
    assert np.allclose(forces, forces[0], atol=1e-5)


def test_force_rms_is_matched_across_frequency():
    """Amplitude is sqrt(2)-scaled for f>0 so disturbance ENERGY is constant.

    Without this the sweep would confound frequency with a weaker disturbance
    and every arm would appear to improve the moment f left zero.
    """
    cfg = EnvConfig()
    cfg.disturbances = ("force",)
    env = VecEnv(cfg, 32)
    step = jax.jit(env.step)

    def rms(freq):
        r = env.broadcast_ranges(DisturbRanges.from_config(cfg, force_freq=freq))
        state, _ = env.reset(jax.random.PRNGKey(0), r, stagger=False)
        vals = []
        for _ in range(600):
            state, _, _, _, info = step(state, jnp.zeros((32, 4)), r)
            vals.append(np.asarray(info["force_norm"]))
        return float(np.sqrt(np.mean(np.array(vals) ** 2)))

    base = rms(0.0)
    for f in (0.5, 2.0):
        assert abs(rms(f) - base) / base < 0.02, f"RMS not matched at {f} Hz"


def test_disturbance_bank_is_stable_under_new_quantities():
    """Named fold_in sub-keys, not sequential split.

    With split(key, n), adding one new sampled quantity changes n and silently
    regenerates EVERY other draw, which invalidates a frozen evaluation bank
    and makes results non-comparable across code versions. fold_in with a fixed
    id per quantity is stable. This test pins the actual draws, so if someone
    reintroduces sequential splitting it fails loudly rather than quietly
    shifting every published number.
    """
    cfg = EnvConfig()
    cfg.disturbances = ("massmoi", "force", "motor_coeff")
    env = VecEnv(cfg, 4)
    r = env.broadcast_ranges(DisturbRanges.from_config(cfg))
    state, _ = env.reset(jax.random.PRNGKey(20260829), r, stagger=False)

    # Mirror VecEnv.reset exactly: one split for the stagger key, then a
    # per-env split of the first half.
    k_reset, _k_phase = jax.random.split(jax.random.PRNGKey(20260829))
    seed_keys = jax.random.split(k_reset, 4)
    for i in range(4):
        expected_k = jax.random.uniform(
            jax.random.fold_in(seed_keys[i], 1), (),
            minval=cfg.k_min, maxval=cfg.k_max)
        assert float(state.k[i]) == pytest.approx(float(expected_k), abs=1e-6)


def test_frequency_does_not_perturb_the_disturbance_bank():
    """A sweep must vary ONLY the temporal profile.

    If changing the frequency also redrew mass, force amplitude, or actuator
    scales, the sweep would confound frequency with a different set of
    vehicles.
    """
    cfg = EnvConfig()
    env = VecEnv(cfg, 32)
    ref = None
    for f in (0.0, 0.5, 2.0, 4.0):
        r = env.broadcast_ranges(DisturbRanges.from_config(cfg, force_freq=f))
        state, _ = env.reset(jax.random.PRNGKey(1), r, stagger=False)
        fp = np.concatenate([
            np.asarray(state.k)[:, None],
            np.asarray(state.external_force),
            np.asarray(state.mixer_true.kf_scale),
        ], axis=1)
        if ref is None:
            ref = fp
        else:
            assert np.allclose(fp, ref), f"bank changed at f={f}"


def test_dc_prob_zero_leaves_constant_configs_untouched():
    """Adding the DC-mass option must not perturb existing experiments.

    A constant-disturbance config (force_freq_max = 0) has to stay bit-identical
    to before force_dc_prob existed, or every Stage 1 number silently shifts.
    """
    cfg = EnvConfig()
    env = VecEnv(cfg, 32)
    r = env.broadcast_ranges(DisturbRanges.from_config(cfg))
    state, _ = env.reset(jax.random.PRNGKey(20260829), r, stagger=False)
    assert np.all(np.asarray(state.force_freq) == 0.0)
    assert np.all(np.asarray(state.force_phase) == 0.0)

    # the disturbance bank itself must be the documented one
    k_reset, _ = jax.random.split(jax.random.PRNGKey(20260829))
    seed_keys = jax.random.split(k_reset, 32)
    expected = jax.random.uniform(jax.random.fold_in(seed_keys[0], 1), (),
                                  minval=cfg.k_min, maxval=cfg.k_max)
    assert float(state.k[0]) == pytest.approx(float(expected), abs=1e-6)


def test_dc_prob_puts_real_mass_at_exactly_zero():
    """f ~ U[0, fmax] gives P(f == 0) = 0, so a band alone never trains DC."""
    cfg = EnvConfig()
    cfg.force_freq_min, cfg.force_freq_max = 0.0, 4.0

    cfg.force_dc_prob = 0.0
    env = VecEnv(cfg, 1000)
    state, _ = env.reset(jax.random.PRNGKey(0),
                         env.broadcast_ranges(DisturbRanges.from_config(cfg)),
                         stagger=False)
    assert np.mean(np.asarray(state.force_freq) == 0.0) < 0.01

    cfg.force_dc_prob = 0.3
    env = VecEnv(cfg, 2000)
    state, _ = env.reset(jax.random.PRNGKey(0),
                         env.broadcast_ranges(DisturbRanges.from_config(cfg)),
                         stagger=False)
    f = np.asarray(state.force_freq)
    assert 0.25 < np.mean(f == 0.0) < 0.35
    # DC episodes must be exactly constant, not merely slow
    assert np.all(np.asarray(state.force_phase)[f == 0.0] == 0.0)


def test_explicit_frequency_overrides_training_dc_prob():
    """A sweep at an explicit frequency must not inherit the training dc_prob.

    force_dc_prob lives in the TRAINING config. A mixed-trained policy set
    would otherwise be evaluated with ~30% of episodes at f=0 at every
    requested frequency, while const-trained sets get the pure frequency --
    silently comparing policy sets on different disturbance distributions.
    """
    from qrjax.rl.curriculum import flat_ranges
    cfg = EnvConfig()
    cfg.force_freq_min, cfg.force_freq_max = 0.0, 4.0
    cfg.force_dc_prob = 0.3

    env = VecEnv(cfg, 512)
    leaked = env.broadcast_ranges(flat_ranges(cfg, force_freq=4.0))
    state, _ = env.reset(jax.random.PRNGKey(0), leaked, stagger=False)
    assert np.mean(np.asarray(state.force_freq) == 0.0) > 0.2   # the leak

    fixed = env.broadcast_ranges(
        flat_ranges(cfg, force_freq=4.0, force_dc_prob=0.0))
    state, _ = env.reset(jax.random.PRNGKey(0), fixed, stagger=False)
    assert np.all(np.asarray(state.force_freq) == 4.0)


def test_history_length_changes_obs_dim_consistently():
    for h, expected in ((10, 156), (30, 476)):
        cfg = EnvConfig()
        cfg.history = h
        assert cfg.obs_dim() == expected
        env = VecEnv(cfg, 2)
        _, obs = env.reset(jax.random.PRNGKey(0),
                           env.broadcast_ranges(DisturbRanges.from_config(cfg)))
        assert obs.shape == (2, expected)


def test_hidden_size_round_trips_through_config():
    """--hidden must survive JSON serialization as a tuple.

    JSON has no tuples, so a naive reload gives a list, and Flax builds a
    different module. Config._clean converts lists back.
    """
    cfg = Config()
    cfg.sac.hidden = (512, 512)
    reloaded = Config.from_dict(cfg.to_dict())
    assert reloaded.sac.hidden == (512, 512)
    assert isinstance(reloaded.sac.hidden, tuple)


def test_wider_network_has_more_parameters_and_same_obs_dim():
    """Unlike --history, --hidden leaves obs_dim untouched, so policies of
    different sizes can share one evaluation sweep."""
    counts = []
    for hidden in ((256, 256), (512, 512)):
        cfg = Config()
        cfg.sac.hidden = hidden
        agent = SAC(156, 4, cfg.sac)
        st = agent.init(jax.random.PRNGKey(0))
        counts.append(sum(x.size for x in jax.tree.leaves(st.actor_params)))
        mean, _ = agent.actor.apply(st.actor_params, jnp.ones((1, 156)))
        assert mean.shape == (1, 4)
    assert counts[1] > 2 * counts[0]


def test_moment_filter_is_independent_of_thrust_filter():
    """Per-channel filtering, so the thrust/moment asymmetry can be tested."""
    from qrjax.rl.curriculum import flat_ranges

    def applied_amplitude(thrust_beta, moment_beta):
        cfg = EnvConfig()
        cfg.thrust_filter_beta = thrust_beta
        cfg.moment_filter_beta = moment_beta
        env = VecEnv(cfg, 4)
        r = env.broadcast_ranges(flat_ranges(cfg))
        state, _ = env.reset(jax.random.PRNGKey(0), r, stagger=False)
        step = jax.jit(env.step)
        out = []
        for k in range(40):
            a = jnp.tile(jnp.full(4, 1.0 if k % 2 == 0 else -1.0), (4, 1))
            state, *_ = step(state, a, r)
            out.append(np.asarray(state.prev_action_norm[0]))
        return np.abs(np.array(out)[10:]).mean(axis=0)

    # default: thrust filtered, moments raw
    amp = applied_amplitude(0.2, 1.0)
    assert amp[0] < 0.3 and np.all(amp[1:] > 0.99)

    # both filtered
    amp = applied_amplitude(0.2, 0.2)
    assert np.all(amp < 0.3)

    # neither
    amp = applied_amplitude(1.0, 1.0)
    assert np.all(amp > 0.99)


def test_true_allocation_removes_the_gain_mismatch():
    """The T4 diagnostic: with true allocation the produced wrench equals the
    command, so the effective loop gain is 1 regardless of the disturbance."""
    from qrjax.core import mixer
    p = mixer.MixerParams(kf_scale=jnp.full(4, 1.27),
                          moment_scale=jnp.full(4, 1.015),
                          arm_scale=jnp.full(4, 1.202))
    M_cmd = jnp.array([0.3, -0.2, 0.05])
    f_n, M_n, _, _ = mixer.apply(20.0, M_cmd, p, use_true_allocation=False)
    f_t, M_t, _, _ = mixer.apply(20.0, M_cmd, p, use_true_allocation=True)
    assert float(f_t) == pytest.approx(20.0, rel=1e-3)
    assert np.allclose(np.asarray(M_t), np.asarray(M_cmd), atol=1e-4)
    assert float(f_n) > 24.0                       # nominal allocation over-produces
    assert not np.allclose(np.asarray(M_n), np.asarray(M_cmd), atol=1e-3)


def test_env_exposes_positions_for_trajectory_plots():
    cfg = EnvConfig()
    env = VecEnv(cfg, 2)
    r = env.broadcast_ranges(DisturbRanges.from_config(cfg))
    state, _ = env.reset(jax.random.PRNGKey(0), r, stagger=False)
    _, _, _, _, info = env.step(state, jnp.zeros((2, 4)), r)
    for key in ("x_true", "x_nom", "x_des", "motor_cmd"):
        assert key in info, f"info is missing {key}"
        assert np.all(np.isfinite(np.asarray(info[key])))


def test_lpf_transfer_function_matches_the_env():
    """The analytic response of u_t = (1-b)u_{t-1} + b*a_t must match what the
    env actually applies, at every frequency including Nyquist."""
    from scripts.analyze_lpf import transfer_magnitude
    from qrjax.rl.curriculum import flat_ranges

    beta, dt = 0.2, 0.01
    cfg = EnvConfig()
    cfg.thrust_filter_beta = beta
    env = VecEnv(cfg, 1)
    r = env.broadcast_ranges(flat_ranges(cfg, force_freq=0.0, force_dc_prob=0.0))
    step = jax.jit(env.step)

    for f in (1.0, 10.0, 50.0):
        state, _ = env.reset(jax.random.PRNGKey(0), r, stagger=False)
        applied, requested = [], []
        for k in range(400):
            # cosine, so the Nyquist probe is (-1)^k rather than identically 0
            a = jnp.array([[0.5 * np.cos(2 * np.pi * f * k * dt), 0., 0., 0.]])
            state, *_ = step(state, a, r)
            applied.append(float(state.prev_action_norm[0, 0]))
            requested.append(float(a[0, 0]))
        measured = np.std(applied[100:]) / np.std(requested[100:])
        predicted = float(transfer_magnitude([f], beta, dt)[0])
        assert measured == pytest.approx(predicted, rel=1e-3), f"at {f} Hz"


def test_orchestrator_can_reach_every_training_flag():
    """run_stage1 must not silently drop a scripts.train flag.

    Twice now a documented command failed because a flag existed in train.py
    but was never plumbed through the orchestrator. Any train.py flag not
    explicitly declared in run_stage1 must be reachable via --train_args.
    """
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "scripts"
    train_flags = set(re.findall(r'add_argument\("--(\w+)',
                                 (root / "train.py").read_text()))
    stage_src = (root / "run_stage1.py").read_text()
    stage_flags = set(re.findall(r'add_argument\("--(\w+)', stage_src))

    # set by the orchestrator itself, or mode flags it owns
    owned = {"run_name", "runs_root", "curriculum", "no_curriculum",
             "curriculum_path", "base_controller", "seed"}
    unreachable = train_flags - stage_flags - owned
    assert "train_args" in stage_flags, (
        "run_stage1 needs a --train_args escape hatch so no train.py flag is "
        "unreachable")
    # everything else must either be declared or go through the escape hatch,
    # which shlex-splits arbitrary flags
    assert "shlex.split(args.train_args)" in stage_src



def test_error_convention_is_true_minus_nominal():
    """Observation error must be actual-relative-to-reference, matching the
    geometric controller's own convention and the paper's equations."""
    from qrjax.envs.obs import discrepancy
    from qrjax.core.dynamics import RigidState
    from qrjax.core.so3 import hat

    cfg = EnvConfig()
    delta = jnp.array([0.02, 0.0, 0.0])
    nom = RigidState(x=jnp.zeros(3), v=jnp.zeros(3), R=jnp.eye(3),
                     omega=jnp.zeros(3))
    true = RigidState(x=jnp.array([1.0, 0.0, 0.0]), v=jnp.array([0.5, 0., 0.]),
                      R=jnp.eye(3) + hat(delta), omega=jnp.array([0.3, 0., 0.]))
    e = discrepancy(nom, true, cfg)
    assert float(e[0]) > 0, "position must be true - nominal"
    assert float(e[3]) > 0, "velocity must be true - nominal"
    assert float(e[6]) > 0, "attitude error must share the sign of the rotation"
    assert float(e[9]) > 0, "body rate must be true - nominal"
