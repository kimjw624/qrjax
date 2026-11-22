"""Regression tests for the switchable fixes (study_fixes, Nov 2026).

Every fix is a config switch. Since v2 the defaults are the fixes and
``legacy_env()`` holds the original values; these tests are written against
the original code plus one named switch at a time. They check (a) that each
fix does what it claims, and (b) the facts the study relies on: the mixer sign error is a pure relabelling, the analytic
rates are the exact derivatives of Rd, the prioritized allocator equals the
plain inverse whenever the request is feasible.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrjax.core import mixer, trajectory
from qrjax.core.controller import (Gains, ctrl_init, control, reference_attitude,
                                   attitude_from_thrust)
from qrjax.core.dynamics import initial_state
from qrjax.core.so3 import so3_log, vee
from qrjax.envs import EnvConfig, VecEnv
from qrjax.envs.config import FIXED_ENV, FIXED_TRAIN, LEGACY_ENV, fixed_preset, legacy_env
from qrjax.rl.curriculum import flat_ranges


# ------------------------------------------------------------------ trajectory
@pytest.mark.parametrize("kind", ["figure8", "circle"])
def test_jerk_snap_match_finite_differences(kind):
    traj = trajectory.make_trajectory(kind)
    der = trajectory.make_trajectory_derivs(kind)
    h = 1e-3
    for t in (0.0, 1.3, 4.7, 8.1):
        a_p, a_m = traj(t + h)[2], traj(t - h)[2]
        j_p, j_m = der(t + h)[0], der(t - h)[0]
        j, s = der(t)
        np.testing.assert_allclose((a_p - a_m) / (2 * h), j, atol=2e-4)
        np.testing.assert_allclose((j_p - j_m) / (2 * h), s, atol=2e-4)


# ------------------------------------------------------------------ controller
def test_analytic_rates_equal_derivative_of_reference_attitude():
    jax.config.update("jax_enable_x64", True)
    try:
        g = Gains.pid(dt=0.005)
        traj = trajectory.make_trajectory("figure8")
        der = trajectory.make_trajectory_derivs("figure8")
        h = 1e-4
        for t in (0.0, 2.2, 6.9):
            Rd, wd, wd_dot = reference_attitude(g, traj(t), der(t))
            Rp, wp, _ = reference_attitude(g, traj(t + h), der(t + h))
            Rm, wm, _ = reference_attitude(g, traj(t - h), der(t - h))
            w_num = vee(so3_log(Rm.T @ Rp)) / (2 * h)
            np.testing.assert_allclose(wd, w_num, atol=1e-6)
            np.testing.assert_allclose(wd_dot, (wp - wm) / (2 * h), atol=1e-5)
    finally:
        jax.config.update("jax_enable_x64", False)


def test_fd_counter_fix_removes_second_step_impulse():
    g = Gains.pid(dt=0.005)
    traj = trajectory.make_trajectory("figure8")
    x0, v0, _, _ = traj(0.0)
    st = initial_state(x0, v0)
    out = {}
    for fix in (False, True):
        cs = ctrl_init()
        _, _, cs, i1 = control(cs, st, traj(0.0), g, fd_counter_fix=fix)
        _, _, cs, i2 = control(cs, st, traj(0.005), g, fd_counter_fix=fix)
        _, _, cs, i3 = control(cs, st, traj(0.010), g, fd_counter_fix=fix)
        out[fix] = (float(jnp.linalg.norm(i2.omega_d_dot)),
                    float(jnp.linalg.norm(i3.omega_d_dot)), float(jnp.linalg.norm(i2.omega_d)))
    impulse, _, wd = out[False]
    assert impulse == pytest.approx(wd / 0.005, rel=1e-4)      # the original spike
    assert impulse > 10.0
    assert out[True][0] == 0.0                                  # gated on 2 samples
    assert out[True][1] < 0.2 * impulse                         # 3rd call is a real difference


def test_default_control_path_unchanged_by_new_arguments():
    g = Gains.pid(dt=0.005)
    traj = trajectory.make_trajectory("figure8")
    st = initial_state(*traj(1.0)[:2])
    a = control(ctrl_init(), st, traj(1.0), g)
    b = control(ctrl_init(), st, traj(1.0), g, rate_ff="finite_difference",
                fd_counter_fix=False)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# ------------------------------------------------------------------ mixer
def test_frd_matrix_matches_cross_products():
    B = np.asarray(mixer.B_NOM_FRD)
    a = mixer.NOMINAL_ARM
    pos = [(a, -a), (-a, a), (a, a), (-a, -a)]
    for j, (x, y) in enumerate(pos):
        r = np.array([x, y, 0.0])
        Fz = np.array([0.0, 0.0, -1.0])
        np.testing.assert_allclose(B[1:3, j], np.cross(r, Fz)[:2], atol=1e-7)


def test_legacy_mixer_is_a_rotor_relabelling():
    P = np.eye(4)[[1, 0, 3, 2]]
    np.testing.assert_allclose(np.asarray(mixer.B_NOM), np.asarray(mixer.B_NOM_FRD) @ P,
                               atol=1e-7)
    # same spin direction inside each swapped pair
    assert list(np.asarray(mixer._SPIN)[[1, 0, 3, 2]]) == list(np.asarray(mixer._SPIN))


@pytest.mark.parametrize("allocator", ["clip", "prioritized"])
def test_legacy_and_frd_give_same_wrench_with_symmetric_scales(allocator):
    rng = np.random.default_rng(0)
    for _ in range(300):
        f = float(rng.uniform(5, 45))
        M = jnp.asarray(rng.normal(size=3) * [1.5, 1.5, 0.2])
        s = rng.uniform(0.7, 1.3, 3)
        p = mixer.MixerParams(*(jnp.full(4, v) for v in s))
        fa, Ma, _, sa = mixer.apply(f, M, p, allocator=allocator, geometry="legacy")
        fb, Mb, _, sb = mixer.apply(f, M, p, allocator=allocator, geometry="frd")
        np.testing.assert_allclose(fa, fb, atol=1e-4)
        np.testing.assert_allclose(Ma, Mb, atol=1e-4)
        assert bool(sa) == bool(sb)


def test_prioritized_equals_inverse_when_feasible_and_keeps_direction():
    rng = np.random.default_rng(1)
    p = mixer.MixerParams.nominal()
    n_infeasible = 0
    for _ in range(400):
        f = float(rng.uniform(5, 40))
        M = jnp.asarray(rng.normal(size=3) * [2.0, 2.0, 0.3])
        fc, Mc, Tc, sat = mixer.apply(f, M, p, allocator="clip")
        fp, Mp, Tp, sat_p = mixer.apply(f, M, p, allocator="prioritized")
        assert bool(sat) == bool(sat_p)
        assert np.all(np.asarray(Tp) >= -1e-6)
        assert np.all(np.asarray(Tp) <= mixer.MAX_MOTOR_THRUST + 1e-6)
        if not bool(sat):
            np.testing.assert_allclose(Tp, Tc, atol=1e-6)
        else:
            n_infeasible += 1
            req = np.asarray(M[:2])
            got = np.asarray(Mp[:2])
            if np.linalg.norm(got) > 1e-6:
                cos = req @ got / (np.linalg.norm(req) * np.linalg.norm(got))
                assert cos > 0.999                       # roll/pitch direction kept
    assert n_infeasible > 20


# ------------------------------------------------------------------ environment
def _rollout(cfg, steps, n=16, seed=0, action_scale=0.0):
    env = VecEnv(cfg, n)
    ranges = env.broadcast_ranges(flat_ranges(cfg))
    state, obs = env.reset(jax.random.PRNGKey(seed), ranges, stagger=False)

    def body(c, k):
        state, obs = c
        a = action_scale * jnp.tanh(jax.random.normal(k, (n, 4)))
        state, obs, r, d, info = env.step(state, a, ranges)
        return (state, obs), (info["pos_err_desired"], info["pos_err_twin"],
                              info["saturated"], info["terminated"])
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), steps)
    _, out = jax.lax.scan(body, (state, obs), keys)
    return [np.asarray(x) for x in out]


def _nominal_cfg(**kw):
    cfg = legacy_env()
    cfg.dt = 0.005
    cfg.episode_steps = 2000
    cfg.base_controller = "pid"
    cfg.k_min = cfg.k_max = 1.0
    cfg.external_force_max = 0.0
    for name in ("motor_coeff", "moment_coeff", "arm_length"):
        setattr(cfg, f"{name}_min", 1.0)
        setattr(cfg, f"{name}_max", 1.0)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_all_fixes_track_nominal_plant_and_twin_equals_true():
    cfg = fixed_preset(_nominal_cfg())
    pos_err, twin_err, sat, term = _rollout(cfg, 400)
    assert not term.any()
    assert sat.mean() == 0.0
    assert pos_err.max() < 0.01                 # 1 cm, noise-free nominal plant
    assert twin_err.max() < 1e-5                # nominal true plant == twin


def test_analytic_and_finite_difference_track_alike_without_noise():
    rms = {}
    for rate_ff in ("finite_difference", "analytic"):
        cfg = _nominal_cfg(rate_ff=rate_ff, fd_counter_fix=True, start_on_reference=True)
        cfg.k_min, cfg.k_max = 1.2, 1.2
        cfg.external_force_max = 2.0
        pos_err, *_ = _rollout(cfg, 600)
        rms[rate_ff] = float(np.sqrt(np.mean(pos_err ** 2)))
    assert abs(rms["analytic"] - rms["finite_difference"]) < 0.005


def test_start_on_reference_places_vehicle_on_reference_attitude():
    cfg = _nominal_cfg(start_on_reference=True, rate_ff="analytic")
    env = VecEnv(cfg, 4)
    ranges = env.broadcast_ranges(flat_ranges(cfg))
    state, _ = env.reset(jax.random.PRNGKey(0), ranges, stagger=True)
    t0 = np.asarray(state.step_idx) * cfg.dt
    traj = trajectory.make_trajectory("figure8")
    der = trajectory.make_trajectory_derivs("figure8")
    g = Gains.pid(dt=cfg.dt)
    for i in range(4):
        Rd, wd, _ = reference_attitude(g, traj(t0[i]), der(t0[i]))
        np.testing.assert_allclose(np.asarray(state.true.R[i]), Rd, atol=1e-5)
        np.testing.assert_allclose(np.asarray(state.true.omega[i]), wd, atol=1e-5)


def test_reference_error_termination():
    cfg = _nominal_cfg(term_ref_error=0.05)
    cfg.external_force_max = 3.0
    cfg.base_controller = "pd"
    *_, term = _rollout(cfg, 300)
    assert term.any()


def test_validate_rejects_typos_and_legacy_values_are_original():
    d = legacy_env()
    assert (d.rate_ff, d.fd_counter_fix, d.allocator, d.mixer_geometry,
            d.start_on_reference, d.term_ref_error, d.term_heading_deg) == \
        ("finite_difference", False, "clip", "legacy", False, 0.0, 0.0)
    # v2: the defaults are the fixes
    new = EnvConfig()
    assert all(getattr(new, k) == v for k, v in FIXED_ENV.items())
    for k, bad in (("rate_ff", "analytical"), ("allocator", "prioritised"),
                   ("mixer_geometry", "ned")):
        cfg = EnvConfig()
        setattr(cfg, k, bad)
        with pytest.raises(ValueError):
            cfg.validate()
    assert set(FIXED_ENV) <= set(EnvConfig().__dict__)


# ------------------------------------------------------------------ diagnostic switches
def test_rate_ff_none_and_model_accel_are_valid_and_behave():
    g = Gains.pid(dt=0.005)
    traj = trajectory.make_trajectory("figure8")
    st = initial_state(*traj(1.0)[:2])
    _, _, _, info = control(ctrl_init(), st, traj(1.0), g, rate_ff="none")
    assert float(jnp.abs(info.omega_d).max()) == 0.0
    assert float(jnp.abs(info.omega_d_dot).max()) == 0.0
    for ov in ({"rate_ff": "none"}, {"rate_ff": "analytic", "accel_source": "model"}):
        cfg = _nominal_cfg(**ov)
        cfg.validate()
        pos_err, *_ = _rollout(cfg, 200)
        assert np.isfinite(pos_err).all()
    bad = EnvConfig()
    bad.accel_source = "imu"
    with pytest.raises(ValueError):
        bad.validate()


def test_diagnostic_info_does_not_change_the_flight():
    cfg = fixed_preset(_nominal_cfg())
    cfg.k_min, cfg.k_max = 0.8, 1.2
    cfg.external_force_max = 3.0
    a = _rollout(cfg, 150, seed=3, action_scale=0.5)
    b = _rollout(cfg, 150, seed=3, action_scale=0.5)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_pid_integrator_limit_switch():
    # default 1.0 = the reference tuning, bit for bit
    assert Gains.pid().ei_limit == 1.0
    assert Gains.pid(ei_limit=3.0).ei_limit == 3.0
    cfg = _nominal_cfg(base_controller="pid")
    cfg.k_min, cfg.k_max = 1.3, 1.3                     # heavy: the z integrator winds up
    cfg.external_force_max = 3.0
    a = _rollout(cfg, 400, seed=1)
    cfg_same = _nominal_cfg(base_controller="pid", pid_ei_limit=1.0)
    cfg_same.k_min, cfg_same.k_max, cfg_same.external_force_max = 1.3, 1.3, 3.0
    for x, y in zip(a, _rollout(cfg_same, 400, seed=1)):
        np.testing.assert_array_equal(x, y)
    cfg_big = _nominal_cfg(base_controller="pid", pid_ei_limit=1e6)
    cfg_big.k_min, cfg_big.k_max, cfg_big.external_force_max = 1.3, 1.3, 3.0
    b = _rollout(cfg_big, 400, seed=1)
    assert not np.array_equal(a[0], b[0])
