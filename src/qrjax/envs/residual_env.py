"""Twin-plant residual-RL environment, written as pure functions over a pytree.

Structure mirrors the reference implementation: a disturbance-free NOMINAL twin
and a disturbed TRUE plant both run the same geometric controller on their own
state, and the policy learns a residual that pushes the true plant back onto the
nominal twin's behaviour. The reward and the observation are both built from the
twin discrepancy, so neither depends on the absolute trajectory.

Everything is a pure function of ``EnvState``:

    reset(key, ranges)                 -> EnvState, obs
    step(state, action, ranges)        -> EnvState, obs, reward, done, info

``vmap`` over the leading axis gives N parallel environments; ``lax.scan`` over
``step`` gives a rollout. Neither needs any Python loop.

Two design points worth calling out:

*Disturbance ranges are traced, not static.* ``DisturbRanges`` is passed in as
an argument rather than baked into the closure. A curriculum stage change is
then just a different value, not a different graph, so advancing the curriculum
does not trigger an XLA recompile mid-training. Disabling a disturbance type is
expressed as a degenerate range (min == max == 1, or force_max == 0) for the
same reason.

*Auto-reset is built in.* A terminated env is replaced in-place on the same
step, because a vectorized rollout cannot stop for one env out of N. The
observation returned is the one for the NEW episode, and ``info`` carries the
final statistics of the episode that just ended.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..core import dynamics, mixer
from ..core.controller import Gains, CtrlState, ctrl_init, control
from ..core.dynamics import Inertia, RigidState
from ..core.mixer import MixerParams, WRENCH_MAX
from ..core.trajectory import make_trajectory
from .obs import ObsState, obs_init, obs_push, obs_vector, discrepancy


class DisturbRanges(NamedTuple):
    """Per-episode disturbance sampling ranges.

    Traced, so a curriculum stage change costs nothing at the XLA level.
    A disabled disturbance is a degenerate range rather than a flag.
    """
    k_min: jnp.ndarray
    k_max: jnp.ndarray
    force_max: jnp.ndarray
    force_freq_min: jnp.ndarray
    force_freq_max: jnp.ndarray
    force_dc_prob: jnp.ndarray
    motor_min: jnp.ndarray
    motor_max: jnp.ndarray
    moment_min: jnp.ndarray
    moment_max: jnp.ndarray
    arm_min: jnp.ndarray
    arm_max: jnp.ndarray

    @staticmethod
    def from_config(cfg, enabled=None, force_freq=None, force_dc_prob=None):
        """Build ranges from an EnvConfig, honouring ``cfg.disturbances``."""
        enabled = set(cfg.disturbances if enabled is None else enabled)
        if "none" in enabled:
            enabled = set()
        on = lambda name, lo, hi: (lo, hi) if name in enabled else (1.0, 1.0)
        k = on("massmoi", cfg.k_min, cfg.k_max)
        mo = on("motor_coeff", cfg.motor_coeff_min, cfg.motor_coeff_max)
        mm = on("moment_coeff", cfg.moment_coeff_min, cfg.moment_coeff_max)
        ar = on("arm_length", cfg.arm_length_min, cfg.arm_length_max)
        fmax = cfg.external_force_max if "force" in enabled else 0.0
        # Frequency band for the external force. (0, 0) means a constant force
        # with zero phase, which is EXACTLY the constant-disturbance case, so a
        # frequency sweep evaluated at f=0 must reproduce constant-disturbance
        # numbers bit for bit. That equality is the correctness check on this
        # whole mechanism.
        if force_freq is None:
            flo, fhi = cfg.force_freq_min, cfg.force_freq_max
        elif isinstance(force_freq, (tuple, list)):
            flo, fhi = force_freq
        else:
            flo = fhi = force_freq
        return DisturbRanges(
            k_min=jnp.float32(k[0]), k_max=jnp.float32(k[1]),
            force_max=jnp.float32(fmax),
            force_freq_min=jnp.float32(flo), force_freq_max=jnp.float32(fhi),
            # Probability that an episode draws an exactly-constant force.
            # Under f ~ U[0, fmax] the point f = 0 has measure zero, so a
            # continuous band never trains the DC regime at all -- it only
            # trains near-DC. Since integral action fully rejects a constant
            # disturbance but lags a time-varying one, DC is a physically
            # distinct regime and needs explicit mass, not just proximity.
            force_dc_prob=jnp.float32(
                getattr(cfg, "force_dc_prob", 0.0)
                if force_dc_prob is None else force_dc_prob),
            motor_min=jnp.float32(mo[0]), motor_max=jnp.float32(mo[1]),
            moment_min=jnp.float32(mm[0]), moment_max=jnp.float32(mm[1]),
            arm_min=jnp.float32(ar[0]), arm_max=jnp.float32(ar[1]),
        )


class EnvState(NamedTuple):
    true: RigidState
    nom: RigidState
    cs_true: CtrlState
    cs_nom: CtrlState
    obs_state: ObsState
    inertia_true: Inertia
    mixer_true: MixerParams
    external_force: jnp.ndarray      # amplitude A; instantaneous force is
    force_freq: jnp.ndarray          # A * cos(2 pi f t + phi)
    force_phase: jnp.ndarray
    k: jnp.ndarray
    prev_action_norm: jnp.ndarray
    prev_action_req: jnp.ndarray
    hold_integral: jnp.ndarray
    step_idx: jnp.ndarray
    key: jnp.ndarray
    ep_return: jnp.ndarray
    ep_length: jnp.ndarray
    ep_pos_err_sum: jnp.ndarray      # running sum of ||x_true - x_desired||


def _sample_rotor_scale(key, lo, hi, per_motor):
    if per_motor:
        return jax.random.uniform(key, (4,), minval=lo, maxval=hi)
    v = jax.random.uniform(key, (), minval=lo, maxval=hi)
    return jnp.full((4,), v)


def make_env(cfg):
    """Build ``(reset, step)`` for one environment. vmap them for N envs.

    ``cfg`` is captured statically: changing a gain or a reward weight means a
    new closure and a recompile, which is the correct behaviour -- it makes
    stale-config bugs impossible.
    """
    traj = make_trajectory(
        cfg.trajectory,
        **({"radius": cfg.traj_radius, "speed": cfg.traj_speed, "z0": cfg.traj_z0}
           if cfg.trajectory == "circle" else
           {"z0": cfg.traj_z0} if cfg.trajectory in ("figure8", "hover") else {})
    )

    gains = Gains.make(
        cfg.base_controller, mass=cfg.mass_nom, gravity=cfg.gravity,
        dt=cfg.dt, J=cfg.J_nom,
        **({"ki": cfg.pid_ki, "kI": cfg.pid_kI} if cfg.base_controller == "pid" else {}),
    )
    J_nom = jnp.diag(jnp.asarray(cfg.J_nom, dtype=jnp.float32))
    inertia_nom = Inertia(mass=jnp.float32(cfg.mass_nom), J=J_nom)

    action_dim = cfg.action_dim()
    action_scale = cfg.residual_authority * WRENCH_MAX
    beta = cfg.thrust_filter_beta
    beta_M = getattr(cfg, "moment_filter_beta", 1.0)
    tilt_cos = jnp.cos(jnp.deg2rad(cfg.term_tilt_deg))

    # ------------------------------------------------------------------ reset
    def reset(key, ranges: DisturbRanges, start_step=0) -> EnvState:
        """Build a fresh episode.

        ``start_step`` phase-shifts the episode: the plant starts on the
        trajectory at t = start_step*dt and the episode truncates that many
        steps early. Used ONLY on the very first reset, to stagger the parallel
        envs. Without it all N envs reset together, so every transition in an
        iteration comes from the same point on the trajectory and the replay
        buffer is far less diverse than the env count suggests. Auto-resets
        pass 0 and episodes stay desynchronized thereafter.
        """
        # Named sub-keys via fold_in, NOT sequential split.
        #
        # jax.random.split(key, n) produces a completely different set of keys
        # for a different n, so adding one new disturbance quantity silently
        # regenerates every OTHER draw and invalidates a "frozen" evaluation
        # bank. fold_in with a fixed integer per quantity is stable: a new
        # quantity takes the next unused id and disturbs nothing existing.
        # Never reuse or renumber these ids.
        next_key = jax.random.fold_in(key, 0)
        k_key = jax.random.fold_in(key, 1)
        f_key = jax.random.fold_in(key, 2)
        mo_key = jax.random.fold_in(key, 3)
        mm_key = jax.random.fold_in(key, 4)
        ar_key = jax.random.fold_in(key, 5)
        freq_key = jax.random.fold_in(key, 6)
        ph_key = jax.random.fold_in(key, 7)
        dc_key = jax.random.fold_in(key, 8)

        k = jax.random.uniform(k_key, (), minval=ranges.k_min, maxval=ranges.k_max)
        fext = jax.random.uniform(
            f_key, (3,), minval=-ranges.force_max, maxval=ranges.force_max
        )
        # With probability force_dc_prob the episode is exactly constant;
        # otherwise the frequency is drawn from the band. A band whose maximum
        # is zero is always DC, which keeps constant-disturbance configurations
        # bit-identical to before this option existed.
        draw_dc = jax.random.uniform(dc_key, ()) < ranges.force_dc_prob
        band_freq = jax.random.uniform(
            freq_key, (), minval=ranges.force_freq_min, maxval=ranges.force_freq_max
        )
        force_freq = jnp.where(draw_dc, 0.0, band_freq)

        # Phase is forced to zero whenever the force is constant, so a DC
        # episode is exactly a constant force of amplitude `fext` -- identical
        # to the constant-disturbance experiments.
        is_dc = force_freq <= 0.0
        force_phase = jnp.where(
            is_dc, jnp.zeros(3),
            jax.random.uniform(ph_key, (3,), minval=0.0, maxval=2.0 * jnp.pi),
        )
        mixer_true = MixerParams(
            kf_scale=_sample_rotor_scale(mo_key, ranges.motor_min, ranges.motor_max,
                                         cfg.per_motor_params),
            moment_scale=_sample_rotor_scale(mm_key, ranges.moment_min,
                                             ranges.moment_max, cfg.per_motor_params),
            arm_scale=_sample_rotor_scale(ar_key, ranges.arm_min, ranges.arm_max,
                                          cfg.per_motor_params),
        )

        start_step = jnp.asarray(start_step, dtype=jnp.int32)
        x0, v0, _, _ = traj(start_step * cfg.dt)
        st = dynamics.initial_state(x0, v0)

        return EnvState(
            true=st, nom=st,
            cs_true=ctrl_init(), cs_nom=ctrl_init(),
            obs_state=obs_init(cfg.history, action_dim),
            inertia_true=Inertia(mass=k * cfg.mass_nom, J=k * J_nom),
            mixer_true=mixer_true,
            external_force=fext,
            force_freq=force_freq,
            force_phase=force_phase,
            k=k,
            prev_action_norm=jnp.zeros(action_dim),
            prev_action_req=jnp.zeros(action_dim),
            hold_integral=jnp.array(False),
            step_idx=start_step,
            key=next_key,
            ep_return=jnp.float32(0.0),
            ep_length=jnp.int32(0),
            ep_pos_err_sum=jnp.float32(0.0),
        )

    def reset_obs(state: EnvState):
        return obs_vector(state.obs_state, cfg)

    # ------------------------------------------------------------------- step
    def step(state: EnvState, action, ranges: DisturbRanges):
        action_req = jnp.clip(action, -1.0, 1.0)

        # Independent first-order low-pass per channel group. Thrust defaults
        # to beta=0.2 (the empirically validated value); moments default to 1.0
        # (no filtering).
        betas = jnp.array([beta, beta_M, beta_M, beta_M])
        filtered = (1.0 - betas) * state.prev_action_norm + betas * action_req
        action_norm = jnp.where(state.step_idx > 0, filtered, action_req)
        # The effort/smoothness penalty uses the RAW request, so the reward
        # sees the policy's own chatter rather than the filter's smoothed
        # output. Penalising the filtered signal would let the policy chatter
        # for free.
        reward_a, reward_a_prev = action_req, state.prev_action_req

        residual = action_norm * action_scale

        t = state.step_idx * cfg.dt
        desired = traj(t)

        # Instantaneous external force.
        #
        # At f = 0 the phase is zero, so this reduces identically to
        # state.external_force and the constant-disturbance case is recovered
        # exactly -- which is the correctness check on a frequency sweep.
        #
        # For f > 0 the amplitude is scaled by sqrt(2) so that the TIME-RMS of
        # the force is the same as the DC case. A cos(wt+phi) has RMS A/sqrt(2)
        # while a constant A has RMS A, so without this the disturbance would
        # get weaker the instant frequency left zero, and a sweep would show an
        # artefactual improvement that has nothing to do with frequency. RMS is
        # matched rather than peak; the peak force at f > 0 is correspondingly
        # sqrt(2) larger.
        amp = state.external_force * jnp.where(state.force_freq > 0.0,
                                               jnp.sqrt(2.0), 1.0)
        fext_t = amp * jnp.cos(
            2.0 * jnp.pi * state.force_freq * t + state.force_phase
        )
        hold = jnp.logical_and(state.hold_integral, cfg.pid_antiwindup)

        # --- nominal twin: same law, own state, no disturbance ---
        f_n, M_n, cs_nom, _ = control(state.cs_nom, state.nom, desired, gains)

        # --- true plant: baseline wrench, then the residual correction ---
        f_base, M_base, cs_true, info_t = control(
            state.cs_true, state.true, desired, gains, hold_integral=hold
        )
        f_cmd = f_base + residual[0]
        M_cmd = M_base + residual[1:4]
        u_base = jnp.concatenate([jnp.atleast_1d(f_base), M_base])

        # Both plants allocate with the nominal mixer; only the true plant
        # reconstructs the wrench with the disturbed actuator model.
        # motor_cmd is the per-rotor NOMINAL thrust command after saturation --
        # the physical actuator signal, and the thing that visibly oscillates.
        # The normalized policy action is one linear map away from it, but the
        # motor command is what a real ESC would have to track.
        f_true, M_true, motor_cmd, saturated = mixer.apply(
            f_cmd, M_cmd, state.mixer_true,
            use_true_allocation=getattr(cfg, "allocation_uses_true_params", False))
        f_nom, M_nom, _, _ = mixer.apply_nominal(f_n, M_n)

        true_next = dynamics.step(state.true, f_true, M_true, state.inertia_true,
                                  cfg.gravity, fext_t, cfg.dt)
        nom_next = dynamics.step(state.nom, f_nom, M_nom, inertia_nom,
                                 cfg.gravity, jnp.zeros(3), cfg.dt)

        err = discrepancy(nom_next, true_next, cfg)
        obs_state = obs_push(state.obs_state, err, action_norm, u_base, cfg)

        reward, r_terms = _reward(err, reward_a, reward_a_prev, cfg)

        xd = desired[0]
        pos_err_des = jnp.linalg.norm(true_next.x - xd)
        pos_err_twin = jnp.linalg.norm(nom_next.x - true_next.x)

        terminated = jnp.logical_or(
            pos_err_twin > cfg.term_pos_error,
            jnp.logical_or(true_next.R[2, 2] < tilt_cos,
                           jnp.logical_not(jnp.all(jnp.isfinite(true_next.x)))),
        )
        step_idx = state.step_idx + 1
        truncated = step_idx >= cfg.episode_steps
        done = jnp.logical_or(terminated, truncated)

        ep_return = state.ep_return + reward
        ep_length = state.ep_length + 1
        ep_pos_err_sum = state.ep_pos_err_sum + pos_err_des

        stepped = EnvState(
            true=true_next, nom=nom_next,
            cs_true=cs_true, cs_nom=cs_nom,
            obs_state=obs_state,
            inertia_true=state.inertia_true,
            mixer_true=state.mixer_true,
            external_force=state.external_force,
            force_freq=state.force_freq,
            force_phase=state.force_phase,
            k=state.k,
            prev_action_norm=action_norm,
            prev_action_req=action_req,
            hold_integral=saturated,
            step_idx=step_idx,
            key=state.key,
            ep_return=ep_return,
            ep_length=ep_length,
            ep_pos_err_sum=ep_pos_err_sum,
        )

        # --- auto-reset ---
        reset_key, carry_key = jax.random.split(state.key)
        fresh = reset(reset_key, ranges, 0)._replace(key=carry_key)
        new_state = jax.tree.map(
            lambda a, b: jnp.where(done, a, b), fresh, stepped
        )

        # Two different observations, and conflating them is a classic
        # auto-reset bug:
        #   obs           -> post-reset. What the agent acts on next.
        #   final_obs     -> pre-reset. The TRUE successor of (obs, action),
        #                    and the only correct thing to store in the replay
        #                    buffer. On a done step these differ completely:
        #                    the post-reset obs belongs to a fresh episode with
        #                    a different disturbance draw, so bootstrapping
        #                    from it learns a value function for the wrong
        #                    state.
        obs = obs_vector(new_state.obs_state, cfg)
        final_obs = obs_vector(stepped.obs_state, cfg)

        info = {
            "final_obs": final_obs,
            "episode_return": ep_return,
            "episode_length": ep_length,
            "episode_pos_mae": ep_pos_err_sum / jnp.maximum(ep_length, 1),
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            "pos_err_desired": pos_err_des,
            "pos_err_twin": pos_err_twin,
            "saturated": saturated,
            "reward_state": r_terms[0],
            "reward_effort": r_terms[1],
            "reward_smooth": r_terms[2],
            "k": state.k,
            "force_norm": jnp.linalg.norm(fext_t),
            "force_amp_norm": jnp.linalg.norm(state.external_force),
            "force_freq": state.force_freq,
            "ei_norm": jnp.linalg.norm(info_t.ei),
            "u_total": jnp.concatenate([jnp.atleast_1d(f_true), M_true]),
            "motor_cmd": motor_cmd,
            # Positions for trajectory plots: the true (disturbed) plant, the
            # disturbance-free nominal twin, and the reference being tracked.
            "x_true": true_next.x,
            "x_nom": nom_next.x,
            "x_des": xd,
            "u_base": u_base,
        }
        return new_state, obs, reward, done, info

    return reset, reset_obs, step


def _reward(err, action_norm, prev_action_norm, cfg):
    """Exponential tracking reward on the normalized twin discrepancy.

    ``err`` is already scale-normalized, so the tau values are dimensionless.
    """
    ep, ev, eR, ew = err[0:3], err[3:6], err[6:9], err[9:12]
    state_raw = (
        cfg.w_pos * jnp.exp(-jnp.dot(ep, ep) / cfg.tau_pos ** 2)
        + cfg.w_vel * jnp.exp(-jnp.dot(ev, ev) / cfg.tau_vel ** 2)
        + cfg.w_att * jnp.exp(-jnp.dot(eR, eR) / cfg.tau_att ** 2)
        + cfg.w_omega * jnp.exp(-jnp.dot(ew, ew) / cfg.tau_omega ** 2)
    )
    state_reward = state_raw / cfg.reward_norm
    effort = cfg.w_action_effort * jnp.linalg.norm(action_norm)
    smooth = cfg.w_action_smooth * jnp.linalg.norm(action_norm - prev_action_norm)
    return state_reward - effort - smooth, (state_reward, effort, smooth)


class VecEnv:
    """Thin vmap wrapper: N independent envs sharing one compiled graph."""

    def __init__(self, cfg, num_envs):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        _reset, _reset_obs, _step = make_env(cfg)
        self.obs_dim = cfg.obs_dim()
        self.action_dim = cfg.action_dim()

        # Ranges are mapped PER ENV (in_axes=0), not shared. That is what lets
        # a single batch contain a mixture of curriculum stages, which is how
        # rehearsal is implemented without running separate env pools.
        self._reset = jax.vmap(_reset, in_axes=(0, 0, 0))
        self._reset_obs = jax.vmap(_reset_obs)
        self._step = jax.vmap(_step, in_axes=(0, 0, 0))

    def reset(self, key, ranges, stagger=True):
        """``ranges`` must be a DisturbRanges whose fields are (num_envs,) arrays.

        ``stagger=True`` spreads the envs uniformly over the trajectory period
        so episode boundaries do not all land on the same iteration. Pass
        ``False`` for evaluation, where every episode should start at t=0 so
        the arms are directly comparable.
        """
        k_reset, k_phase = jax.random.split(key)
        keys = jax.random.split(k_reset, self.num_envs)
        if stagger:
            start = jax.random.randint(k_phase, (self.num_envs,), 0,
                                       self.cfg.episode_steps)
        else:
            start = jnp.zeros(self.num_envs, jnp.int32)
        state = self._reset(keys, ranges, start)
        return state, self._reset_obs(state)

    def step(self, state, action, ranges):
        return self._step(state, action, ranges)

    def broadcast_ranges(self, ranges: "DisturbRanges") -> "DisturbRanges":
        """Tile one scalar DisturbRanges across all envs."""
        return jax.tree.map(
            lambda v: jnp.broadcast_to(jnp.asarray(v), (self.num_envs,)), ranges
        )
