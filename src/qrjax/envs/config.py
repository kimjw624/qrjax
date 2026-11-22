"""Configuration for the residual-RL quadrotor task.

Plain dataclasses. Anything that changes the shape of a compiled graph
(history length, observation mode, number of envs, residual interface) is a
static Python value read at construction time; anything that only changes a
number (gain, disturbance range, reward weight) is baked in as a constant when
the env is built. Neither is traced, so changing a config value triggers a
recompile rather than being silently ignored.

Defaults (v2, after the paper). The defaults are the FIXED code at the paper's
task settings: 200 Hz, 10 s episodes, PID base, all five disturbance types,
every bug fix on. The submitted paper ran the ORIGINAL code; ``paper_env()`` /
``paper_config()`` rebuild exactly that, and ``legacy_env()`` the original
defaults (100 Hz, PD, mass/inertia only). Saved run configs from before v2 load
with the original values of every switch they predate (``Config.from_dict``),
so old checkpoints keep flying the code they were trained on.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple

ALL_DISTURBANCES = ("massmoi", "force", "motor_coeff", "moment_coeff", "arm_length")


@dataclass
class EnvConfig:
    # --- integration / episode ---
    dt: float = 0.005                      # 200 Hz (paper); original default 0.01
    episode_steps: int = 2000              # one 10 s period of the figure eight

    # --- nominal plant; the base controller always assumes these ---
    mass_nom: float = 2.0
    J_nom: Tuple[float, float, float] = (0.022, 0.022, 0.04)
    gravity: float = 9.807

    # --- trajectory ---
    trajectory: str = "figure8"
    traj_radius: float = 0.79
    traj_speed: float = 0.5
    traj_z0: float = 0.0

    # --- baseline controller ---
    # "pd" reproduces the legacy geometric controller; "pid" adds translational
    # and attitude integral action. Identical code path, different gains.
    base_controller: str = "pid"           # original default "pd"
    pid_ki: float = None                   # None -> kx / 10
    pid_kI: float = None                   # None -> kR / 10
    pid_antiwindup: bool = True            # freeze integrators while saturated
    # Clip on the translational integral state |e_i| (the reference tuning's
    # 1.0 caps the integral force at ki = 3.6 N per axis). Diagnostic: a larger
    # value shows how much of PID's error the clip alone explains.
    pid_ei_limit: float = 1.0

    # --- per-episode disturbances ---
    # Any subset of: massmoi, force, motor_coeff, moment_coeff, arm_length.
    # All listed types are sampled independently and applied for the whole
    # episode. Use ("none",) for a nominal plant. Default: all five (the
    # paper's flat training set); original default ("massmoi",).
    disturbances: Tuple[str, ...] = ALL_DISTURBANCES
    k_min: float = 0.7                     # m_true = k m_nom, J_true = k J_nom
    k_max: float = 1.3
    external_force_max: float = 3.0        # N, per NED axis (amplitude)
    # External force is A * cos(2 pi f t + phi) with f drawn from this band.
    # (0, 0) is a constant force with zero phase -- the default, and identical
    # to the constant-disturbance formulation.
    force_freq_min: float = 0.0
    force_freq_max: float = 0.0
    # Fraction of episodes with an exactly-constant force. Needed because
    # f ~ U[0, fmax] assigns zero probability to f = 0, so a continuous band
    # never trains the DC regime -- only its neighbourhood.
    force_dc_prob: float = 0.0
    motor_coeff_min: float = 0.7
    motor_coeff_max: float = 1.3
    moment_coeff_min: float = 0.7
    moment_coeff_max: float = 1.3
    arm_length_min: float = 0.7
    arm_length_max: float = 1.3
    per_motor_params: bool = False         # True -> independent scale per rotor
    # Diagnostic only. Allocate using the TRUE rotor constants and arm lengths
    # instead of the nominal ones, removing the commanded-to-produced moment
    # gain mismatch. Used to test whether that mismatch causes the attitude
    # oscillation; a real vehicle cannot do this.
    allocation_uses_true_params: bool = False

    # --- observation ---
    # "history"  : H stacked twin-discrepancy errors interleaved with H-1 past
    #              normalized actions -> H*12 + (H-1)*action_dim
    # "pid"      : proportional / leaky-integral / derivative of the twin
    #              discrepancy plus the normalized base wrench -> 40
    obs_mode: str = "history"
    history: int = 10
    pid_integral_leak: float = 0.99

    obs_pos_scale: float = 0.25
    obs_vel_scale: float = 0.15
    obs_att_scale: float = 0.25
    obs_omega_scale: float = 0.20

    # --- reward ---
    w_pos: float = 1.0
    w_vel: float = 0.5
    w_att: float = 0.5
    w_omega: float = 0.2
    tau_pos: float = 0.5
    tau_vel: float = 1.0
    tau_att: float = 1.0
    tau_omega: float = 1.5
    reward_norm: float = 2.2
    w_action_effort: float = 0.01
    w_action_smooth: float = 0.01

    # --- termination ---
    term_pos_error: float = 2.0            # m, twin discrepancy
    term_tilt_deg: float = 90.0

    # --- residual interface ---
    # 4-D wrench residual [df, dMx, dMy, dMz], added after the controller and
    # before control allocation.
    residual_authority: float = 0.20       # fraction of the wrench envelope

    # First-order low-pass on the THRUST channel only:
    #     u_t = (1 - beta) u_{t-1} + beta * a_t
    # beta = 0.2 is the empirically validated value; 1.0 disables filtering.
    #
    # Only the thrust channel is filtered, and the asymmetry is structural.
    # Thrust reaches the observed position error through two integrators, so
    # the policy closes a fast, high-gain loop on it at 100 Hz and can drive
    # alternating-sign chatter. Moments reach position through four
    # integrators, so the plant itself low-passes them and no such loop can
    # form. Set this to 1.0 to reproduce the oscillation deliberately.
    thrust_filter_beta: float = 0.2
    # Same first-order filter on the three moment channels. Left at 1.0
    # (disabled) because moments have never needed it -- the baseline attitude
    # controller opposes a full-authority moment residual roughly 11x more
    # strongly than it opposes a thrust residual, so the moment channel is
    # heavily damped by comparison. Exposed so that asymmetry can be tested
    # rather than assumed.
    moment_filter_beta: float = 1.0

    # --- fixes found while building qrjax_adapt / qrjax_ood ---------------
    # Defaults are the fixes (v2). LEGACY_ENV below holds the original values;
    # with them the code reproduces the submitted paper bit for bit (tested in
    # tests/test_fixes.py).
    #
    # Desired body rate: "finite_difference" (original) differentiates the
    # commanded attitude Rd twice numerically; "analytic" uses the exact
    # derivatives of Rd with an ideal accelerometer (true acceleration at the
    # current state under the previous step's wrench), as in qrjax_ood.
    rate_ff: str = "analytic"
    # Analytic rates only: acceleration fed to the feedforward. "accelerometer"
    # (the fix) = true acceleration under the previous wrench; "model" = the
    # nominal-model prediction from the controller's own thrust (diagnostic:
    # the residual's thrust then cannot reach omega_d through this path).
    accel_source: str = "accelerometer"
    # Finite-difference mode only: gate omega_d_dot on two previous samples,
    # removing the omega_d/dt impulse on the second step after every reset.
    fd_counter_fix: bool = True
    # "clip" (original) or "prioritized" (roll/pitch > collective > yaw).
    allocator: str = "prioritized"
    # "legacy" (original signs, a rotor relabelling) or "frd" (documented geometry).
    mixer_geometry: str = "frd"
    # Start each episode ON the reference: its attitude, body rate and a warm
    # controller memory, instead of R = I, omega = 0 and a cold controller.
    start_on_reference: bool = True
    # Extra terminations (0 = off, the original): true-vs-desired position
    # error [m] and heading error to b1d [deg]. The original terminates on the
    # twin discrepancy and tilt only.
    term_ref_error: float = 3.0
    term_heading_deg: float = 90.0

    # --- further switches, OFF by default (= original behaviour) ----------
    # Filters as a time constant [s] instead of the per-step beta (0 = use
    # beta). beta is per STEP, so one beta is a different filter at another
    # rate: beta 0.2 is tau 22.4 ms at 200 Hz but 44.8 ms at 100 Hz (qrjax_ood's
    # value). Shared with qrjax_ood's thrust_filter_tau.
    thrust_filter_tau: float = 0.0
    moment_filter_tau: float = 0.0
    # Translational-integrator bound: "clip" (original, ei_dot jumps to 0 at
    # the bound, which kicks omega_d through A_dot) or "smooth" (projection
    # operator, continuous ei_dot; pid_ei_eps sets the width of the zone where
    # the outward rate fades, from ei_limit / sqrt(1 + eps) to ei_limit).
    pid_ei_mode: str = "clip"
    pid_ei_eps: float = 0.2
    # Ramp the disturbances in over the first seconds of every episode
    # (0 = off: they act as a step at t = 0, which saturates the rotors for
    # the first steps even on a vehicle that starts on the reference).
    disturbance_ramp_s: float = 0.0

    def validate(self):
        """Raise on settings that would silently train something else."""
        if self.rate_ff not in ("finite_difference", "analytic", "none"):
            raise ValueError(f"rate_ff must be 'finite_difference', 'analytic' or 'none', "
                             f"got {self.rate_ff!r}")
        if self.accel_source not in ("accelerometer", "model"):
            raise ValueError(f"accel_source must be 'accelerometer' or 'model', "
                             f"got {self.accel_source!r}")
        if self.allocator not in ("clip", "prioritized"):
            raise ValueError(f"allocator must be 'clip' or 'prioritized', "
                             f"got {self.allocator!r}")
        if self.mixer_geometry not in ("legacy", "frd"):
            raise ValueError(f"mixer_geometry must be 'legacy' or 'frd', "
                             f"got {self.mixer_geometry!r}")
        if self.base_controller not in ("pd", "pid"):
            raise ValueError(f"base_controller must be 'pd' or 'pid'")
        if not 0.0 < self.thrust_filter_beta <= 1.0 \
                or not 0.0 < self.moment_filter_beta <= 1.0:
            raise ValueError("filter betas must be in (0, 1]")
        if not 0.0 <= self.residual_authority <= 1.0:
            raise ValueError("residual_authority must be in [0, 1]")
        if min(self.tau_pos, self.tau_vel, self.tau_att, self.tau_omega) <= 0:
            raise ValueError("reward tau values must be positive")
        if self.term_ref_error < 0 or not 0.0 <= self.term_heading_deg <= 180.0:
            raise ValueError("term_ref_error >= 0 and 0 <= term_heading_deg <= 180")
        if self.dt <= 0 or self.episode_steps <= 0:
            raise ValueError("dt and episode_steps must be positive")
        if self.pid_ei_mode not in ("clip", "smooth"):
            raise ValueError(f"pid_ei_mode must be 'clip' or 'smooth', got {self.pid_ei_mode!r}")
        if self.pid_ei_eps <= 0:
            raise ValueError("pid_ei_eps must be positive")
        if min(self.thrust_filter_tau, self.moment_filter_tau, self.disturbance_ramp_s) < 0:
            raise ValueError("filter time constants and disturbance_ramp_s must be >= 0")
        unknown = set(self.disturbances) - set(ALL_DISTURBANCES) - {"none"}
        if unknown:
            raise ValueError(f"unknown disturbance type(s) {sorted(unknown)}")
        return self

    def action_dim(self) -> int:
        return 4

    def obs_dim(self) -> int:
        if self.obs_mode == "history":
            return self.history * 12 + (self.history - 1) * self.action_dim()
        if self.obs_mode == "pid":
            return 36 + 4
        raise ValueError(f"unknown obs_mode {self.obs_mode!r}")


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 1_000_000
    batch_size: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    learning_starts: int = 5_000
    policy_frequency: int = 2
    target_frequency: int = 1
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    target_entropy_scale: float = 0.5
    grad_clip: float = 1.0
    hidden: Tuple[int, ...] = (256, 256)
    critic_layernorm: bool = True


@dataclass
class TrainConfig:
    total_steps: int = 1_000_000           # total ENV steps across all envs
    num_envs: int = 64
    rollout_len: int = 16                  # env steps per env per iteration
    # Gradient steps per collected transition. 1.0 matches the sequential
    # PyTorch reference (one update per env step); lower values trade sample
    # efficiency for wall-clock speed and are usually the right call once
    # num_envs is large.
    utd: float = 1.0
    seed: int = 0

    # --- curriculum --- (default flat, as in the paper; original default True)
    use_curriculum: bool = False
    curriculum_path: str = "configs/curriculum.toml"

    # --- evaluation during training ---
    eval_every_iters: int = 25
    eval_episodes: int = 16

    # --- logging / checkpointing ---
    log_every_iters: int = 1
    checkpoint_every_iters: int = 100
    runs_root: str = "runs"
    run_name: str = "residual_sac"
    live_plot: bool = False

    # --- evaluation-protocol fixes (defaults = fixes; originals in LEGACY_TRAIN) ---
    # Use the SAME disturbance draw (eval_seed) for every in-training
    # evaluation instead of a fresh one each time, so checkpoint scores are
    # comparable across iterations.
    eval_fixed_key: bool = True
    eval_seed: int = 20260829
    # "best" checkpoint = lowest mean of the last N evaluations (1 = the
    # original single-best rule, which on a plateau selects evaluation noise).
    best_window: int = 3


# Every bug fix (all are defaults since v2; kept to switch them on in a legacy config).
FIXED_ENV = {
    "rate_ff": "analytic",
    "fd_counter_fix": True,
    "allocator": "prioritized",
    "mixer_geometry": "frd",
    "start_on_reference": True,
    "term_ref_error": 3.0,
    "term_heading_deg": 90.0,
}
FIXED_TRAIN = {"eval_fixed_key": True, "best_window": 3}


# Original values of every setting whose default changed in v2, plus the
# switches added later (all off). With these the code is the submitted paper's.
LEGACY_ENV = {
    "dt": 0.01, "episode_steps": 1000, "base_controller": "pd",
    "disturbances": ("massmoi",),
    "rate_ff": "finite_difference", "fd_counter_fix": False, "allocator": "clip",
    "mixer_geometry": "legacy", "start_on_reference": False,
    "term_ref_error": 0.0, "term_heading_deg": 0.0,
    "thrust_filter_tau": 0.0, "moment_filter_tau": 0.0,
    "pid_ei_mode": "clip", "pid_ei_eps": 0.2, "disturbance_ramp_s": 0.0,
}
LEGACY_TRAIN = {"use_curriculum": True, "eval_fixed_key": False, "best_window": 1}
# The submitted paper: original code at its task settings (run_final_study).
PAPER_ENV = {**LEGACY_ENV, "dt": 0.005, "episode_steps": 2000,
             "base_controller": "pid", "disturbances": ALL_DISTURBANCES}
PAPER_TRAIN = {**LEGACY_TRAIN, "use_curriculum": False}


def _with(cls_or_obj, values, overrides):
    out = cls_or_obj() if isinstance(cls_or_obj, type) else cls_or_obj
    for k, v in {**values, **overrides}.items():
        if not hasattr(out, k):
            raise KeyError(f"unknown {type(out).__name__} field {k!r}")
        setattr(out, k, v)
    return out


def legacy_env(**overrides) -> "EnvConfig":
    """EnvConfig() as it was before v2: original code, original defaults."""
    return _with(EnvConfig, LEGACY_ENV, overrides)


def paper_env(**overrides) -> "EnvConfig":
    """The submitted paper's environment: original code, 200 Hz, 10 s, PID,
    all five disturbance types. Pass base_controller="pd" for the PD arms."""
    return _with(EnvConfig, PAPER_ENV, overrides)


def paper_config() -> "Config":
    """Config of the paper's training runs (environment and training protocol)."""
    cfg = Config()
    _with(cfg.env, PAPER_ENV, {})
    _with(cfg.train, PAPER_TRAIN, {})
    return cfg


def fixed_preset(env: "EnvConfig") -> "EnvConfig":
    """Return a copy of ``env`` with every bug fix switched on."""
    out = EnvConfig(**{**env.__dict__})
    for k, v in FIXED_ENV.items():
        setattr(out, k, v)
    return out


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)

    # Config keys that were renamed or removed. Loading an old run's
    # config.json must not crash, or every checkpoint predating a refactor
    # becomes unloadable.
    _RENAMED = {"wrench_thrust_filter_beta": "thrust_filter_beta"}
    _REMOVED = ("residual_interface", "force_vector_limit_N",
                "force_vector_filter_beta")

    @staticmethod
    def _clean(section, cls, warn_prefix, legacy=None):
        import dataclasses
        import warnings
        fields = {f.name for f in dataclasses.fields(cls)}
        out, dropped = {}, []
        for k, v in (section or {}).items():
            k = Config._RENAMED.get(k, k)
            if isinstance(v, list):
                v = tuple(v)
            if k in fields:
                out[k] = v
            else:
                dropped.append(k)
        # A key the saved config does not have predates it: give it the value
        # the code had when the config was written (the original), not the
        # current default -- otherwise loading an old run would silently
        # switch on fixes it was never trained with.
        if section is not None:
            for k, v in (legacy or {}).items():
                if k in fields and k not in out:
                    out[k] = tuple(v) if isinstance(v, list) else v
        if dropped:
            warnings.warn(
                f"{warn_prefix}: ignoring unknown config key(s) "
                f"{sorted(dropped)}. This config predates a refactor; the "
                f"corresponding defaults will be used instead.",
                stacklevel=2,
            )
        return cls(**out)

    @staticmethod
    def from_dict(d):
        return Config(
            env=Config._clean(d.get("env"), EnvConfig, "env", LEGACY_ENV),
            sac=Config._clean(d.get("sac"), SACConfig, "sac"),
            train=Config._clean(d.get("train"), TrainConfig, "train", LEGACY_TRAIN),
        )
