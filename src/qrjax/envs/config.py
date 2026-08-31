"""Configuration for the residual-RL quadrotor task.

Plain dataclasses. Anything that changes the shape of a compiled graph
(history length, observation mode, number of envs, residual interface) is a
static Python value read at construction time; anything that only changes a
number (gain, disturbance range, reward weight) is baked in as a constant when
the env is built. Neither is traced, so changing a config value triggers a
recompile rather than being silently ignored.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class EnvConfig:
    # --- integration / episode ---
    dt: float = 0.01
    episode_steps: int = 1000              # one 10 s period of the figure eight

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
    base_controller: str = "pd"
    pid_ki: float = None                   # None -> kx / 10
    pid_kI: float = None                   # None -> kR / 10
    pid_antiwindup: bool = True            # freeze integrators while saturated

    # --- per-episode disturbances ---
    # Any subset of: massmoi, force, motor_coeff, moment_coeff, arm_length.
    # All listed types are sampled independently and applied for the whole
    # episode. Use ("none",) for a nominal plant.
    disturbances: Tuple[str, ...] = ("massmoi",)
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

    # --- curriculum ---
    use_curriculum: bool = True
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
    def _clean(section, cls, warn_prefix):
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
            env=Config._clean(d.get("env"), EnvConfig, "env"),
            sac=Config._clean(d.get("sac"), SACConfig, "sac"),
            train=Config._clean(d.get("train"), TrainConfig, "train"),
        )
