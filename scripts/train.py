"""Train a residual SAC policy on the twin-plant quadrotor task.

Examples
--------
# Flat training: every disturbance active from step zero, no staging.
python -m scripts.train --no_curriculum --total_steps 1000000 --live_plot

# Six-stage curriculum (budget comes from the TOML, not --total_steps).
python -m scripts.train --curriculum --live_plot

# Fast sweep settings: many envs, few gradient steps per transition.
python -m scripts.train --no_curriculum --num_envs 512 --utd 0.0625

# PID baseline underneath the residual instead of PD.
python -m scripts.train --no_curriculum --base_controller pid
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qrjax.envs import Config          # noqa: E402
from qrjax.train import train          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    g = p.add_argument_group("curriculum")
    mode = g.add_mutually_exclusive_group()
    mode.add_argument("--curriculum", dest="use_curriculum", action="store_true",
                      help="six-stage curriculum (default)")
    mode.add_argument("--no_curriculum", dest="use_curriculum", action="store_false",
                      help="flat training: all disturbances at full range from step 0")
    p.set_defaults(use_curriculum=True)
    g.add_argument("--curriculum_path", default="configs/curriculum.toml")

    g = p.add_argument_group("budget and parallelism")
    g.add_argument("--total_steps", type=int, default=1_000_000,
                   help="total env steps; ignored with --curriculum "
                        "(the TOML stage durations define the budget)")
    g.add_argument("--num_envs", type=int, default=64)
    g.add_argument("--rollout_len", type=int, default=16)
    g.add_argument("--utd", type=float, default=1.0,
                   help="gradient steps per collected transition. 1.0 matches "
                        "the sequential PyTorch reference; lower is much faster "
                        "and usually fine once num_envs is large")
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("task")
    g.add_argument("--base_controller", choices=("pd", "pid"), default="pd")
    g.add_argument("--thrust_filter_beta", type=float, default=0.2,
                   help="first-order low-pass on the thrust residual: "
                        "u = (1-beta) u_prev + beta a. 1.0 disables it, which "
                        "reproduces the thrust-channel oscillation")
    g.add_argument("--moment_filter_beta", type=float, default=1.0,
                   help="same filter on the three moment channels. 1.0 "
                        "disables it, which is the default and the setting "
                        "used in every experiment so far")
    g.add_argument("--residual_authority", type=float, default=0.20,
                   help="residual limit as a fraction of the wrench envelope")
    g.add_argument("--obs_mode", choices=("history", "pid"), default="history")
    g.add_argument("--history", type=int, default=10,
                   help="observation history length in steps (dt=0.01, so 10 "
                        "is 0.1 s). Rejecting a constant disturbance needs a "
                        "long accumulation window; rejecting a 4 Hz one needs "
                        "a short reaction time. Changes obs_dim, so a policy "
                        "is only loadable by a run with the same value")
    g.add_argument("--trajectory", choices=("figure8", "circle", "hover"),
                   default="figure8")
    g.add_argument("--control_hz", type=float, default=100.0,
                   help="control AND integration rate. The attitude loop's "
                        "tolerable gain mismatch scales as 1/dt: g_crit = "
                        "2J/(dt*kOmega) is 1.76 at 100 Hz but 3.46 at 200 Hz. "
                        "The disturbance box only reaches g = 2.41, so 200 Hz "
                        "removes the gain-margin failure entirely")
    g.add_argument("--episode_seconds", type=float, default=10.0,
                   help="episode length in SECONDS. episode_steps is derived "
                        "from this and --control_hz, so changing the rate "
                        "keeps the physical episode the same length")
    g.add_argument("--episode_steps", type=int, default=None,
                   help="override the derived step count; normally leave unset")
    g.add_argument("--force_freq_min", type=float, default=0.0,
                   help="external force is A cos(2 pi f t + phi) with f drawn "
                        "from [min, max] Hz. Both 0 (the default) is a "
                        "constant force")
    g.add_argument("--force_freq_max", type=float, default=0.0)
    g.add_argument("--force_dc_prob", type=float, default=0.0,
                   help="fraction of episodes with an exactly-constant force. "
                        "f ~ U[0, max] assigns ZERO probability to f = 0, so a "
                        "band alone never trains the DC regime. Use ~0.3 to "
                        "train constant and time-varying together")

    g = p.add_argument_group("SAC")
    g.add_argument("--batch_size", type=int, default=256)
    g.add_argument("--buffer_size", type=int, default=300_000,
                   help="transitions held on device; 1e6 x 156-D float32 is "
                        "~1.3 GB, so keep this modest on a laptop GPU")
    g.add_argument("--learning_starts", type=int, default=5_000)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--hidden", default="256,256",
                   type=lambda v: tuple(int(x) for x in v.split(",") if x.strip()),
                   help="actor/critic hidden layer sizes. '512,512' doubles "
                        "width; '256,256,256,256' doubles depth. Unlike "
                        "--history this leaves obs_dim unchanged, so policies "
                        "with different sizes can share one evaluation sweep")

    g = p.add_argument_group("output")
    g.add_argument("--runs_root", default="runs")
    g.add_argument("--run_name", default=None,
                   help="default: residual_sac_<curriculum|flat>_<controller>")
    g.add_argument("--live_plot", action="store_true",
                   help="open an interactive training-curve window; "
                        "progress.png is written either way")
    g.add_argument("--eval_every_iters", type=int, default=25)
    g.add_argument("--eval_episodes", type=int, default=128,
                   help="one batched rollout, so this is nearly free. Below "
                        "~128 the termination count is too coarse to resolve a "
                        "few-percent failure rate")
    g.add_argument("--checkpoint_every_iters", type=int, default=100)
    g.add_argument("--x64", action="store_true",
                   help="float64 everywhere; matches the NumPy reference bit "
                        "for bit but roughly halves throughput")
    return p.parse_args()


def main():
    args = parse_args()

    if args.x64:
        import jax
        jax.config.update("jax_enable_x64", True)

    cfg = Config()
    cfg.env.base_controller = args.base_controller
    cfg.env.thrust_filter_beta = args.thrust_filter_beta
    cfg.env.moment_filter_beta = args.moment_filter_beta
    cfg.env.residual_authority = args.residual_authority
    cfg.env.obs_mode = args.obs_mode
    cfg.env.history = args.history
    cfg.env.trajectory = args.trajectory
    cfg.env.dt = 1.0 / args.control_hz
    cfg.env.episode_steps = (args.episode_steps if args.episode_steps
                             else int(round(args.control_hz * args.episode_seconds)))
    cfg.env.force_freq_min = args.force_freq_min
    cfg.env.force_freq_max = args.force_freq_max
    cfg.env.force_dc_prob = args.force_dc_prob

    cfg.sac.batch_size = args.batch_size
    cfg.sac.buffer_size = args.buffer_size
    cfg.sac.learning_starts = args.learning_starts
    cfg.sac.hidden = args.hidden
    cfg.sac.lr_actor = cfg.sac.lr_critic = cfg.sac.lr_alpha = args.lr

    cfg.train.use_curriculum = args.use_curriculum
    cfg.train.curriculum_path = args.curriculum_path
    cfg.train.total_steps = args.total_steps
    cfg.train.num_envs = args.num_envs
    cfg.train.rollout_len = args.rollout_len
    cfg.train.utd = args.utd
    cfg.train.seed = args.seed
    cfg.train.runs_root = args.runs_root
    cfg.train.eval_every_iters = args.eval_every_iters
    cfg.train.eval_episodes = args.eval_episodes
    cfg.train.checkpoint_every_iters = args.checkpoint_every_iters
    cfg.train.run_name = args.run_name or (
        f"residual_sac_{'curriculum' if args.use_curriculum else 'flat'}"
        f"_{args.base_controller}"
        + (f"_f{args.force_freq_min:g}-{args.force_freq_max:g}Hz"
           if args.force_freq_max > 0 else "")
        + (f"_dc{args.force_dc_prob:g}" if args.force_dc_prob > 0 else "")
        + (f"_h{args.history}" if args.history != 10 else "")
        + (f"_{args.control_hz:g}Hz" if args.control_hz != 100.0 else "")
        + ("_net" + "x".join(str(h) for h in args.hidden)
           if tuple(args.hidden) != (256, 256) else "")
    )

    train(cfg, live_plot=args.live_plot)


if __name__ == "__main__":
    main()
