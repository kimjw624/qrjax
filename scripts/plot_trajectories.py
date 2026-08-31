"""3D trajectory and actuator plots for all four arms on matched uncertainty cases.

For each selected case the same disturbance draw is run through:

    pd        PD base, no residual
    pid       PID base, no residual
    pd_res    PD base  + PD-trained policy
    pid_res   PID base + PID-trained policy

and three figures are produced:

  trajectory_<case>.png   3D desired / nominal-twin / true path per arm
  actuators_<case>.png    per-rotor thrust commands, full plus a 1 s zoom
  errors_<case>.png       tracking error, twin discrepancy, residual action

Why plot the nominal twin as well as the desired path: the twin is the
disturbance-free reference the residual is trained to match, so the gap
"desired vs twin" is what the BASE controller cannot do, while "twin vs true"
is what the DISTURBANCE does and what the residual is trying to remove. Showing
only desired-vs-true conflates the two.

Cases are selected to span the uncertainty space rather than at random, and
each is labelled with its effective attitude gain g = kf·arm/k, since that is
what governs whether the baseline stays stable.

Example
-------
python -m scripts.plot_trajectories \\
    --run_dir_pd  experiments/stage1_.../seed_00/pd/train/trial_001 \\
    --run_dir_pid experiments/stage1_.../seed_00/pid/train/trial_001 \\
    --n_cases 4
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from mpl_toolkits.mplot3d import Axes3D             # noqa: F401,E402
import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
import numpy as np                                   # noqa: E402

from qrjax.envs import Config, EnvConfig, VecEnv     # noqa: E402
from qrjax.rl import SAC                             # noqa: E402
from qrjax.rl.curriculum import flat_ranges          # noqa: E402
from qrjax.utils import load_params, write_manifest  # noqa: E402

ARMS = ("pd", "pid", "pd_res", "pid_res")
LABEL = {"pd": "PD", "pid": "PID", "pd_res": "PD + residual",
         "pid_res": "PID + residual"}
CONTROLLER = {"pd": "pd", "pid": "pid", "pd_res": "pd", "pid_res": "pid"}
USES_RESIDUAL = {"pd": False, "pid": False, "pd_res": True, "pid_res": True}
ROTOR_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]


def load_policy(run_dir, checkpoint="best"):
    run_dir = Path(run_dir)
    cfg = Config.from_dict(json.loads((run_dir / "config.json").read_text()))
    ckpt = run_dir / "checkpoints" / (
        checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt")
    probe = VecEnv(cfg.env, 1)
    agent = SAC(probe.obs_dim, probe.action_dim, cfg.sac)
    params = load_params(ckpt, agent.init(jax.random.PRNGKey(0)).actor_params)
    return cfg, agent, params


def rollout_all(env_cfg, agent_pd, params_pd, agent_pid, params_pid,
                key, episodes, ranges):
    """Run all four arms on the same episodes; return traces plus the draw."""
    out = {}
    draw = None
    for arm in ARMS:
        cfg = EnvConfig(**{**env_cfg.__dict__})
        cfg.base_controller = CONTROLLER[arm]
        env = VecEnv(cfg, episodes)
        batched = env.broadcast_ranges(ranges)
        agent = agent_pd if arm == "pd_res" else agent_pid
        params = params_pd if arm == "pd_res" else params_pid
        use = USES_RESIDUAL[arm]

        def run(key, use=use, agent=agent, params=params, env=env,
                batched=batched, cfg=cfg):
            k_reset, _ = jax.random.split(key)
            state, obs = env.reset(k_reset, batched, stagger=False)
            info0 = {"k": state.k,
                     "force": state.external_force,
                     "kf": state.mixer_true.kf_scale[:, 0],
                     "arm": state.mixer_true.arm_scale[:, 0],
                     "moment": state.mixer_true.moment_scale[:, 0]}

            def body(carry, _):
                state, obs, alive = carry
                if use:
                    action = jnp.tanh(agent.actor.apply(params, obs)[0])
                else:
                    action = jnp.zeros((episodes, env.action_dim))
                state, obs, reward, done, info = env.step(state, action, batched)
                m = alive.astype(jnp.float32)
                return (state, obs,
                        jnp.logical_and(alive,
                                        jnp.logical_not(info["terminated"]))), (
                    info["x_true"], info["x_nom"], info["x_des"],
                    info["motor_cmd"], action, m,
                    info["pos_err_desired"], info["pos_err_twin"])

            (_, _, alive), tr = jax.lax.scan(
                body, (state, obs, jnp.ones(episodes, bool)), None,
                length=cfg.episode_steps)
            return tr, info0, alive

        tr, info0, alive = jax.jit(run)(key)
        out[arm] = tuple(np.asarray(x) for x in tr) + (np.asarray(alive),)
        if draw is None:
            draw = {k: np.asarray(v) for k, v in info0.items()}
    return out, draw


def plot_trajectory(case, traces, draw, out_path, title_extra=""):
    fig = plt.figure(figsize=(15, 11))
    for i, arm in enumerate(ARMS):
        x_true, x_nom, x_des, motor, act, mask, pe, pt, alive = traces[arm]
        n = max(int(mask[:, case].sum()), 2)
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        # NED -> plot with z up for readability
        ax.plot(x_des[:n, case, 0], x_des[:n, case, 1], -x_des[:n, case, 2],
                color="k", ls="--", lw=1.4, label="desired")
        ax.plot(x_nom[:n, case, 0], x_nom[:n, case, 1], -x_nom[:n, case, 2],
                color="tab:green", lw=1.2, alpha=0.85, label="nominal twin")
        ax.plot(x_true[:n, case, 0], x_true[:n, case, 1], -x_true[:n, case, 2],
                color="tab:red", lw=1.2, alpha=0.9, label="true (disturbed)")
        ax.scatter(*[x_true[0, case, j] * (1 if j < 2 else -1) for j in range(3)],
                   color="tab:red", s=25)
        rmse = float(np.sqrt(np.mean(pe[:n, case] ** 2)))
        ax.set_title(f"{LABEL[arm]}\nRMSE {rmse:.4f} m"
                     f"{'  TERMINATED' if not alive[case] else ''}", fontsize=10)
        ax.set_xlabel("N [m]", fontsize=8)
        ax.set_ylabel("E [m]", fontsize=8)
        ax.set_zlabel("up [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Case {case}: {title_extra}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_actuators(case, traces, out_path, dt, title_extra="", zoom_s=1.0):
    fig, axes = plt.subplots(4, 2, figsize=(15, 13))
    for i, arm in enumerate(ARMS):
        x_true, x_nom, x_des, motor, act, mask, pe, pt, alive = traces[arm]
        n = max(int(mask[:, case].sum()), 8)
        t = np.arange(n) * dt
        nz = min(int(zoom_s / dt), n)
        z0 = max(0, n // 2 - nz // 2)

        ax = axes[i, 0]
        for r in range(4):
            ax.plot(t, motor[:n, case, r], lw=0.6, color=ROTOR_COLORS[r])
        ax.axvspan(t[z0], t[z0 + nz - 1], color="grey", alpha=0.15)
        ax.set_ylabel(f"{LABEL[arm]}\nrotor cmd [N]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title("full episode", fontsize=10)

        ax = axes[i, 1]
        for r in range(4):
            ax.plot(t[z0:z0 + nz], motor[z0:z0 + nz, case, r], lw=1.0,
                    marker=".", ms=2, color=ROTOR_COLORS[r])
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title(f"{zoom_s:g} s zoom", fontsize=10)
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    fig.suptitle(f"Case {case} actuators: {title_extra}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_errors(case, traces, out_path, dt, title_extra=""):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for arm in ARMS:
        x_true, x_nom, x_des, motor, act, mask, pe, pt, alive = traces[arm]
        n = max(int(mask[:, case].sum()), 2)
        t = np.arange(n) * dt
        axes[0].plot(t, pe[:n, case], lw=1.1, label=LABEL[arm])
        axes[1].plot(t, pt[:n, case], lw=1.1, label=LABEL[arm])
        if USES_RESIDUAL[arm]:
            axes[2].plot(t, act[:n, case, 0], lw=1.0,
                         label=f"{LABEL[arm]} thrust")
    axes[0].set_ylabel("‖x − x_des‖ [m]")
    axes[0].set_title("tracking error vs the reference")
    axes[1].set_ylabel("‖x_nom − x_true‖ [m]")
    axes[1].set_title("twin discrepancy — what the residual is trained to remove")
    axes[2].set_ylabel("normalized thrust residual")
    axes[2].set_title("residual thrust command")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Case {case}: {title_extra}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir_pd", required=True)
    p.add_argument("--run_dir_pid", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--episodes", type=int, default=64,
                   help="bank size to select cases from")
    p.add_argument("--n_cases", type=int, default=4)
    p.add_argument("--cases", default=None,
                   help="explicit comma-separated episode indices")
    p.add_argument("--force_freq", type=float, default=0.0)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--zoom_s", type=float, default=1.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg, agent_pd, params_pd = load_policy(args.run_dir_pd, args.checkpoint)
    _, agent_pid, params_pid = load_policy(args.run_dir_pid, args.checkpoint)

    out = Path(args.out) if args.out else Path("experiments") / (
        f"trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    ranges = flat_ranges(cfg.env, force_freq=args.force_freq, force_dc_prob=0.0)
    key = jax.random.PRNGKey(args.eval_seed)
    traces, draw = rollout_all(cfg.env, agent_pd, params_pd, agent_pid,
                               params_pid, key, args.episodes, ranges)

    g = draw["kf"] * draw["arm"] / draw["k"]
    fmag = np.linalg.norm(draw["force"], axis=-1)

    if args.cases:
        cases = [int(x) for x in args.cases.split(",") if x.strip()]
    else:
        # Span the uncertainty space rather than sampling at random: the
        # benign corner, the largest external force, the largest mass error,
        # and the highest effective attitude gain (where the baseline is
        # closest to its stability limit).
        cases = []
        cases.append(int(np.argmin(g + fmag)))                    # benign
        cases.append(int(np.argmax(fmag)))                        # max force
        cases.append(int(np.argmax(np.abs(draw["k"] - 1.0))))     # max mass err
        cases.append(int(np.argmax(g)))                           # max gain
        seen, uniq = set(), []
        for c in cases:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        cases = uniq[:args.n_cases]

    print(f"PD policy  : {args.run_dir_pd}")
    print(f"PID policy : {args.run_dir_pid}")
    print(f"cases      : {cases}\n")

    summary = []
    for c in cases:
        extra = (f"k={draw['k'][c]:.2f}  |F|={fmag[c]:.2f} N  "
                 f"kf={draw['kf'][c]:.2f}  arm={draw['arm'][c]:.2f}  "
                 f"g=kf·arm/k={g[c]:.2f}")
        print(f"  case {c:>3}  {extra}")
        plot_trajectory(c, traces, draw, out / f"trajectory_case{c:03d}.png", extra)
        plot_actuators(c, traces, out / f"actuators_case{c:03d}.png",
                       cfg.env.dt, extra, args.zoom_s)
        plot_errors(c, traces, out / f"errors_case{c:03d}.png",
                    cfg.env.dt, extra)
        row = {"case": c, "k": float(draw["k"][c]), "force": float(fmag[c]),
               "kf": float(draw["kf"][c]), "arm": float(draw["arm"][c]),
               "g": float(g[c])}
        for arm in ARMS:
            *_, mask, pe, pt, alive = traces[arm]
            n = max(int(mask[:, c].sum()), 2)
            row[f"rmse_{arm}"] = float(np.sqrt(np.mean(pe[:n, c] ** 2)))
            row[f"terminated_{arm}"] = bool(not alive[c])
            print(f"      {LABEL[arm]:<16} RMSE {row[f'rmse_{arm}']:.4f} m")
        summary.append(row)

    import csv
    with (out / "cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    write_manifest(out, {"run_dir_pd": args.run_dir_pd,
                         "run_dir_pid": args.run_dir_pid,
                         "cases": cases, "force_freq": args.force_freq,
                         "eval_seed": args.eval_seed})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
