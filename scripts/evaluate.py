"""Paired evaluation of the baseline controllers, with and without the residual.

Arms:

    pd        base PD controller, zero residual
    pid       base PID controller, zero residual
    pd_res    base PD  + trained residual policy
    pid_res   base PID + trained residual policy

Pairing is exact. Every arm is reset from the same PRNG key with the same
disturbance ranges, so episode i sees an identical (k, F_ext, actuator scales)
draw in every arm. The script verifies that rather than assuming it, and aborts
if the draws diverge.

Note on ``pid_res``: a policy trained on the PD baseline is running on a PID
baseline it never saw in training. That is a transfer result, not a
like-for-like comparison, and the report says so next to the number. To make
the claim properly, retrain with ``--base_controller pid``.

Examples
--------
# No trained policy required.
python -m scripts.evaluate --arms pd,pid --episodes 64 --out evaluations/pd_vs_pid

# Three-way against a trained checkpoint.
python -m scripts.evaluate \
    --run_dir runs/residual_sac_flat_pd/trial_001 --checkpoint best \
    --arms pd,pid,pd_res --episodes 64
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import numpy as np                                            # noqa: E402

from qrjax.envs import Config, EnvConfig, VecEnv               # noqa: E402
from qrjax.rl import SAC                                       # noqa: E402
from qrjax.rl.curriculum import flat_ranges, load_curriculum    # noqa: E402
from qrjax.utils import load_params, write_json, write_manifest  # noqa: E402


ARMS = ("pd", "pid", "pd_res", "pid_res")
LABEL = {"pd": "PD", "pid": "PID", "pd_res": "PD + residual",
         "pid_res": "PID + residual"}
CONTROLLER = {"pd": "pd", "pid": "pid", "pd_res": "pd", "pid_res": "pid"}
USES_RESIDUAL = {"pd": False, "pid": False, "pd_res": True, "pid_res": True}


def rollout_arm(env_cfg, controller, agent, actor_params, key, episodes, ranges):
    """Run one arm for ``episodes`` full episodes in parallel.

    Returns per-episode metrics and the mean position-error time series.
    """
    cfg = EnvConfig(**{**env_cfg.__dict__})
    cfg.base_controller = controller
    env = VecEnv(cfg, episodes)
    batched = env.broadcast_ranges(ranges)

    def run(key):
        k_reset, k_run = jax.random.split(key)
        # stagger=False: every episode starts at t=0 so all arms see identical
        # episodes and each covers the whole trajectory.
        state, obs = env.reset(k_reset, batched, stagger=False)

        # Fingerprint the disturbance at RESET, not at the end of the scan.
        # The env auto-resets on termination, so a final-state fingerprint
        # describes whichever episode happened to be running last -- which
        # differs between arms precisely when they fail at different times,
        # exactly the case the pairing check exists to catch.
        # Include the TEMPORAL profile, not just the plant parameters. Without
        # force_freq and force_phase here, two arms could be run on the same
        # vehicles under different disturbance waveforms and the pairing check
        # would pass.
        fingerprint = jnp.concatenate([
            state.k[:, None], state.external_force,
            state.force_freq[:, None], state.force_phase,
            state.mixer_true.kf_scale, state.mixer_true.moment_scale,
            state.mixer_true.arm_scale,
        ], axis=1)

        def body(carry, _):
            state, obs, alive = carry
            if agent is None:
                action = jnp.zeros((episodes, env.action_dim))
            else:
                mean, _ = agent.actor.apply(actor_params, obs)
                action = jnp.tanh(mean)
            state, obs, reward, done, info = env.step(state, action, batched)
            # Freeze each env at its first termination, and charge the failure
            # at the termination threshold for the rest of the episode. Without
            # this, auto-reset splices a failed episode onto a fresh one with a
            # DIFFERENT disturbance and reports the blend as one episode.
            err = jnp.where(alive, info["pos_err_desired"], cfg.term_pos_error)
            twin = jnp.where(alive, info["pos_err_twin"], cfg.term_pos_error)
            alive_next = jnp.logical_and(alive,
                                         jnp.logical_not(info["terminated"]))
            m = alive.astype(jnp.float32)
            return (state, obs, alive_next), (
                err, twin, reward * m, info["terminated"],
                info["u_total"] * m[:, None], info["saturated"] * m,
                action * m[:, None],
            )

        alive0 = jnp.ones(episodes, dtype=bool)
        (_, _, alive), out = jax.lax.scan(
            body, (state, obs, alive0), None, length=cfg.episode_steps)
        return out, fingerprint, alive

    (pos_des, pos_twin, reward, term, wrench, sat, act), fingerprint, alive = \
        jax.jit(run)(key)

    pos_des = np.asarray(pos_des)          # (T, E)
    tail = pos_des.shape[0] // 2
    dwrench = np.diff(np.asarray(wrench), axis=0)

    act = np.asarray(act)                  # (T, E, 4)

    def lag1_autocorr(x):
        """Lag-1 autocorrelation per episode, per action channel.

        Strongly negative means the signal alternates sign every step, which is
        the signature of the Nyquist-frequency chatter seen on the thrust
        channel. Near zero means a smooth command.
        """
        x = x - x.mean(axis=0, keepdims=True)
        num = np.sum(x[:-1] * x[1:], axis=0)
        den = np.sum(x ** 2, axis=0) + 1e-12
        return num / den

    ac = lag1_autocorr(act)                # (E, 4)

    per_ep = {
        "pos_rmse": np.sqrt(np.mean(pos_des ** 2, axis=0)),
        "pos_steady": np.mean(pos_des[tail:], axis=0),
        "pos_max": np.max(pos_des, axis=0),
        "twin_rmse": np.sqrt(np.mean(np.asarray(pos_twin) ** 2, axis=0)),
        "return": np.sum(np.asarray(reward), axis=0),
        "terminated": 1.0 - np.asarray(alive).astype(float),
        "wrench_rms": np.sqrt(np.mean(np.sum(np.asarray(wrench) ** 2, -1), axis=0)),
        "smoothness": np.sqrt(np.mean(np.sum(dwrench ** 2, -1), axis=0)),
        "saturation_frac": np.mean(np.asarray(sat), axis=0),
        # Chatter diagnostics: thrust vs moments, the asymmetry that motivates
        # filtering only the thrust channel.
        "chatter_thrust": ac[:, 0],
        "chatter_moment": ac[:, 1:].mean(axis=1),
        "residual_thrust_absmean": np.mean(np.abs(act[:, :, 0]), axis=0),
    }
    return per_ep, pos_des.mean(axis=1), np.asarray(fingerprint)


def paired_bootstrap(a, b, rng, n=20000):
    """Paired bootstrap on the mean of b - a, plus a win rate."""
    d = np.asarray(b) - np.asarray(a)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return {}
    idx = rng.integers(0, d.size, size=(n, d.size))
    boot = d[idx].mean(axis=1)
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "n": int(d.size),
        "mean_diff": float(d.mean()),
        "pct": float(100 * d.mean() / np.mean(a)) if np.mean(a) else float("nan"),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p": float(min(1.0, p)),
        "win_frac": float(np.mean(d < 0)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default="pd,pid")
    p.add_argument("--run_dir", default=None,
                   help="policy for the pd_res arm (and pid_res, unless "
                        "--run_dir_pid is given)")
    p.add_argument("--run_dir_pid", default=None,
                   help="policy for the pid_res arm. Supply this to compare a "
                        "PD-trained policy on a PD base against a PID-trained "
                        "policy on a PID base -- the like-for-like four-way. "
                        "Without it, pid_res reuses the PD-trained policy, "
                        "which measures TRANSFER, not PID+RL.")
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260829)
    p.add_argument("--stage", default=None,
                   help="curriculum stage name to evaluate on; "
                        "default is the flat all-disturbance distribution")
    p.add_argument("--curriculum_path", default="configs/curriculum.toml")
    p.add_argument("--force_dc_prob", type=float, default=None,
                   help="override the training config's constant-force "
                        "fraction. Set 0 when evaluating at one explicit "
                        "frequency, or policy sets trained with different "
                        "dc_prob get different evaluation distributions")
    p.add_argument("--force_freq", type=float, default=None,
                   help="external-force frequency in Hz. 0 (the default from "
                        "config) is a constant force and reproduces the "
                        "constant-disturbance results exactly")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; choose from {list(ARMS)}")
    if "pd_res" in arms and not args.run_dir:
        raise SystemExit("the pd_res arm requires --run_dir")
    if "pid_res" in arms and not (args.run_dir or args.run_dir_pid):
        raise SystemExit("the pid_res arm requires --run_dir or --run_dir_pid")

    # ---- config and policies ----
    def load_run(run_dir):
        """Load one trial's config and greedy actor parameters."""
        import json
        run_dir = Path(run_dir)
        saved = json.loads((run_dir / "config.json").read_text())
        cfg = Config.from_dict(saved)
        ckpt = run_dir / "checkpoints" / (
            args.checkpoint if args.checkpoint.endswith(".pt")
            else f"{args.checkpoint}.pt")
        if not ckpt.is_file():
            raise SystemExit(f"checkpoint not found: {ckpt}")
        probe = VecEnv(cfg.env, 1)
        agent = SAC(probe.obs_dim, probe.action_dim, cfg.sac)
        params = load_params(ckpt, agent.init(jax.random.PRNGKey(0)).actor_params)
        print(f"loaded {ckpt}  (trained on base_controller="
              f"{cfg.env.base_controller})")
        return cfg, agent, params

    cfg = Config()
    policies = {}          # arm -> (agent, params)
    if args.run_dir:
        cfg, agent_pd, params_pd = load_run(args.run_dir)
        policies["pd_res"] = (agent_pd, params_pd)
        policies["pid_res"] = (agent_pd, params_pd)
    if args.run_dir_pid:
        cfg_pid, agent_pid, params_pid = load_run(args.run_dir_pid)
        policies["pid_res"] = (agent_pid, params_pid)
        if not args.run_dir:
            cfg = cfg_pid

    transfer_arm = ("pid_res" in arms and args.run_dir and not args.run_dir_pid)
    if transfer_arm:
        print("\n  NOTE: pid_res is using the PD-trained policy. That is a "
              "TRANSFER result,\n        not PID+RL. Pass --run_dir_pid "
              "<pid-trained trial> for the real comparison.\n")

    # ---- disturbance distribution ----
    if args.stage:
        cur = load_curriculum(args.curriculum_path)
        stage = next((s for s in cur.stages if s.name == args.stage), None)
        if stage is None:
            raise SystemExit(f"stage {args.stage!r} not in {args.curriculum_path}")
        ranges, dist_name = stage.ranges(cfg.env), stage.name
    else:
        ranges = flat_ranges(cfg.env, force_freq=args.force_freq,
                             force_dc_prob=(0.0 if args.force_freq is not None
                                            and args.force_dc_prob is None
                                            else args.force_dc_prob))
        dist_name = ("flat_all_disturbances" if args.force_freq is None
                     else f"flat_all_disturbances_f{args.force_freq:g}Hz")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else (
        (Path(args.run_dir) / "evaluation" / f"compare_{stamp}") if args.run_dir
        else Path("evaluations") / f"compare_{stamp}"
    )
    (out / "plots").mkdir(parents=True, exist_ok=True)

    print(f"arms         : {', '.join(arms)}")
    print(f"episodes     : {args.episodes} (paired)")
    print(f"distribution : {dist_name}\n")

    # ---- run every arm on the SAME key ----
    key = jax.random.PRNGKey(args.seed)
    results, curves, reference_fp = {}, {}, None
    for arm in arms:
        arm_agent, arm_params = policies.get(arm, (None, None))
        per_ep, curve, fp = rollout_arm(
            cfg.env, CONTROLLER[arm],
            arm_agent if USES_RESIDUAL[arm] else None,
            arm_params if USES_RESIDUAL[arm] else None,
            key, args.episodes, ranges,
        )
        if reference_fp is None:
            reference_fp = fp
        elif not np.allclose(fp, reference_fp, atol=1e-6):
            raise SystemExit(
                f"arm {arm!r} saw a different disturbance draw; arms are not "
                "paired. This is a bug, not a config issue."
            )
        results[arm], curves[arm] = per_ep, curve
        print(f"  {LABEL[arm]:<16} pos RMSE mean {per_ep['pos_rmse'].mean():.4f} "
              f"median {np.median(per_ep['pos_rmse']):.4f}   "
              f"term {100*per_ep['terminated'].mean():.1f}%   "
              f"chatter f/M {per_ep['chatter_thrust'].mean():+.2f}/"
              f"{per_ep['chatter_moment'].mean():+.2f}")

    # ---- per-episode CSV ----
    rows = []
    for arm in arms:
        for i in range(args.episodes):
            rows.append({"arm": arm, "label": LABEL[arm], "episode": i,
                         **{k: float(v[i]) for k, v in results[arm].items()}})
    with (out / "per_episode.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- summary + paired contrasts ----
    rng = np.random.default_rng(args.seed)
    summary = [{"arm": a, "label": LABEL[a],
                **{f"{k}_mean": float(np.mean(v)) for k, v in results[a].items()}}
               for a in arms]
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    tests = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            for metric in ("pos_rmse", "pos_steady", "wrench_rms", "smoothness"):
                r = paired_bootstrap(results[a][metric], results[b][metric], rng)
                if r:
                    tests.append({"arm_a": a, "arm_b": b, "metric": metric, **r})
    if tests:
        with (out / "paired_tests.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tests[0]))
            w.writeheader()
            w.writerows(tests)

    # ---- plots ----
    plt.figure(figsize=(9, 4.5))
    t = np.arange(len(next(iter(curves.values())))) * cfg.env.dt
    for arm in arms:
        plt.plot(t, curves[arm], lw=1.4, label=LABEL[arm])
    plt.xlabel("time [s]")
    plt.ylabel("mean ‖x − x_d‖ [m]")
    plt.title(f"Tracking error, {args.episodes} paired episodes ({dist_name})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "plots" / "tracking_error.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    xs = np.arange(len(arms))
    means = [results[a]["pos_rmse"].mean() for a in arms]
    errs = [results[a]["pos_rmse"].std() / np.sqrt(args.episodes) for a in arms]
    plt.bar(xs, means, yerr=errs, capsize=4)
    plt.xticks(xs, [LABEL[a] for a in arms])
    plt.ylabel("Position RMSE [m]")
    plt.title("Mean position RMSE (± s.e.)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "plots" / "pos_rmse.png", dpi=150)
    plt.close()

    # ---- report ----
    lines = ["# Baseline controller comparison\n",
             f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
             f"- Episodes: {args.episodes}, paired on seed {args.seed}",
             f"- Distribution: `{dist_name}`",
             f"- PD-base policy: `{args.run_dir}`",
             f"- PID-base policy: `{args.run_dir_pid or args.run_dir or 'none'}`",
             ""]
    if transfer_arm:
        lines += ["> **Caveat.** `PID + residual` here uses a policy trained on "
                  "the PD baseline, so it measures TRANSFER, not PID+RL. Train "
                  "with `--base_controller pid` and pass `--run_dir_pid` for a "
                  "like-for-like number.\n"]
    lines += ["| Arm | Pos RMSE [m] | Steady [m] | Max [m] | Wrench RMS | Smoothness | Terminated |",
              "|---|---|---|---|---|---|---|"]
    for a in arms:
        r = results[a]
        lines.append(
            f"| {LABEL[a]} | {r['pos_rmse'].mean():.4f} | {r['pos_steady'].mean():.4f} "
            f"| {r['pos_max'].mean():.4f} | {r['wrench_rms'].mean():.3f} "
            f"| {r['smoothness'].mean():.4f} | {100*r['terminated'].mean():.0f}% |")
    if tests:
        lines += ["", "## Paired contrasts", "",
                  "Negative diff means arm B is better.", "",
                  "| A | B | Metric | Diff | 95% CI | p | B wins |",
                  "|---|---|---|---|---|---|---|"]
        for tst in tests:
            lines.append(
                f"| {LABEL[tst['arm_a']]} | {LABEL[tst['arm_b']]} | {tst['metric']} "
                f"| {tst['mean_diff']:+.4f} ({tst['pct']:+.1f}%) "
                f"| [{tst['ci_lo']:+.4f}, {tst['ci_hi']:+.4f}] | {tst['p']:.4f} "
                f"| {100*tst['win_frac']:.0f}% |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_manifest(out, {"arms": arms, "episodes": args.episodes,
                         "distribution": dist_name, "seed": args.seed})
    write_json(out / "config_used.json", cfg.to_dict())

    print(f"\nwrote {out}")
    print("  REPORT.md, summary.csv, per_episode.csv, paired_tests.csv, plots/")


if __name__ == "__main__":
    main()
