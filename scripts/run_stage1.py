"""Stage 1: does the residual improve BOTH baseline controllers?

Trains a residual policy on the PD base and another on the PID base, across
several seeds, then evaluates all four arms on one frozen set of disturbance
draws:

    pd        PD base, no residual              (reference)
    pid       PID base, no residual             (reference)
    pd_res    PD base  + PD-trained policy
    pid_res   PID base + PID-trained policy

Each seed's policies are trained under identical conditions apart from the base
controller, and every seed is evaluated against the SAME frozen evaluation seed
so numbers are comparable across seeds and across future experiments. Do not
change ``--eval_seed`` between experiments you intend to compare -- it defines
the disturbance bank.

Why per-seed policies matter: a single RL run is not evidence. Differences of
0.03 vs 0.04 m are well inside seed-to-seed spread, so every claim here is
reported as a mean over seeds with the spread shown.

Output:

    experiments/<name>/
        seed_00/
            pd/trial_001/         PD-base training run
            pid/trial_001/        PID-base training run
            comparison/           four-way evaluation for this seed
        seed_01/ ...
        AGGREGATE.md              across-seed summary
        aggregate.csv
        aggregate_pos_rmse.png

Examples
--------
# Full stage 1: 3 seeds, ~1 hour on a laptop GPU.
python -m scripts.run_stage1 --seeds 0,1,2 --total_steps 1000000

# Quick check that the pipeline works end to end (~5 min).
python -m scripts.run_stage1 --seeds 0 --total_steps 100000 --episodes 64

# Re-evaluate an existing experiment's policies after an evaluation-code
# change, without retraining and without overwriting the original results.
python -m scripts.run_stage1 --seeds 0,1,2 \
    --reuse_from experiments/stage1_20260830_104453 \
    --name stage1_rebaseline
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402

from qrjax.utils import write_json, write_manifest   # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ARMS = ("pd", "pid", "pd_res", "pid_res")
LABEL = {"pd": "PD", "pid": "PID", "pd_res": "PD + residual",
         "pid_res": "PID + residual"}


def run(cmd, cwd=REPO):
    """Run a subprocess, streaming output, and abort loudly on failure.

    Subprocesses rather than in-process calls: each training run gets a fresh
    interpreter, so JAX releases device memory between runs instead of
    accumulating compiled executables across a dozen configurations.
    """
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(f"command failed with code {result.returncode}")


def trial_dir(runs_root: Path, run_name: str) -> Path:
    """The most recent trial directory for a training run."""
    base = Path(runs_root) / run_name
    if not base.is_dir():
        raise SystemExit(
            f"no training runs at {base}.\n"
            f"With --skip_training this directory must already contain them. "
            f"To re-evaluate an EXISTING experiment while writing results "
            f"somewhere new, use --reuse_from <that experiment dir> instead of "
            f"--skip_training."
        )
    trials = sorted(p for p in base.iterdir() if p.is_dir()
                    and p.name.startswith("trial_"))
    if not trials:
        raise SystemExit(f"no trial_* directory under {base}")
    return trials[-1]


def passthrough_preview(args):
    """Record which optional training flags were forwarded, for the manifest."""
    out = []
    for flag in ("control_hz", "episode_seconds", "thrust_filter_beta",
                 "moment_filter_beta", "residual_authority", "rollout_len",
                 "lr", "batch_size", "learning_starts", "episode_steps",
                 "trajectory", "obs_mode"):
        v = getattr(args, flag, None)
        if v is not None:
            out += [f"--{flag}", str(v)]
    if getattr(args, "train_args", None):
        import shlex
        out += shlex.split(args.train_args)
    return out


def read_summary(path: Path):
    with path.open() as f:
        return {row["arm"]: row for row in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="0,1,2",
                   type=lambda s: [int(x) for x in s.split(",") if x.strip()])
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--utd", type=float, default=1.0)
    p.add_argument("--buffer_size", type=int, default=300_000)
    p.add_argument("--force_freq_min", type=float, default=0.0,
                   help="train with a time-varying external force drawn from "
                        "[min, max] Hz. Both 0 = constant force (Stage 1)")
    p.add_argument("--force_freq_max", type=float, default=0.0)
    p.add_argument("--force_dc_prob", type=float, default=0.0,
                   help="fraction of episodes with an exactly-constant force; "
                        "~0.3 trains constant and time-varying together")
    p.add_argument("--history", type=int, default=10,
                   help="observation history length in steps (dt=0.01)")
    p.add_argument("--hidden", default="256,256",
                   help="actor/critic hidden sizes, comma separated, e.g. "
                        "'512,512' for double width or '256,256,256,256' for "
                        "double depth")
    p.add_argument("--thrust_filter_beta", type=float, default=None,
                   help="low-pass on the thrust residual: "
                        "u = (1-beta) u_prev + beta a. 1.0 disables it")
    p.add_argument("--moment_filter_beta", type=float, default=None)
    p.add_argument("--residual_authority", type=float, default=None)
    p.add_argument("--rollout_len", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_starts", type=int, default=None)
    p.add_argument("--episode_steps", type=int, default=None)
    p.add_argument("--control_hz", type=float, default=None,
                   help="control and integration rate; 200 removes the "
                        "gain-margin failure")
    p.add_argument("--episode_seconds", type=float, default=None)
    p.add_argument("--trajectory", default=None)
    p.add_argument("--obs_mode", default=None)
    p.add_argument("--train_args", default=None,
                   help="escape hatch: any other scripts.train flags, as one "
                        "quoted string, e.g. --train_args \"--x64 "
                        "--checkpoint_every_iters 50\". Every training run "
                        "gets them verbatim, so nothing in train.py is "
                        "unreachable from here.")
    p.add_argument("--eval_episodes_train", type=int, default=128)
    p.add_argument("--episodes", type=int, default=256,
                   help="episodes in the final paired comparison")
    p.add_argument("--eval_seed", type=int, default=20260829,
                   help="FROZEN. Defines the evaluation disturbance bank; keep "
                        "it fixed across every experiment you want to compare")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="experiments")
    p.add_argument("--live_plot", action="store_true",
                   help="open the interactive training-curve window for each "
                        "training run. progress.png is written either way")
    p.add_argument("--skip_training", action="store_true",
                   help="reuse the training runs already inside --out/--name "
                        "and only redo the evaluations, writing them back in "
                        "place")
    p.add_argument("--reuse_from", default=None,
                   help="re-evaluate the training runs of an EXISTING "
                        "experiment directory, writing fresh evaluations into "
                        "--name. Use this to re-baseline after a change to the "
                        "evaluation code without retraining, and without "
                        "overwriting the original results.")
    args = p.parse_args()

    name = args.name or f"stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = Path(args.out) / name
    root.mkdir(parents=True, exist_ok=True)
    write_manifest(root, {"stage": 1, "seeds": args.seeds,
                          "train_passthrough": passthrough_preview(args),
                          "reuse_from": args.reuse_from,
                          "retrained": not (args.skip_training or args.reuse_from),
                          "force_freq_min": args.force_freq_min,
                          "force_freq_max": args.force_freq_max,
                          "force_dc_prob": args.force_dc_prob,
                          "history": args.history,
                          "hidden": args.hidden,
                          "total_steps": args.total_steps,
                          "eval_seed": args.eval_seed,
                          "episodes": args.episodes})

    py = sys.executable
    per_seed = {}

    # Forward the optional training flags that were actually supplied. Anything
    # left at None keeps train.py's own default, so this never silently
    # overrides it.
    passthrough = []
    for flag in ("control_hz", "episode_seconds", "thrust_filter_beta",
                 "moment_filter_beta", "residual_authority", "rollout_len",
                 "lr", "batch_size", "learning_starts", "episode_steps",
                 "trajectory", "obs_mode"):
        val = getattr(args, flag)
        if val is not None:
            passthrough += [f"--{flag}", str(val)]
    if args.train_args:
        import shlex
        passthrough += shlex.split(args.train_args)
    if passthrough:
        print(f"forwarding to scripts.train: {' '.join(passthrough)}")

    skip_training = args.skip_training or bool(args.reuse_from)
    source_root = Path(args.reuse_from) if args.reuse_from else root

    for seed in args.seeds:
        seed_dir = root / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        # Training runs may live in a different experiment than the one we are
        # writing evaluations into.
        src_seed_dir = source_root / f"seed_{seed:02d}"
        print("\n" + "=" * 78)
        print(f"SEED {seed}")
        print("=" * 78)

        run_dirs = {}
        for base in ("pd", "pid"):
            runs_root = (src_seed_dir if skip_training else seed_dir) / base
            if not skip_training:
                run([py, "-m", "scripts.train",
                     "--no_curriculum",
                     "--base_controller", base,
                     "--seed", seed,
                     "--total_steps", args.total_steps,
                     "--num_envs", args.num_envs,
                     "--utd", args.utd,
                     "--buffer_size", args.buffer_size,
                     "--eval_episodes", args.eval_episodes_train,
                     "--eval_every_iters", 10,
                     "--force_freq_min", args.force_freq_min,
                     "--force_freq_max", args.force_freq_max,
                     "--force_dc_prob", args.force_dc_prob,
                     "--history", args.history,
                     "--hidden", args.hidden,
                     "--runs_root", runs_root,
                     "--run_name", "train"]
                    + passthrough
                    + (["--live_plot"] if args.live_plot else []))
            run_dirs[base] = trial_dir(runs_root, "train")

        out_dir = seed_dir / "comparison"
        run([py, "-m", "scripts.evaluate",
             "--arms", ",".join(ARMS),
             "--run_dir", run_dirs["pd"],
             "--run_dir_pid", run_dirs["pid"],
             "--checkpoint", "best",
             "--episodes", args.episodes,
             "--seed", args.eval_seed,
             "--out", out_dir])
        per_seed[seed] = read_summary(out_dir / "summary.csv")

    # ------------------------------------------------------- aggregate
    metrics = ["pos_rmse", "pos_steady", "terminated", "wrench_rms",
               "smoothness", "chatter_thrust", "chatter_moment"]
    rows = []
    for arm in ARMS:
        entry = {"arm": arm, "label": LABEL[arm], "seeds": len(per_seed)}
        for m in metrics:
            vals = [float(per_seed[s][arm][f"{m}_mean"]) for s in per_seed
                    if arm in per_seed[s]]
            if vals:
                entry[f"{m}_mean"] = float(np.mean(vals))
                entry[f"{m}_std"] = float(np.std(vals))
                entry[f"{m}_per_seed"] = vals
        rows.append(entry)

    with (root / "aggregate.csv").open("w", newline="") as f:
        fields = [k for k in rows[0] if not k.endswith("_per_seed")]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    write_json(root / "aggregate.json", rows)

    # plot
    plt.figure(figsize=(8, 4.6))
    xs = np.arange(len(ARMS))
    means = [r.get("pos_rmse_mean", np.nan) for r in rows]
    stds = [r.get("pos_rmse_std", 0.0) for r in rows]
    plt.bar(xs, means, yerr=stds, capsize=5, color=["#888", "#888", "#3b7", "#37b"])
    for i, r in enumerate(rows):
        for v in r.get("pos_rmse_per_seed", []):
            plt.plot(i, v, "k.", ms=7)
    plt.xticks(xs, [LABEL[a] for a in ARMS])
    plt.ylabel("Position RMSE [m]")
    plt.title(f"Stage 1: does the residual improve both baselines?\n"
              f"{len(per_seed)} seeds, {args.episodes} paired episodes "
              f"(dots = individual seeds)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(root / "aggregate_pos_rmse.png", dpi=150)
    plt.close()

    # report
    L = [f"# Stage 1 — residual on PD vs PID\n",
         f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
         f"- Seeds: {args.seeds}",
         f"- Training: {args.total_steps:,} env steps, num_envs={args.num_envs}, "
         f"utd={args.utd}",
         f"- Evaluation: {args.episodes} paired episodes, frozen eval seed "
         f"{args.eval_seed}",
         "",
         "Each residual arm uses a policy trained on ITS OWN base controller, "
         "so this is a like-for-like comparison rather than a transfer test.",
         "",
         "| Arm | Pos RMSE [m] | Steady [m] | Terminated | Wrench RMS | Chatter f / M |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(
            f"| {r['label']} "
            f"| {r.get('pos_rmse_mean', float('nan')):.4f} "
            f"± {r.get('pos_rmse_std', 0):.4f} "
            f"| {r.get('pos_steady_mean', float('nan')):.4f} "
            f"| {100*r.get('terminated_mean', float('nan')):.1f}% "
            f"| {r.get('wrench_rms_mean', float('nan')):.3f} "
            f"| {r.get('chatter_thrust_mean', float('nan')):+.2f} / "
            f"{r.get('chatter_moment_mean', float('nan')):+.2f} |")

    def gain(a, b):
        ra = next(r for r in rows if r["arm"] == a)
        rb = next(r for r in rows if r["arm"] == b)
        va, vb = ra.get("pos_rmse_mean"), rb.get("pos_rmse_mean")
        if not va or not vb:
            return "n/a"
        return f"{100*(vb-va)/va:+.1f}%"

    L += ["", "## The question this stage asks", "",
          f"- Residual on the PD base: **{gain('pd', 'pd_res')}** position RMSE",
          f"- Residual on the PID base: **{gain('pid', 'pid_res')}** position RMSE",
          f"- PID vs PD, no residual: **{gain('pd', 'pid')}**",
          f"- PID+residual vs PD+residual: **{gain('pd_res', 'pid_res')}**",
          "",
          "If the residual improves PD but not PID, the residual is largely "
          "rediscovering integral action. If it improves both by similar "
          "margins, it is doing something integral action cannot.",
          "",
          "## Per-seed values", ""]
    for r in rows:
        vals = ", ".join(f"{v:.4f}" for v in r.get("pos_rmse_per_seed", []))
        L.append(f"- {r['label']}: {vals}")
    L += ["", "## Files", "",
          "- `seed_XX/comparison/` — full paired report and plots per seed",
          "- `seed_XX/{pd,pid}/train/trial_001/` — training runs and checkpoints",
          "- `aggregate.csv`, `aggregate.json`, `aggregate_pos_rmse.png`"]
    (root / "AGGREGATE.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print("STAGE 1 COMPLETE")
    print("=" * 78)
    for r in rows:
        print(f"  {r['label']:<18} {r.get('pos_rmse_mean', float('nan')):.4f} "
              f"± {r.get('pos_rmse_std', 0):.4f} m    "
              f"term {100*r.get('terminated_mean', float('nan')):.1f}%")
    print()
    print(f"  residual on PD  : {gain('pd', 'pd_res')}")
    print(f"  residual on PID : {gain('pid', 'pid_res')}")
    print()
    print(f"wrote {root}/AGGREGATE.md")


if __name__ == "__main__":
    main()
