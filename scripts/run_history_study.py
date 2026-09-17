"""History ablation: how much memory does the residual need?

Trains a residual policy at several observation-history lengths and evaluates
them on one frozen disturbance bank.

`--history 1` is the interesting end of the range: the policy sees only the
CURRENT twin discrepancy, 12 numbers, with no past states and no past actions.
A memoryless policy cannot accumulate evidence over time, so it cannot
implement anything integral-like. That matters because the two baselines differ
precisely in whether they integrate:

  PD base   the residual has to supply the integral action itself, so it should
            depend heavily on history
  PID base  the baseline already integrates, so the residual should need less

If that split appears, it is direct evidence about what the residual is
actually doing, rather than an inference from tracking numbers alone.

Note that history is measured in FRAMES, not seconds. At 200 Hz a frame is
5 ms, so 10 frames span 50 ms against 100 ms at 100 Hz. The report prints both.

Observation dimension grows as `12*H + 4*(H-1)`, so long histories cost replay
memory: at H=40 the buffer is 5.7x larger than at H=1 for the same capacity.
Policies with different H are NOT interchangeable — each is only loadable by a
config with the same H, so each history value needs its own evaluation.

Output:

    experiments/<name>/
        h001/  h002/  h005/  h010/  h020/     seed_XX/{pd,pid}/train/trial_001
        aggregate.csv
        history_comparison.png
        AGGREGATE.md

Example
-------
python -m scripts.run_history_study --seeds 0 --total_steps 2000000
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
ARM_LABEL = {"pd": "PD", "pid": "PID",
             "pd_res": "PD + residual", "pid_res": "PID + residual"}


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode})")


def obs_dim(h):
    return 12 * h + 4 * (h - 1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--histories", default="1,2,5,10,20",
                   type=lambda s: [int(x) for x in s.split(",") if x.strip()],
                   help="history lengths in FRAMES. 1 = current state only")
    p.add_argument("--seeds", default="0",
                   type=lambda s: [int(x) for x in s.split(",") if x.strip()])
    p.add_argument("--total_steps", type=int, default=2_000_000)
    p.add_argument("--control_hz", type=float, default=200.0)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--utd", type=float, default=1.0)
    p.add_argument("--buffer_size", type=int, default=300_000,
                   help="held constant across history lengths, so long "
                        "histories use proportionally more device memory")
    p.add_argument("--thrust_filter_beta", type=float, default=0.2)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="experiments")
    p.add_argument("--skip_training", action="store_true")
    p.add_argument("--live_plot", action="store_true")
    args = p.parse_args()

    name = args.name or f"history_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = Path(args.out) / name
    root.mkdir(parents=True, exist_ok=True)

    print(f"histories : {args.histories} frames "
          f"= {[round(h/args.control_hz*1000) for h in args.histories]} ms "
          f"at {args.control_hz:g} Hz")
    print(f"obs dims  : {[obs_dim(h) for h in args.histories]}")
    print(f"runs      : {len(args.histories)} x {len(args.seeds)} seeds x 2 bases "
          f"= {len(args.histories)*len(args.seeds)*2}\n")

    write_manifest(root, {
        "study": "history_ablation", "histories": args.histories,
        "obs_dims": [obs_dim(h) for h in args.histories],
        "seeds": args.seeds, "control_hz": args.control_hz,
        "total_steps": args.total_steps, "episodes": args.episodes,
        "eval_seed": args.eval_seed,
        "thrust_filter_beta": args.thrust_filter_beta,
        "note": "history=1 is the current twin discrepancy only, no memory",
    })

    py = sys.executable
    for h in args.histories:
        tag = f"h{h:03d}"
        print("\n" + "=" * 78)
        print(f"HISTORY {h} frames  ({h/args.control_hz*1000:.0f} ms, "
              f"obs_dim {obs_dim(h)})")
        print("=" * 78)
        if not args.skip_training:
            run([py, "-m", "scripts.run_stage1",
                 "--seeds", ",".join(str(s) for s in args.seeds),
                 "--total_steps", args.total_steps,
                 "--control_hz", args.control_hz,
                 "--num_envs", args.num_envs,
                 "--utd", args.utd,
                 "--buffer_size", args.buffer_size,
                 "--history", h,
                 "--thrust_filter_beta", args.thrust_filter_beta,
                 "--episodes", args.episodes,
                 "--eval_seed", args.eval_seed,
                 "--name", tag,
                 "--out", root]
                + (["--live_plot"] if args.live_plot else []))

    # ---------------- aggregate ----------------
    rows = []
    for h in args.histories:
        agg = root / f"h{h:03d}" / "aggregate.csv"
        if not agg.is_file():
            print(f"  (missing {agg}, skipping)")
            continue
        with agg.open() as f:
            for r in csv.DictReader(f):
                rows.append({
                    "history": h,
                    "span_ms": round(h / args.control_hz * 1000),
                    "obs_dim": obs_dim(h),
                    "arm": r["arm"], "label": ARM_LABEL[r["arm"]],
                    "pos_rmse": float(r.get("pos_rmse_mean", "nan")),
                    "pos_rmse_std": float(r.get("pos_rmse_std", "nan")),
                    "terminated": float(r.get("terminated_mean", "nan")),
                    "smoothness": float(r.get("smoothness_mean", "nan")),
                })
    if not rows:
        raise SystemExit("no per-history aggregates found; did training run?")

    with (root / "aggregate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(root / "aggregate.json", rows)

    def get(h, arm, field="pos_rmse"):
        for r in rows:
            if r["history"] == h and r["arm"] == arm:
                return r[field]
        return float("nan")

    hs = [h for h in args.histories if not np.isnan(get(h, "pd_res"))]

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    for arm, colour in (("pd_res", "#2ca02c"), ("pid_res", "#1f77b4")):
        ax.plot(hs, [get(h, arm) for h in hs], marker="o", lw=1.8,
                color=colour, label=ARM_LABEL[arm])
    for arm, colour in (("pd", "#999999"), ("pid", "#444444")):
        ax.axhline(np.nanmean([get(h, arm) for h in hs]), ls="--", lw=1.2,
                   color=colour, label=f"{ARM_LABEL[arm]} baseline")
    ax.set_xscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs])
    ax.set_xlabel("history [frames]")
    ax.set_ylabel("Position RMSE [m]")
    ax.set_title("Tracking vs observation history")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for arm, base, colour in (("pd_res", "pd", "#2ca02c"),
                              ("pid_res", "pid", "#1f77b4")):
        gain = [100 * (get(h, arm) - get(h, base)) / get(h, base) for h in hs]
        ax.plot(hs, gain, marker="o", lw=1.8, color=colour,
                label=f"{ARM_LABEL[arm]} vs {ARM_LABEL[base]}")
    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs])
    ax.set_xlabel("history [frames]")
    ax.set_ylabel("change vs own baseline [%]")
    ax.set_title("How much the residual buys\n"
                 "below zero = residual improves its baseline")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    for arm, colour in (("pd_res", "#2ca02c"), ("pid_res", "#1f77b4")):
        ax.plot(hs, [get(h, arm, "smoothness") for h in hs], marker="o",
                lw=1.8, color=colour, label=ARM_LABEL[arm])
    ax.set_xscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs])
    ax.set_xlabel("history [frames]")
    ax.set_ylabel("command smoothness RMS")
    ax.set_title("Command roughness vs history")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(f"History ablation at {args.control_hz:g} Hz "
                 f"(1 frame = {1000/args.control_hz:.1f} ms), "
                 f"{len(args.seeds)} seed(s), {args.episodes} paired episodes")
    fig.tight_layout()
    fig.savefig(root / "history_comparison.png", dpi=150)
    plt.close(fig)

    # ---------------- report ----------------
    L = [f"# History ablation at {args.control_hz:g} Hz\n",
         f"Generated {datetime.now().isoformat(timespec='seconds')}",
         "",
         f"- Seeds {args.seeds}, {args.total_steps:,} env steps per run",
         f"- {args.episodes} paired episodes, frozen eval seed {args.eval_seed}",
         f"- Thrust filter beta {args.thrust_filter_beta}",
         f"- 1 frame = {1000/args.control_hz:.1f} ms",
         "",
         "`history = 1` gives the policy the current twin discrepancy only: "
         "12 numbers, no past states, no past actions. A memoryless policy "
         "cannot accumulate evidence over time and so cannot implement "
         "integral-like behaviour.",
         "",
         "## Results", "",
         "| History | Span | obs_dim | Arm | RMSE [m] | vs baseline | "
         "Smoothness | Term % |",
         "|---|---|---|---|---|---|---|---|"]
    for h in hs:
        for arm in ARMS:
            base = "pd" if arm in ("pd", "pd_res") else "pid"
            rel = ("---" if arm in ("pd", "pid") else
                   f"{100*(get(h, arm)-get(h, base))/get(h, base):+.1f}%")
            L.append(f"| {h} | {round(h/args.control_hz*1000)} ms "
                     f"| {obs_dim(h)} | {ARM_LABEL[arm]} "
                     f"| {get(h, arm):.4f} | {rel} "
                     f"| {get(h, arm, 'smoothness'):.4f} "
                     f"| {100*get(h, arm, 'terminated'):.1f} |")

    L += ["", "## What to look for", "",
          "**The split between bases is the informative part.** With a PD base "
          "the residual must supply integral action itself, which requires "
          "accumulating evidence and therefore memory. With a PID base the "
          "controller already integrates, so the residual should be far less "
          "sensitive to history length. If `pd_res` degrades sharply at "
          "history = 1 while `pid_res` barely moves, that is direct evidence "
          "about the function the residual is performing, rather than an "
          "inference from tracking numbers alone.",
          "",
          "**Watch for saturation.** If RMSE stops improving beyond some "
          "history length, that length is the useful memory horizon for this "
          "task, and anything longer is paying observation dimension for "
          "nothing.",
          "",
          "**Baselines should be flat.** The `pd` and `pid` rows contain no "
          "policy, so their numbers must be identical across history values. "
          "Any variation there indicates an evaluation bug, not an effect.",
          "",
          "## Caveats", "",
          "- Policies with different history lengths have different "
          "observation dimensions and are not interchangeable. Each history "
          "value is evaluated with its own policy.",
          "- Replay memory scales with obs_dim. At the same `--buffer_size`, "
          f"history {max(hs)} uses {obs_dim(max(hs))/obs_dim(min(hs)):.1f}x "
          f"the memory of history {min(hs)}. Buffer size is held constant here, "
          "so the longest history has proportionally more device memory "
          "pressure but the same number of stored transitions.",
          "- History is in frames. Comparing against 100 Hz results confounds "
          "control rate with observation window.",
          "",
          "## Files", "",
          "- `aggregate.csv`, `aggregate.json`",
          "- `history_comparison.png`",
          "- `h<NNN>/AGGREGATE.md` and `h<NNN>/seed_XX/comparison/` per history"]
    (root / "AGGREGATE.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print("HISTORY STUDY COMPLETE")
    print("=" * 78)
    print(f"{'hist':>6}{'span':>8}{'obs':>6}"
          f"{'PD+res':>10}{'vs PD':>9}{'PID+res':>10}{'vs PID':>9}")
    for h in hs:
        gpd = 100 * (get(h, "pd_res") - get(h, "pd")) / get(h, "pd")
        gpid = 100 * (get(h, "pid_res") - get(h, "pid")) / get(h, "pid")
        print(f"{h:>6}{round(h/args.control_hz*1000):>6} ms{obs_dim(h):>6}"
              f"{get(h, 'pd_res'):>10.4f}{gpd:>8.1f}%"
              f"{get(h, 'pid_res'):>10.4f}{gpid:>8.1f}%")
    print(f"\nwrote {root}/AGGREGATE.md")


if __name__ == "__main__":
    main()
