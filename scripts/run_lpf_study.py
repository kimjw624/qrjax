"""LPF ablation: none / thrust only / moment only / both.

Trains a residual policy under each of the four filter configurations and
evaluates them on one frozen disturbance bank. The filter is applied
consistently everywhere it appears: the plant receives the filtered command,
and the observation history stores the filtered command, because the filtered
signal is what physically acted on the vehicle and is therefore what the policy
should see it did.

Run at 200 Hz by default. At 100 Hz the attitude loop tolerates a parameter
mismatch of only g = kf*arm/k < 1.76, while the disturbance box reaches 2.41,
so a few percent of episodes fail for reasons unrelated to the filter and
contaminate the comparison. At 200 Hz the limit is 3.49 and the whole box is
inside it, which isolates the LPF effect.

Note on what is NOT filtered: the reward's effort and smoothness penalties use
the RAW policy request, not the filtered command. Penalising the filtered
signal would let the policy chatter for free and rely on the filter to hide it,
removing any learning pressure toward smooth commands. That choice is
deliberate; ``--penalize_filtered`` flips it if you want to test the
alternative.

Output:

    experiments/<name>/
        none/           seed_XX/{pd,pid}/train/trial_001
        thrust/         ...
        moment/         ...
        both/           ...
        evaluation/     per-config comparison against the shared bank
        AGGREGATE.md    the four configurations side by side
        aggregate.csv
        lpf_comparison.png

Example
-------
python -m scripts.run_lpf_study --seeds 0 --total_steps 2000000
"""

import argparse
import csv
import json
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

# (name, thrust_beta, moment_beta)
CONFIGS = [
    ("none", 1.0, 1.0),
    ("thrust", 0.2, 1.0),
    ("moment", 1.0, 0.2),
    ("both", 0.2, 0.2),
]
LABEL = {"none": "no LPF", "thrust": "LPF on f",
         "moment": "LPF on M", "both": "LPF on f and M"}
ARMS = ("pd", "pid", "pd_res", "pid_res")


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode})")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="0",
                   type=lambda s: [int(x) for x in s.split(",") if x.strip()])
    p.add_argument("--total_steps", type=int, default=2_000_000,
                   help="env steps per run. At 200 Hz an episode is 2000 steps "
                        "rather than 1000, so this is doubled from the 100 Hz "
                        "default to keep the same number of episodes")
    p.add_argument("--control_hz", type=float, default=200.0)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--utd", type=float, default=1.0)
    p.add_argument("--buffer_size", type=int, default=300_000)
    p.add_argument("--history", type=int, default=10,
                   help="NOTE: at 200 Hz, 10 frames span 0.05 s rather than "
                        "0.1 s. Use 20 to hold the time window fixed if you "
                        "intend to compare against 100 Hz results")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--configs", default="none,thrust,moment,both",
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="experiments")
    p.add_argument("--skip_training", action="store_true")
    p.add_argument("--live_plot", action="store_true")
    args = p.parse_args()

    name = args.name or f"lpf_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = Path(args.out) / name
    root.mkdir(parents=True, exist_ok=True)
    configs = [c for c in CONFIGS if c[0] in args.configs]

    write_manifest(root, {
        "study": "lpf_ablation", "configs": [c[0] for c in configs],
        "seeds": args.seeds, "control_hz": args.control_hz,
        "total_steps": args.total_steps, "history": args.history,
        "episodes": args.episodes, "eval_seed": args.eval_seed,
        "note": "filter applied to both plant input and observation history; "
                "reward penalty uses the raw request",
    })

    py = sys.executable
    for cfg_name, tb, mb in configs:
        print("\n" + "=" * 78)
        print(f"CONFIG {cfg_name}   thrust beta {tb}   moment beta {mb}")
        print("=" * 78)
        if not args.skip_training:
            run([py, "-m", "scripts.run_stage1",
                 "--seeds", ",".join(str(s) for s in args.seeds),
                 "--total_steps", args.total_steps,
                 "--control_hz", args.control_hz,
                 "--num_envs", args.num_envs,
                 "--utd", args.utd,
                 "--buffer_size", args.buffer_size,
                 "--history", args.history,
                 "--thrust_filter_beta", tb,
                 "--moment_filter_beta", mb,
                 "--episodes", args.episodes,
                 "--eval_seed", args.eval_seed,
                 "--name", cfg_name,
                 "--out", root]
                + (["--live_plot"] if args.live_plot else []))

    # ---------------- aggregate across configurations ----------------
    rows = []
    for cfg_name, tb, mb in configs:
        agg = root / cfg_name / "aggregate.csv"
        if not agg.is_file():
            print(f"  (missing {agg}, skipping)")
            continue
        with agg.open() as f:
            for r in csv.DictReader(f):
                rows.append({"config": cfg_name, "label": LABEL[cfg_name],
                             "thrust_beta": tb, "moment_beta": mb,
                             "arm": r["arm"],
                             "pos_rmse": float(r.get("pos_rmse_mean", "nan")),
                             "pos_rmse_std": float(r.get("pos_rmse_std", "nan")),
                             "terminated": float(r.get("terminated_mean", "nan")),
                             "smoothness": float(r.get("smoothness_mean", "nan")),
                             "chatter_thrust": float(r.get("chatter_thrust_mean", "nan")),
                             "chatter_moment": float(r.get("chatter_moment_mean", "nan"))})
    if not rows:
        raise SystemExit("no per-config aggregates found; did training run?")

    with (root / "aggregate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(root / "aggregate.json", rows)

    def get(cfg_name, arm, field):
        for r in rows:
            if r["config"] == cfg_name and r["arm"] == arm:
                return r[field]
        return float("nan")

    names = [c[0] for c in configs]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    x = np.arange(len(names))
    for ax, (field, title) in zip(axes, (
            ("pos_rmse", "Position RMSE [m]"),
            ("smoothness", "Command smoothness RMS"),
            ("terminated", "Termination rate"))):
        for i, arm in enumerate(("pd_res", "pid_res")):
            v = [get(n, arm, field) for n in names]
            ax.bar(x + i * 0.35, v, 0.35, label=arm)
        base = [get(n, "pid", field) for n in names]
        ax.plot(x + 0.175, base, "k--o", lw=1.2, ms=4, label="PID baseline")
        ax.set_xticks(x + 0.175)
        ax.set_xticklabels([LABEL[n] for n in names], fontsize=8, rotation=12)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"LPF ablation at {args.control_hz:g} Hz "
                 f"({len(args.seeds)} seed(s), {args.episodes} paired episodes)")
    fig.tight_layout()
    fig.savefig(root / "lpf_comparison.png", dpi=150)
    plt.close(fig)

    L = [f"# LPF ablation at {args.control_hz:g} Hz\n",
         f"Generated {datetime.now().isoformat(timespec='seconds')}",
         "",
         f"- Seeds {args.seeds}, {args.total_steps:,} env steps per run",
         f"- {args.episodes} paired episodes, frozen eval seed {args.eval_seed}",
         f"- History {args.history} frames = "
         f"{args.history/args.control_hz*1000:.0f} ms",
         "",
         f"At {args.control_hz:g} Hz the attitude loop tolerates "
         f"g = kf*arm/k up to "
         f"{2*0.022/((1/args.control_hz)*(2.54-(1/args.control_hz)*8.81/2)):.2f}, "
         "and the disturbance box reaches 2.41, so the gain-margin failure "
         "seen at 100 Hz is absent and this comparison isolates the filter.",
         "",
         "## Results", "",
         "| config | thrust beta | moment beta | arm | pos RMSE [m] | "
         "smoothness | chatter f | chatter M | term % |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | {r['thrust_beta']:g} | {r['moment_beta']:g} "
                 f"| {r['arm']} | {r['pos_rmse']:.4f} | {r['smoothness']:.4f} "
                 f"| {r['chatter_thrust']:+.2f} | {r['chatter_moment']:+.2f} "
                 f"| {100*r['terminated']:.1f} |")
    L += ["",
          "## Reading it", "",
          "- `chatter f` and `chatter M` are lag-1 autocorrelations of the "
          "residual command. Values near +1 are smooth; near or below 0 mean "
          "the command reverses sign step to step.",
          "- The filter is expected to raise the chatter figure on the channel "
          "it acts on and leave the other roughly unchanged. If filtering the "
          "moment channel also improves the thrust channel, the two are "
          "coupled through the plant rather than independent.",
          "- Compare each residual arm against its own baseline row, not "
          "across configurations only.",
          "",
          "## Caveats", "",
          f"- History is {args.history} frames = "
          f"{args.history/args.control_hz*1000:.0f} ms at this rate. The "
          "100 Hz experiments used 10 frames = 100 ms, so a direct comparison "
          "with those numbers confounds rate with observation window. Use "
          "`--history 20` at 200 Hz to hold the window fixed.",
          "- The reward penalty uses the RAW request, so a configuration with "
          "a filter is penalised for chatter the filter then removes. That is "
          "intentional: penalising the filtered signal would remove the "
          "pressure to be smooth in the first place.",
          "",
          "## Files", "",
          "- `aggregate.csv`, `aggregate.json`",
          "- `lpf_comparison.png`",
          "- `<config>/AGGREGATE.md` and `<config>/seed_XX/comparison/` per configuration"]
    (root / "AGGREGATE.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print("LPF STUDY COMPLETE")
    print("=" * 78)
    print(f"{'config':<18}{'arm':<10}{'pos RMSE':>10}{'smooth':>10}"
          f"{'chatter f':>11}{'term %':>9}")
    for r in rows:
        if r["arm"] in ("pd_res", "pid_res"):
            print(f"{r['label']:<18}{r['arm']:<10}{r['pos_rmse']:>10.4f}"
                  f"{r['smoothness']:>10.4f}{r['chatter_thrust']:>11.2f}"
                  f"{100*r['terminated']:>8.1f}%")
    print(f"\nwrote {root}/AGGREGATE.md")


if __name__ == "__main__":
    main()
