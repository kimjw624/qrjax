"""Stage 2a: do constant-trained residual policies survive time-varying disturbances?

Evaluation only. It reuses the policies already trained in Stage 1 on CONSTANT
per-episode disturbances, and tests them against an oscillating external force

    F(t) = A cos(2 pi f t + phi)

across a sweep of f. Nothing is retrained, so this measures out-of-distribution
generalization before any fix exists for it -- which is the only point at which
that measurement is honest.

The question is not only "does it get worse". It is whether the residual still
HELPS its baseline at each frequency, or starts actively hurting it. A policy
that learned to cancel a constant offset may fight an oscillating one and end
up worse than the bare controller. Each frequency therefore reports the
residual arms against their own baselines, not just against f = 0.

Two properties make the sweep interpretable:

*f = 0 reproduces Stage 1 exactly.* At zero frequency the phase is forced to
zero and the force is identically constant, so the first column of the sweep
must match the constant-disturbance numbers. If it does not, the disturbance
implementation is wrong and nothing else in the sweep can be trusted.

*Force RMS is held constant across frequency.* A cos(wt+phi) has time-RMS
A/sqrt(2) while a constant A has RMS A, so the amplitude is scaled by sqrt(2)
for f > 0. Without that, disturbance energy would drop the instant frequency
left zero and every arm would appear to improve for reasons unrelated to
frequency.

Example
-------
python -m scripts.run_stage2_sweep \
    --stage1_dir experiments/stage1_20260830_104453 \
    --freqs 0,0.1,0.25,0.5,1.0,2.0,4.0 --episodes 256
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import jax                                           # noqa: E402
import numpy as np                                   # noqa: E402

from qrjax.envs import Config, VecEnv                # noqa: E402
from qrjax.rl import SAC                             # noqa: E402
from qrjax.rl.curriculum import flat_ranges          # noqa: E402
from qrjax.utils import load_params, write_json, write_manifest   # noqa: E402

from scripts.evaluate import rollout_arm, paired_bootstrap        # noqa: E402

ARMS = ("pd", "pid", "pd_res", "pid_res")
LABEL = {"pd": "PD", "pid": "PID", "pd_res": "PD + residual",
         "pid_res": "PID + residual"}
CONTROLLER = {"pd": "pd", "pid": "pid", "pd_res": "pd", "pid_res": "pid"}
USES_RESIDUAL = {"pd": False, "pid": False, "pd_res": True, "pid_res": True}
COLOR = {"pd": "#999999", "pid": "#444444",
         "pd_res": "#2ca02c", "pid_res": "#1f77b4"}


def load_policy(run_dir: Path, checkpoint="best"):
    run_dir = Path(run_dir)
    cfg = Config.from_dict(json.loads((run_dir / "config.json").read_text()))
    ckpt = run_dir / "checkpoints" / (
        checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt")
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    probe = VecEnv(cfg.env, 1)
    agent = SAC(probe.obs_dim, probe.action_dim, cfg.sac)
    params = load_params(ckpt, agent.init(jax.random.PRNGKey(0)).actor_params)
    return cfg, agent, params


def find_seed_runs(stage1_dir: Path):
    """Locate the {pd, pid} training trials for each seed of a Stage 1 run."""
    out = {}
    for seed_dir in sorted(Path(stage1_dir).glob("seed_*")):
        entry = {}
        for base in ("pd", "pid"):
            trials = sorted((seed_dir / base / "train").glob("trial_*"))
            if trials:
                entry[base] = trials[-1]
        if len(entry) == 2:
            out[seed_dir.name] = entry
    if not out:
        raise SystemExit(
            f"no seed_*/{{pd,pid}}/train/trial_* under {stage1_dir}. "
            "Point --stage1_dir at a completed run_stage1 output directory."
        )
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1_dir", required=True,
                   help="policy set A: a completed run_stage1 output directory")
    p.add_argument("--compare_dir", nargs="*", default=[],
                   help="additional policy sets, evaluated on the SAME "
                        "episodes. Pass several to overlay e.g. "
                        "constant-trained, varying-trained and mixed-trained")
    p.add_argument("--label_a", default="const-trained")
    p.add_argument("--label_b", nargs="*", default=None,
                   help="labels for --compare_dir, in order. Defaults to each "
                        "directory's own name")
    p.add_argument("--freqs", default="0,0.1,0.25,0.5,1.0,2.0,4.0",
                   type=lambda s: [float(x) for x in s.split(",") if x.strip()])
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--eval_seed", type=int, default=20260829,
                   help="FROZEN. Same bank as Stage 1, so the f=0 column is "
                        "directly comparable to the constant-disturbance run")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    stage1 = Path(args.stage1_dir)
    sets = {args.label_a: find_seed_runs(stage1)}
    extra = list(args.compare_dir or [])
    labels = list(args.label_b or [])
    if len(labels) < len(extra):
        labels += [Path(d).name for d in extra[len(labels):]]
    for label, d in zip(labels, extra):
        if label in sets:
            raise SystemExit(f"duplicate policy-set label {label!r}")
        sets[label] = find_seed_runs(Path(d))
    out = Path(args.out) if args.out else (
        stage1.parent / f"stage2_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    (out / "plots").mkdir(parents=True, exist_ok=True)

    print(f"policy sets    : " + ", ".join(
        f"{n} ({len(v)} seeds)" for n, v in sets.items()))
    print(f"source         : {stage1}")
    for d in extra:
        print(f"                 {d}")
    print(f"frequencies    : {args.freqs} Hz")
    print(f"episodes       : {args.episodes} paired, eval seed {args.eval_seed}")
    print("\nNo retraining: these policies saw CONSTANT disturbances only.\n")

    rows = []
    key = jax.random.PRNGKey(args.eval_seed)
    baselines_done = set()

    for set_name, seeds in sets.items():
        for seed_name, run_dirs in seeds.items():
            cfg, agent_pd, params_pd = load_policy(run_dirs["pd"], args.checkpoint)
            _, agent_pid, params_pid = load_policy(run_dirs["pid"], args.checkpoint)
            policy = {"pd_res": (agent_pd, params_pd),
                      "pid_res": (agent_pid, params_pid)}

            for freq in args.freqs:
                # force_dc_prob=0.0 is essential, not cosmetic. It lives in
                # the TRAINING config, so a mixed-trained policy set would
                # otherwise be evaluated with ~30% of episodes at f=0 at EVERY
                # requested frequency, while const/varying-trained sets get the
                # pure frequency. That silently compares policy sets on
                # different disturbance distributions.
                ranges = flat_ranges(cfg.env, force_freq=freq,
                                     force_dc_prob=0.0)
                reference_fp = None
                line = f"  [{set_name}] {seed_name}  f={freq:>5.2f} Hz  "
                for arm in ARMS:
                    # The bare baselines have no policy, so they are identical
                    # across policy sets. Run them once per frequency.
                    if not USES_RESIDUAL[arm]:
                        if (freq, arm) in baselines_done:
                            continue
                        baselines_done.add((freq, arm))
                    a, pr = policy.get(arm, (None, None))
                    per_ep, _, fp = rollout_arm(
                        cfg.env, CONTROLLER[arm],
                        a if USES_RESIDUAL[arm] else None,
                        pr if USES_RESIDUAL[arm] else None,
                        key, args.episodes, ranges)
                    if reference_fp is None:
                        reference_fp = fp
                    elif not np.allclose(fp, reference_fp, atol=1e-6):
                        raise SystemExit(
                            f"{seed_name} f={freq}: arm {arm} saw a different "
                            "disturbance draw; arms are not paired.")
                    rows.append({
                        "policy_set": ("baseline" if not USES_RESIDUAL[arm]
                                       else set_name),
                        "seed": seed_name, "freq_hz": freq, "arm": arm,
                        "label": LABEL[arm],
                        "pos_rmse": float(np.mean(per_ep["pos_rmse"])),
                        "pos_rmse_median": float(np.median(per_ep["pos_rmse"])),
                        "pos_steady": float(np.mean(per_ep["pos_steady"])),
                        "terminated": float(np.mean(per_ep["terminated"])),
                        "saturation_frac": float(np.mean(per_ep["saturation_frac"])),
                        "chatter_thrust": float(np.mean(per_ep["chatter_thrust"])),
                    })
                    line += f"{LABEL[arm]} {np.mean(per_ep['pos_rmse']):.4f}  "
                print(line, flush=True)

    with (out / "sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------- aggregate
    set_names = list(sets)

    def agg(arm, freq, field="pos_rmse", policy_set=None):
        """Aggregate one cell.

        `policy_set` MUST be specified for residual arms whenever more than one
        set is present: pooling them averages the two things the sweep exists
        to compare. Baselines carry policy_set == "baseline" and are shared.
        """
        if USES_RESIDUAL[arm] and policy_set is None and len(set_names) > 1:
            raise ValueError(
                f"aggregating arm {arm!r} across {len(set_names)} policy sets "
                "would average them together; pass policy_set explicitly")
        v = [r[field] for r in rows
             if r["arm"] == arm and r["freq_hz"] == freq
             and (not USES_RESIDUAL[arm] or policy_set is None
                  or r["policy_set"] == policy_set)]
        return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)

    def cell(arm, f, field="pos_rmse", ps=None):
        return agg(arm, f, field,
                   ps if USES_RESIDUAL[arm] else None)[0]

    # ---- main figure: error vs frequency ----
    plt.figure(figsize=(9, 5))
    for arm in ("pd", "pid"):
        m = [agg(arm, f)[0] for f in args.freqs]
        plt.plot(args.freqs, m, marker="s", ls="--", lw=1.4,
                 label=LABEL[arm], color=COLOR[arm])
    for si, sname in enumerate(set_names):
        marker = ["o", "^", "s", "D"][si % 4]
        style = ["-", ":", "-.", "--"][si % 4]
        for arm in ("pd_res", "pid_res"):
            m = [agg(arm, f, policy_set=sname)[0] for f in args.freqs]
            e = [agg(arm, f, policy_set=sname)[1] for f in args.freqs]
            plt.errorbar(args.freqs, m, yerr=e, marker=marker, ls=style,
                         capsize=3, lw=1.8,
                         label=f"{LABEL[arm]} ({sname})", color=COLOR[arm],
                         alpha=1.0 - 0.25 * si)
    plt.xlabel("external force frequency [Hz]   (RMS held constant)")
    plt.ylabel("Position RMSE [m]")
    plt.title("Constant-trained residual policies under a time-varying force\n"
              "no retraining — f=0 is the training distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "plots" / "rmse_vs_frequency.png", dpi=150)
    plt.close()

    # ---- the question that matters: does the residual still HELP? ----
    plt.figure(figsize=(9, 5))
    for base, res in (("pd", "pd_res"), ("pid", "pid_res")):
        for si, sn in enumerate(set_names):
            gain = []
            for f in args.freqs:
                b = cell(base, f)
                r = cell(res, f, ps=sn)
                gain.append(100.0 * (r - b) / b if b else np.nan)
            plt.plot(args.freqs, gain, marker=["o", "^", "s", "D"][si % 4],
                     ls=["-", ":", "-.", "--"][si % 4], lw=1.8, color=COLOR[res],
                     alpha=1.0 - 0.25 * si,
                     label=f"{LABEL[res]} ({sn}) vs {LABEL[base]}")
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.fill_between(args.freqs, 0, 100, color="red", alpha=0.06)
    plt.text(args.freqs[-1], 3, "residual HURTS", ha="right", fontsize=9,
             color="darkred")
    plt.xlabel("external force frequency [Hz]")
    plt.ylabel("change in position RMSE vs own baseline [%]")
    plt.title("Does the residual still help at each frequency?\n"
              "below zero = residual improves its baseline")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "plots" / "residual_gain_vs_frequency.png", dpi=150)
    plt.close()

    # ------------------------------------------------------- report
    L = ["# Stage 2a — constant-trained policies under a time-varying force\n",
         f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
         f"- Policy set `{args.label_a}`: `{stage1}`"]
    for label, d in zip(labels, extra):
        L.append(f"- Policy set `{label}`: `{d}`")
    L += [f"- Seeds per set: "
          + ", ".join(f"{n}={len(v)}" for n, v in sets.items()),
         f"- {args.episodes} paired episodes per point, frozen eval seed "
         f"{args.eval_seed}",
         f"- Force: `A cos(2 pi f t + phi)`, time-RMS held constant across f",
         "",
         ("**No retraining.** This measures out-of-distribution generalization."
          if not extra else
          "Two policy sets on identical episodes: one trained on constant "
          "disturbances (out-of-distribution here) and one trained on "
          "time-varying disturbances (in-distribution)."),
         "",
         "## Position RMSE [m], mean over seeds", "",
         "Residual columns are labelled by the policy set they come from. The "
         "baselines are policy-free and therefore shared.", "",
         "| f [Hz] | PD | PID | "
         + " | ".join(f"{LABEL[a]} ({sn})" for sn in set_names
                      for a in ("pd_res", "pid_res")) + " |",
         "|---" * (2 + 2 * len(set_names) + 1) + "|"]
    for f in args.freqs:
        cells = [f"{cell('pd', f):.4f}", f"{cell('pid', f):.4f}"]
        for sn in set_names:
            for a in ("pd_res", "pid_res"):
                m, sd = agg(a, f, policy_set=sn)
                cells.append(f"{m:.4f} ± {sd:.4f}")
        L.append(f"| {f:g} | " + " | ".join(cells) + " |")

    L += ["", "## Position RMSE MEDIAN [m] — robust to failure count", "",
          "| f [Hz] | PD | PID | "
          + " | ".join(f"{LABEL[a]} ({sn})" for sn in set_names
                       for a in ("pd_res", "pid_res")) + " |",
          "|---" * (2 + 2 * len(set_names) + 1) + "|"]
    for f in args.freqs:
        cells = [f"{cell('pd', f, 'pos_rmse_median'):.4f}",
                 f"{cell('pid', f, 'pos_rmse_median'):.4f}"]
        for sn in set_names:
            for a in ("pd_res", "pid_res"):
                cells.append(f"{cell(a, f, 'pos_rmse_median', sn):.4f}")
        L.append(f"| {f:g} | " + " | ".join(cells) + " |")

    L += ["", "## Termination rate [%]", "",
          "| f [Hz] | PD | PID | "
          + " | ".join(f"{LABEL[a]} ({sn})" for sn in set_names
                       for a in ("pd_res", "pid_res")) + " |",
          "|---" * (2 + 2 * len(set_names) + 1) + "|"]
    for f in args.freqs:
        cells = [f"{100*cell('pd', f, 'terminated'):.1f}",
                 f"{100*cell('pid', f, 'terminated'):.1f}"]
        for sn in set_names:
            for a in ("pd_res", "pid_res"):
                cells.append(f"{100*cell(a, f, 'terminated', sn):.1f}")
        L.append(f"| {f:g} | " + " | ".join(cells) + " |")

    f0 = args.freqs[0]
    L += ["", "## Reading this", "",
          "- The MEAN charges every failed episode at the 2.0 m termination "
          "threshold, so it folds tracking quality and failure rate into one "
          "number. That makes it sensitive to the exact set of episodes drawn: "
          "across evaluation banks the PD mean ranges 0.145-0.200 while its "
          "median stays near 0.12. Compare arms WITHIN a run (they are paired "
          "on identical episodes); compare the MEDIAN across runs.",
          f"- The `f = {f0:g}` row is the training distribution. It must match "
          "the Stage 1 constant-disturbance numbers; if it does not, the "
          "time-varying implementation is wrong.",
          "- A **negative** residual gain means the residual still improves its "
          "baseline at that frequency. A **positive** value means the "
          "constant-trained residual is actively making things worse than the "
          "bare controller.",
          "- Force RMS is held constant across frequency, so a rise in MEDIAN "
          "error with f reflects frequency, not a stronger disturbance. Note "
          "that RMS-matching makes the PEAK force sqrt(2) larger for f > 0, so "
          "the rise in TERMINATION rate with frequency is partly a peak-force "
          "effect and not purely a bandwidth effect.",
          ""]

    L += ["", "## Files", "",
          "- `sweep.csv` — every (seed, frequency, arm) row",
          "- `plots/rmse_vs_frequency.png` — the four arms across frequency",
          "- `plots/residual_gain_vs_frequency.png` — does the residual still help"]
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    write_manifest(out, {"stage": "2a", "stage1_dir": str(stage1),
                         "compare_dir": extra,
                         "policy_sets": list(sets),
                         "freqs": args.freqs, "episodes": args.episodes,
                         "eval_seed": args.eval_seed, "retrained": False})

    print("\n" + "=" * 78)
    print("STAGE 2a COMPLETE")
    print("=" * 78)
    hdr = f"{'f [Hz]':>8}{'PD':>10}{'PID':>10}"
    for sn in set_names:
        hdr += f"{'pd_res/' + sn[:6]:>16}{'pid_res/' + sn[:6]:>16}"
    print(hdr + "     (median)")
    for f in args.freqs:
        line = (f"{f:>8.2f}{cell('pd', f, 'pos_rmse_median'):>10.4f}"
                f"{cell('pid', f, 'pos_rmse_median'):>10.4f}")
        for sn in set_names:
            for a in ("pd_res", "pid_res"):
                line += f"{cell(a, f, 'pos_rmse_median', sn):>16.4f}"
        print(line)
    print()
    for base, res in (("pd", "pd_res"), ("pid", "pid_res")):
        for sn in set_names:
            gains = [100 * (cell(res, f, ps=sn) - cell(base, f)) / cell(base, f)
                     for f in args.freqs]
            hurt = [f for f, g in zip(args.freqs, gains) if g > 0]
            verdict = (f"HURTS above {min(hurt):g} Hz" if hurt
                       else "helps at every frequency")
            print(f"  {LABEL[res]:<16} ({sn:<16}) vs {LABEL[base]:<4}: {verdict}")
    print(f"\nwrote {out}/REPORT.md")


if __name__ == "__main__":
    main()
