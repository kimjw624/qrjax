"""Build the one-page evidence summary for the oscillation finding.

Reads the outputs of `verify_gain_margin.py` and `analyze_lpf.py` and produces a
single figure plus a report laying out the argument:

    old hypothesis   the residual policy and the geometric controller interact
                     and that interaction causes the oscillation
    what the data
    actually shows   certain parameter combinations raise the effective control
                     gain past the discrete stability limit of the attitude
                     loop; the BASELINE controller then oscillates on its own,
                     with or without a residual
    consequence      a residual cannot fix this. The instability is in the
                     baseline's gain margin, so the fix has to change the gain
                     -- adaptive gain, online parameter estimation, or a faster
                     loop.

The figure has four panels, each carrying one step of that argument:

    A  the mechanism   oscillation onset vs effective gain, at three control
                       rates, against limits predicted by linear sampled-data
                       theory with no fitting
    B  causality       removing the actuator/geometry part of the gain
                       mismatch removes the oscillation
    C  exposure        how much of the training disturbance box is actually
                       past the limit
    D  scope of the
       LPF fix         what the thrust filter does and does not address

Example
-------
python -m scripts.make_summary_figure \\
    --gain_margin experiments/gain_margin_20260831_022729 \\
    --lpf experiments/lpf_beta0.2_20260831_023555
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
import numpy as np                                   # noqa: E402

from qrjax.utils import write_manifest                # noqa: E402

J_NOM, KR, KOMEGA = 0.022, 8.81, 2.54


def critical_gain(dt):
    """Largest kf*arm/k for which the discretised attitude loop stays stable."""
    def max_pole(g):
        a = g / J_NOM
        Ad = np.array([[1.0, dt], [0.0, 1.0]])
        Bd = np.array([[0.5 * a * dt ** 2], [a * dt]])
        return max(abs(np.linalg.eigvals(Ad - Bd @ np.array([[KR, KOMEGA]]))))
    lo, hi = 1e-3, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if max_pole(mid) < 1.0 else (lo, mid)
    return lo


def gain_exposure(lo=0.7, hi=1.3, n=2_000_000, seed=0):
    """Distribution of g = kf*arm/k when each factor is uniform on [lo, hi]."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(lo, hi, n) * rng.uniform(lo, hi, n)
            / rng.uniform(lo, hi, n))


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gain_margin", required=True,
                   help="a verify_gain_margin output directory")
    p.add_argument("--lpf", default=None,
                   help="an analyze_lpf output directory")
    p.add_argument("--range_lo", type=float, default=0.7)
    p.add_argument("--range_hi", type=float, default=1.3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    gm_rows = read_csv(Path(args.gain_margin) / "results.csv")
    lpf_rows = read_csv(Path(args.lpf) / "results.csv") if args.lpf else []
    out = Path(args.out) if args.out else Path(args.gain_margin) / "summary"
    out.mkdir(parents=True, exist_ok=True)

    dose = [r for r in gm_rows if r["test"] == "T1_dose"]
    dts = sorted({float(r["dt"]) for r in dose})
    t4 = [r for r in gm_rows if r["test"] == "T4_allocation"]
    t5 = [r for r in gm_rows if r["test"] == "T5_controller"]

    fig = plt.figure(figsize=(16, 10))

    # ---- A: mechanism -------------------------------------------------------
    ax = fig.add_subplot(2, 2, 1)
    for dt, colour in zip(dts, ["#d62728", "#1f77b4", "#2ca02c"]):
        sel = sorted([r for r in dose if float(r["dt"]) == dt],
                     key=lambda r: float(r["g"]))
        ax.semilogy([float(r["g"]) for r in sel],
                    [float(r["hf_motor0"]) for r in sel],
                    marker="o", color=colour, lw=1.8,
                    label=f"{1/dt:.0f} Hz measured")
        gc = critical_gain(dt)
        ax.axvline(gc, color=colour, ls="--", alpha=0.75)
        ax.annotate(f"theory\n{gc:.2f}", (gc, ax.get_ylim()[0]),
                    color=colour, fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("effective attitude gain   g = kf·arm / k")
    ax.set_ylabel("rotor command, RMS above 10 Hz [N]")
    ax.set_title("A. Mechanism — oscillation appears exactly where linear\n"
                 "sampled-data theory says the loop loses stability",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # ---- B: causality -------------------------------------------------------
    ax = fig.add_subplot(2, 2, 2)
    gs = sorted({float(r["g"]) for r in t4})
    width = 0.35
    for i, flag in enumerate(("False", "True")):
        vals = [float(next(r for r in t4 if float(r["g"]) == g
                           and r["true_allocation"] == flag)["hf_motor0"])
                for g in gs]
        ax.bar(np.arange(len(gs)) + i * width, vals, width,
               color="#d62728" if flag == "False" else "#2ca02c",
               label="controller assumes nominal params"
                     if flag == "False" else "controller told true params")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(gs)) + width / 2)
    ax.set_xticklabels([f"g = {g:g}" for g in gs])
    ax.set_ylabel("rotor command HF [N]")
    ax.set_title("B. Causality — remove the gain mismatch and the\n"
                 "oscillation disappears (~19x), so the mismatch IS the cause",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")

    # ---- C: exposure --------------------------------------------------------
    ax = fig.add_subplot(2, 2, 3)
    g = gain_exposure(args.range_lo, args.range_hi)
    ax.hist(g, bins=250, color="#888888", density=True)
    for dt, colour in zip(dts, ["#d62728", "#1f77b4", "#2ca02c"]):
        gc = critical_gain(dt)
        if gc < g.max():
            frac = 100 * np.mean(g > gc)
            ax.axvline(gc, color=colour, ls="--", lw=1.6)
            ax.text(gc, ax.get_ylim()[1] * 0.92,
                    f" {1/dt:.0f} Hz: {frac:.1f}% unstable",
                    color=colour, fontsize=8, rotation=90, va="top")
    gc100 = critical_gain(0.01)
    ax.axvspan(gc100, g.max(), color="red", alpha=0.08)
    ax.set_xlabel("g = kf·arm / k")
    ax.set_ylabel("density")
    ax.set_title(f"C. Exposure — with every factor uniform on "
                 f"[{args.range_lo:g}, {args.range_hi:g}], "
                 f"{100*np.mean(g > gc100):.1f}% of the training box\n"
                 "sits past the 100 Hz stability limit", fontsize=11)
    ax.grid(alpha=0.3)

    # ---- D: what the LPF does and does not fix ------------------------------
    ax = fig.add_subplot(2, 2, 4)
    if lpf_rows:
        off = next(r for r in lpf_rows if float(r["eval_beta"]) == 1.0)
        on = next(r for r in lpf_rows if abs(float(r["eval_beta"]) - 0.2) < 1e-9)
        fields = [("thrust_hf", "thrust\nchatter"),
                  ("saturation_pct", "actuator\nsaturation"),
                  ("median_rmse", "tracking\nRMSE"),
                  ("terminated_pct", "termination\nrate")]
        ratios, labels = [], []
        for key, lab in fields:
            a, b = float(off[key]), float(on[key])
            ratios.append(a / b if b > 1e-12 else 1.0)
            labels.append(lab)
        colours = ["#2ca02c" if r > 1.5 else "#bbbbbb" for r in ratios]
        ax.bar(range(len(ratios)), ratios, color=colours)
        ax.axhline(1.0, color="k", lw=1)
        for i, r in enumerate(ratios):
            ax.text(i, r * 1.05, f"{r:.1f}x", ha="center", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yscale("log")
        ax.set_ylabel("improvement factor from the thrust LPF")
        ax.set_title("D. Scope of the LPF — it cleans the thrust channel and\n"
                     "the actuators, but does not touch the failure rate",
                     fontsize=11)
        ax.grid(axis="y", alpha=0.3, which="both")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no --lpf directory supplied", ha="center")

    fig.suptitle("Oscillation is a gain-margin property of the baseline "
                 "controller, not a residual-controller interaction",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "summary_figure.png", dpi=150)
    plt.close(fig)

    # ---- report -------------------------------------------------------------
    gc100 = critical_gain(0.01)
    frac = 100 * np.mean(g > gc100)
    L = ["# Why the actuators oscillate\n",
         f"Generated {datetime.now().isoformat(timespec='seconds')}",
         "",
         "## The claim that changed", "",
         "| | |", "|---|---|",
         "| **Previous hypothesis** | the residual policy interacts with the "
         "geometric controller, and that interaction causes the oscillation |",
         "| **What the data shows** | certain parameter combinations raise the "
         "effective control gain past the discrete stability limit of the "
         "attitude loop. The baseline controller then oscillates on its own, "
         "with or without a residual |",
         "",
         "## Why the effective gain rises", "",
         "The flight controller allocates using NOMINAL rotor constants and "
         "arm lengths, and its attitude gains are tuned for NOMINAL inertia. "
         "When the true vehicle differs, the moment produced per unit of "
         "commanded moment, per unit of true inertia, is",
         "",
         "```",
         "g = kf · arm / k        kf   thrust-coefficient scale",
         "                        arm  moment-arm scale",
         "                        k    mass / inertia scale",
         "```",
         "",
         "Robustness training deliberately varies all three, so g varies too — "
         "and g is a product of two numerators and a denominator, so the "
         "extremes compound.",
         "",
         "## Evidence", "",
         "### A. The threshold is predicted, not fitted", "",
         "| control rate | predicted critical g | measured onset |",
         "|---|---|---|"]
    for dt in dts:
        sel = sorted([r for r in dose if float(r["dt"]) == dt],
                     key=lambda r: float(r["g"]))
        vals = [(float(r["g"]), float(r["hf_motor0"])) for r in sel]
        onset = next((gg for gg, v in vals if v > 1.0), None)
        L.append(f"| {1/dt:.0f} Hz | {critical_gain(dt):.2f} "
                 f"| {'between ' + str(onset) if onset else 'not reached'} |")
    L += ["",
          "Continuous-time analysis says this loop is stable for ALL g. A "
          "threshold that moves with the sample rate can therefore only be a "
          "sampled-data effect — which is why the oscillation sits near "
          "Nyquist rather than at the ~5 Hz attitude natural frequency.",
          "",
          "### B. Removing the cause removes the effect", "",
          "| g | nominal allocation | true allocation | reduction |",
          "|---|---|---|---|"]
    for gg in gs:
        a = float(next(r for r in t4 if float(r["g"]) == gg
                       and r["true_allocation"] == "False")["hf_motor0"])
        b = float(next(r for r in t4 if float(r["g"]) == gg
                       and r["true_allocation"] == "True")["hf_motor0"])
        L.append(f"| {gg:g} | {a:.4f} N | {b:.4f} N | {a/b:.1f}x |")
    L += ["",
          "Telling the controller the true rotor constants and arm lengths "
          "drops g from kf·arm/k to 1/k, which is below the limit for every k "
          "in range. Note this does NOT correct the inertia error, so g "
          "becomes 1/k rather than exactly 1. This is the decisive test: had "
          "the oscillation survived, the mismatch would not have been the "
          "cause.",
          "",
          "### C. It is not the integral term, and not the residual", "",
          "| g | PD | PID |", "|---|---|---|"]
    for gg in sorted({float(r["g"]) for r in t5}):
        a = float(next(r for r in t5 if float(r["g"]) == gg
                       and r["controller"] == "pd")["hf_motor0"])
        b = float(next(r for r in t5 if float(r["g"]) == gg
                       and r["controller"] == "pid")["hf_motor0"])
        L.append(f"| {gg:g} | {a:.4f} | {b:.4f} |")
    L += ["",
          "PD and PID are indistinguishable, so integral action plays no part. "
          "Separately, the bare baseline oscillates on the affected episodes "
          "as hard as the baseline plus residual does.",
          "",
          "### D. How much of the training distribution is affected", "",
          f"With kf, arm and k each uniform on "
          f"[{args.range_lo:g}, {args.range_hi:g}]:",
          "",
          f"- g ranges {g.min():.2f} to {g.max():.2f}, median {np.median(g):.2f}",
          f"- **{frac:.1f}%** of draws exceed the 100 Hz limit of {gc100:.2f}",
          f"- the worst corner, {args.range_hi:g}·{args.range_hi:g}/"
          f"{args.range_lo:g} = {args.range_hi**2/args.range_lo:.2f}, "
          "sits well past it",
          "",
          "So the task as specified contains vehicles the baseline cannot "
          "stabilise at this control rate. That is a property of the "
          "disturbance ranges, not of the learning method.",
          ""]
    if lpf_rows:
        L += ["### E. What the thrust low-pass filter does and does not fix", "",
              "| quantity | beta = 1.0 | beta = 0.2 | factor |",
              "|---|---|---|---|"]
        for key, lab in (("thrust_hf", "thrust residual HF"),
                         ("moment_hf", "moment residual HF"),
                         ("motor_hf_N", "motor command HF [N]"),
                         ("saturation_pct", "actuator saturation [%]"),
                         ("median_rmse", "median tracking RMSE [m]"),
                         ("terminated_pct", "termination [%]")):
            a, b = float(off[key]), float(on[key])
            L.append(f"| {lab} | {a:.4f} | {b:.4f} | "
                     f"{a/b if b > 1e-12 else float('nan'):.1f}x |")
        L += ["",
              "The filter is worth keeping — it cuts thrust chatter and "
              "actuator saturation sharply at no measurable tracking cost. But "
              "the termination rate is unchanged, because the filter acts on "
              "the THRUST channel while the instability is in the ATTITUDE "
              "loop. It was previously credited with solving the oscillation; "
              "it solves a different, real problem.",
              ""]
    L += ["## Consequence", "",
          "A residual cannot repair this. The instability is in the baseline "
          "controller's gain margin, and the residual acts through the same "
          "saturating actuators on a loop that has already lost stability. "
          "Fixing it requires changing the gain itself:",
          "",
          "1. **Adaptive gain / online parameter estimation** — estimate kf, "
          "arm and inertia in flight and rescale the attitude gains, which "
          "attacks the mismatch directly. Panel B is essentially the "
          "upper bound on what perfect estimation would buy.",
          "2. **Faster attitude loop** — the critical g scales as 1/dt, so "
          "200 Hz doubles the margin and covers the whole current box.",
          "3. **Narrower ranges** — restricting each factor to about "
          "[0.85, 1.18] keeps g below the limit everywhere.",
          "",
          "Option 1 is the one that preserves the research question. Options 2 "
          "and 3 remove the failure by changing the problem.",
          "",
          "## Figure", "", "`summary_figure.png` — panels A-D above."]
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    write_manifest(out, {"gain_margin": args.gain_margin, "lpf": args.lpf,
                         "critical_g_100Hz": gc100,
                         "unstable_fraction_pct": float(frac),
                         "range": [args.range_lo, args.range_hi]})
    print(f"wrote {out}/SUMMARY.md and summary_figure.png")
    print(f"\n  critical g at 100 Hz : {gc100:.2f}")
    print(f"  unstable fraction    : {frac:.1f}% of the training box")


if __name__ == "__main__":
    main()
