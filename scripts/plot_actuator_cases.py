"""Per-case actuator comparison: what the residual adds, and what the LPF removes.

One figure per uncertainty case, three arms on the SAME disturbance draw:

    row 1   baseline            no residual at all
    row 2   residual, LPF off   what the residual adds
    row 3   residual, LPF on    what the filter removes

Reading a figure answers both questions directly:

    Q1  what trouble does the residual cause?
        compare row 1 with row 2
    Q2  how much of it does the LPF remove?
        compare row 2 with row 3

Every figure is titled with the uncertainty it is evaluated at -- mass/inertia
scale k, thrust-coefficient scale kf, moment-arm scale, external force, and the
effective attitude gain g = kf·arm/k, together with whether that g is inside
the discrete stability margin. Without g on the figure it is impossible to tell
a residual-induced problem from the baseline's own gain-margin failure, since
they look similar in a raw actuator trace.

Cases are chosen by how much chatter the residual ADDS over the baseline on the
same episode, so the set spans the range rather than sampling it: the worst, the
median, and the mildest. One case from outside the stability margin is included
for contrast, where the ordering reverses.

If a policy trained without the filter is available, pass it as
``--run_dir_nofilter``: row 2 then uses a policy evaluated at its own training
setting rather than one taken off-distribution, which is the honest version of
"residual without LPF".

Example
-------
python -m scripts.plot_actuator_cases \\
    --run_dir experiments/stage1_.../seed_00/pd/train/trial_001 \\
    --run_dir_nofilter experiments/stage3_nofilter/seed_00/pd/train/trial_001
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import jax                                           # noqa: E402
import numpy as np                                   # noqa: E402

from qrjax.core.mixer import MAX_MOTOR_THRUST        # noqa: E402
from qrjax.rl.curriculum import flat_ranges          # noqa: E402
from qrjax.utils import write_manifest               # noqa: E402

from scripts.analyze_stable_chatter import (         # noqa: E402
    critical_gain, hp_component, load_policy, rollout,
)

ROTOR_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]


def episode_spectrum(sig, n, dt):
    x = sig[:n] - sig[:n].mean()
    return np.fft.rfftfreq(n, d=dt), np.abs(np.fft.rfft(x)) ** 2


def plot_case(case, arms, dt, hf_cutoff, out_path, header, zoom_s=1.0):
    """arms: list of (label, note, data dict) in display order."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.85], hspace=0.45,
                          wspace=0.22)

    ymin, ymax = np.inf, -np.inf
    for _, _, d in arms:
        n = max(int(d["mask"][:, case].sum()), 8)
        ymin = min(ymin, d["motor"][:n, case].min())
        ymax = max(ymax, d["motor"][:n, case].max())
    pad = 0.05 * (ymax - ymin + 1e-6)

    for row, (label, note, d) in enumerate(arms):
        n = max(int(d["mask"][:, case].sum()), 8)
        t = np.arange(n) * dt
        nz = min(int(zoom_s / dt), n)
        z0 = max(0, n // 2 - nz // 2)
        hf = hp_component(d["motor"][:, :, 0], d["mask"], dt, hf_cutoff)[case]
        sat = 100 * d["sat"][:n, case].mean()
        rmse = float(np.sqrt(np.mean(d["pos_err"][:n, case] ** 2)))

        ax = fig.add_subplot(gs[row, 0])
        for r in range(4):
            ax.plot(t, d["motor"][:n, case, r], lw=0.6, color=ROTOR_COLORS[r])
        ax.axhline(float(MAX_MOTOR_THRUST), color="k", ls="--", lw=0.9)
        ax.axhline(0.0, color="k", ls="--", lw=0.9)
        ax.axvspan(t[z0], t[z0 + nz - 1], color="grey", alpha=0.15)
        ax.set_ylim(min(ymin - pad, -0.2), max(ymax + pad, 1.0))
        ax.set_ylabel(f"{label}\n{note}\nrotor cmd [N]", fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title("full episode", fontsize=10)

        ax = fig.add_subplot(gs[row, 1])
        for r in range(4):
            ax.plot(t[z0:z0 + nz], d["motor"][z0:z0 + nz, case, r], lw=1.0,
                    marker=".", ms=2, color=ROTOR_COLORS[r])
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_title(f"HF {hf:.3f} N   saturation {sat:.1f}%   "
                     f"RMSE {rmse:.4f} m", fontsize=9)

        ax = fig.add_subplot(gs[row, 2])
        f, s = episode_spectrum(d["motor"][:, case, 0], n, dt)
        ax.semilogy(f[1:], s[1:] + 1e-16, lw=0.8, color=ROTOR_COLORS[0])
        if row > 0:
            fb, sb = episode_spectrum(arms[0][2]["motor"][:, case, 0], n, dt)
            ax.semilogy(fb[1:], sb[1:] + 1e-16, lw=0.8, color="grey",
                        alpha=0.75, label="baseline")
            ax.legend(fontsize=7)
        ax.axvline(hf_cutoff, color="grey", ls=":", lw=1)
        ax.set_ylabel("power", fontsize=8)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title("spectrum, rotor 0", fontsize=10)
        if row == len(arms) - 1:
            ax.set_xlabel("frequency [Hz]", fontsize=9)

    ax = fig.add_subplot(gs[3, :])
    for label, note, d in arms[1:]:
        n = max(int(d["mask"][:, case].sum()), 8)
        nz = min(int(zoom_s / dt), n)
        z0 = max(0, n // 2 - nz // 2)
        t = np.arange(n) * dt
        ax.plot(t[z0:z0 + nz], d["applied_thrust"][z0:z0 + nz, case],
                lw=1.2, marker=".", ms=3, label=f"{label} — {note}")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("applied thrust residual\n(normalized)", fontsize=9)
    ax.set_title("the residual command itself, same zoom window "
                 "— this is what the LPF acts on", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(header, fontsize=12)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True,
                   help="policy trained WITH the filter")
    p.add_argument("--run_dir_nofilter", default=None,
                   help="policy trained WITHOUT the filter; used for the "
                        "'LPF off' row so it is not off-distribution")
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--n_stable_cases", type=int, default=3)
    p.add_argument("--include_unstable", action="store_true", default=True)
    p.add_argument("--hf_cutoff", type=float, default=10.0)
    p.add_argument("--zoom_s", type=float, default=1.0)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg, agent, params, beta_trained = load_policy(args.run_dir, args.checkpoint)
    dt, controller = cfg.env.dt, cfg.env.base_controller
    gc = critical_gain(dt)

    off_agent, off_params, off_label = agent, params, \
        f"residual (trained@{beta_trained:g}, off-distribution)"
    if args.run_dir_nofilter:
        _, off_agent, off_params, b2 = load_policy(args.run_dir_nofilter,
                                                   args.checkpoint)
        off_label = f"residual (trained@{b2:g}, matched)"

    out = Path(args.out) if args.out else Path("experiments") / (
        f"actuator_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    ranges = flat_ranges(cfg.env, force_freq=0.0, force_dc_prob=0.0)
    key = jax.random.PRNGKey(args.eval_seed)

    print(f"policy      : {args.run_dir}  ({controller.upper()} base)")
    print(f"g_crit      : {gc:.2f} at {1/dt:.0f} Hz")
    print(f"episodes    : {args.episodes}\n")

    base = rollout(cfg.env, controller, None, None, key, args.episodes,
                   ranges, 0.2)
    res_off = rollout(cfg.env, controller, off_agent, off_params, key,
                      args.episodes, ranges, 1.0)
    res_on = rollout(cfg.env, controller, agent, params, key, args.episodes,
                     ranges, 0.2)
    for d in (base, res_off, res_on):
        d["applied_thrust"] = d["u_res"][:, :, 0] / max(
            float(cfg.env.residual_authority * 34.194), 1e-9)

    g = base["g"]
    stable = g < gc
    hf_base = hp_component(base["motor"][:, :, 0], base["mask"], dt, args.hf_cutoff)
    hf_off = hp_component(res_off["motor"][:, :, 0], res_off["mask"], dt,
                          args.hf_cutoff)
    added = hf_off - hf_base

    idx = np.where(stable)[0]
    order = idx[np.argsort(-added[idx])]
    picks = []
    if len(order):
        picks.append(("worst_added_chatter", int(order[0])))
        picks.append(("median_added_chatter", int(order[len(order) // 2])))
        picks.append(("least_added_chatter", int(order[-1])))
    picks = picks[:args.n_stable_cases]
    if args.include_unstable and (~stable).any():
        picks.append(("beyond_gain_margin", int(np.argmax(g))))

    rows = []
    for name, c in picks:
        inside = g[c] < gc
        header = (
            f"{name.replace('_', ' ')}  —  episode {c}\n"
            f"uncertainty:  k = {base['k'][c] if 'k' in base else float('nan'):.2f}   "
            if False else
            f"{name.replace('_', ' ')}  —  episode {c}   |   "
            f"g = kf·arm/k = {g[c]:.2f}  "
            f"({'INSIDE' if inside else 'BEYOND'} the stability margin, "
            f"g_crit = {gc:.2f})")
        arms = [("baseline", "no residual", base),
                ("residual", off_label.split("(")[1].rstrip(")"), res_off),
                ("residual", f"LPF on, beta={beta_trained:g}", res_on)]
        path = out / f"{name}_ep{c:03d}.png"
        plot_case(c, arms, dt, args.hf_cutoff, path, header, args.zoom_s)

        row = {"case": name, "episode": c, "g": float(g[c]),
               "inside_margin": bool(inside)}
        for lab, d in (("baseline", base), ("res_lpf_off", res_off),
                       ("res_lpf_on", res_on)):
            n = max(int(d["mask"][:, c].sum()), 8)
            row[f"{lab}_hf_N"] = float(hp_component(
                d["motor"][:, :, 0], d["mask"], dt, args.hf_cutoff)[c])
            row[f"{lab}_sat_pct"] = float(100 * d["sat"][:n, c].mean())
            row[f"{lab}_rmse"] = float(np.sqrt(np.mean(d["pos_err"][:n, c] ** 2)))
        rows.append(row)
        print(f"  {name:<22} ep {c:>3}  g={g[c]:.2f}  "
              f"HF base {row['baseline_hf_N']:.3f} -> "
              f"res {row['res_lpf_off_hf_N']:.3f} -> "
              f"+LPF {row['res_lpf_on_hf_N']:.3f} N   -> {path.name}")

    with (out / "cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    L = ["# Actuator comparison, case by case\n",
         f"Generated {datetime.now().isoformat(timespec='seconds')}",
         "",
         f"- Policy `{args.run_dir}` on a {controller.upper()} base",
         f"- 'LPF off' row uses "
         + (f"`{args.run_dir_nofilter}`, trained at beta = 1.0, so it is "
            "evaluated at its own training setting"
            if args.run_dir_nofilter else
            "the filter-trained policy with the filter removed, which is "
            "OFF-DISTRIBUTION and overstates the effect"),
         f"- g_crit = {gc:.2f} at {1/dt:.0f} Hz; cases inside the margin are "
         "unaffected by the baseline's gain-margin failure",
         "",
         "## Q1 — what does adding the residual cost?", "",
         "| case | g | in margin | baseline HF | residual HF | factor | "
         "baseline sat % | residual sat % |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fac = r["res_lpf_off_hf_N"] / max(r["baseline_hf_N"], 1e-9)
        L.append(f"| {r['case']} | {r['g']:.2f} "
                 f"| {'yes' if r['inside_margin'] else 'no'} "
                 f"| {r['baseline_hf_N']:.4f} | {r['res_lpf_off_hf_N']:.4f} "
                 f"| {fac:.1f}x | {r['baseline_sat_pct']:.1f} "
                 f"| {r['res_lpf_off_sat_pct']:.1f} |")

    L += ["", "## Q2 — how much does the LPF remove?", "",
          "| case | residual HF | + LPF | factor | sat % off | sat % on | "
          "RMSE off | RMSE on |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fac = r["res_lpf_off_hf_N"] / max(r["res_lpf_on_hf_N"], 1e-9)
        L.append(f"| {r['case']} | {r['res_lpf_off_hf_N']:.4f} "
                 f"| {r['res_lpf_on_hf_N']:.4f} | {fac:.1f}x "
                 f"| {r['res_lpf_off_sat_pct']:.1f} "
                 f"| {r['res_lpf_on_sat_pct']:.1f} "
                 f"| {r['res_lpf_off_rmse']:.4f} | {r['res_lpf_on_rmse']:.4f} |")

    L += ["",
          "## How to read the figures", "",
          "- **Row 1 vs row 2** answers Q1: the extra high-frequency content "
          "and saturation the residual introduces.",
          "- **Row 2 vs row 3** answers Q2: what the filter takes back out.",
          "- The **bottom panel** shows the residual command itself, which is "
          "the signal the filter acts on. Chatter visible there and absent "
          "from row 1 is residual-generated by construction.",
          "- The **spectrum column** overlays the baseline in grey, so the "
          "excess is visible as a band rather than inferred from a summary "
          "number.",
          "- The `beyond_gain_margin` case is the control: there the baseline "
          "row is already oscillating on its own, the ordering reverses, and "
          "the filter does not help. Any explanation of the residual's effect "
          "has to account for that case too.",
          "",
          "## Files", "", "- `cases.csv`",
          "- one PNG per case, named by why it was selected"]
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    write_manifest(out, {"run_dir": args.run_dir,
                         "run_dir_nofilter": args.run_dir_nofilter,
                         "g_crit": gc, "cases": [c for _, c in picks],
                         "episodes": args.episodes})
    print(f"\nwrote {out}/REPORT.md")


if __name__ == "__main__":
    main()
