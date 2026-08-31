"""Produce the complete, paper-ready figure and table set for one experiment.

Given a `run_stage1` output directory (or an explicit pair of policy
directories), this runs every arm on one frozen disturbance bank and writes an
organised figure tree:

    paper_figures/
      01_summary/
        arm_comparison.png          per-arm RMSE, per-seed points overlaid
        rmse_distribution.png       full per-episode distribution, not just means
        tracking_error_vs_time.png  mean error trace, all arms
        termination_and_effort.png  robustness and control cost
        uncertainty_coverage.png    the evaluation bank itself, cases marked
      02_cases/
        case_<label>_trajectory3d.png
        case_<label>_errors.png
        case_<label>_actuators.png
      03_actuators/
        actuator_summary.png        HF content, saturation, chatter per arm
        spectra.png                 rotor-command spectra, all arms
      TABLES.md                     every number in the figures, as markdown
      summary.csv, cases.csv, per_episode.csv

Two conventions are enforced throughout, because both have caused confusion:

*Every figure names its uncertainty case.* Titles carry k, |F|, kf, arm, the
effective attitude gain g = kf·arm/k, and whether that g sits inside the
discrete stability margin. Without g on the figure a residual-induced problem
and a baseline gain-margin failure look identical in a raw actuator trace.

*Distributions are shown, not only means.* With a few-percent termination rate
the mean is driven by a small number of failed episodes, so `rmse_distribution`
shows the whole spread and the tables report median alongside mean.

Example
-------
python -m scripts.make_paper_figures \\
    --experiment experiments/lpf_study_20260901/both --episodes 256
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

from scripts.analyze_stable_chatter import critical_gain, hp_component  # noqa: E402
from scripts.plot_trajectories import (              # noqa: E402
    ARMS, LABEL, USES_RESIDUAL, load_policy, rollout_all,
    plot_trajectory, plot_actuators, plot_errors,
)

COLOR = {"pd": "#999999", "pid": "#444444",
         "pd_res": "#2ca02c", "pid_res": "#1f77b4"}


def find_policies(experiment: Path):
    """Locate the {pd, pid} training trials inside a run_stage1 output."""
    seeds = sorted(experiment.glob("seed_*"))
    if not seeds:
        raise SystemExit(
            f"no seed_* under {experiment}. Point --experiment at a "
            "run_stage1 output directory, or use --run_dir_pd/--run_dir_pid.")
    out = []
    for sd in seeds:
        entry = {}
        for base in ("pd", "pid"):
            trials = sorted((sd / base / "train").glob("trial_*"))
            if trials:
                entry[base] = trials[-1]
        if len(entry) == 2:
            out.append((sd.name, entry))
    if not out:
        raise SystemExit(f"no complete {{pd,pid}} pairs under {experiment}")
    return out


def case_label(draw, g, gc, c):
    return (f"k={draw['k'][c]:.2f}  |F|={np.linalg.norm(draw['force'][c]):.2f} N  "
            f"kf={draw['kf'][c]:.2f}  arm={draw['arm'][c]:.2f}  "
            f"g={g[c]:.2f} ({'inside' if g[c] < gc else 'BEYOND'} margin)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default=None,
                   help="a run_stage1 output directory")
    p.add_argument("--run_dir_pd", default=None)
    p.add_argument("--run_dir_pid", default=None)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--force_freq", type=float, default=0.0)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--n_cases", type=int, default=4)
    p.add_argument("--zoom_s", type=float, default=1.0)
    p.add_argument("--hf_cutoff", type=float, default=10.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.experiment:
        seed_dirs = find_policies(Path(args.experiment))
        seed_name, dirs = seed_dirs[0]
        pd_dir, pid_dir = dirs["pd"], dirs["pid"]
        default_out = Path(args.experiment) / "paper_figures"
    else:
        if not (args.run_dir_pd and args.run_dir_pid):
            raise SystemExit("give --experiment, or both --run_dir_pd and "
                             "--run_dir_pid")
        seed_dirs = [("seed_00", {"pd": Path(args.run_dir_pd),
                                  "pid": Path(args.run_dir_pid)})]
        seed_name, pd_dir, pid_dir = "seed_00", Path(args.run_dir_pd), Path(args.run_dir_pid)
        default_out = Path(args.run_dir_pd).parent / "paper_figures"

    out = Path(args.out) if args.out else default_out
    for sub in ("01_summary", "02_cases", "03_actuators"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cfg, agent_pd, params_pd = load_policy(pd_dir, args.checkpoint)
    _, agent_pid, params_pid = load_policy(pid_dir, args.checkpoint)
    dt = cfg.env.dt
    gc = critical_gain(dt)
    ranges = flat_ranges(cfg.env, force_freq=args.force_freq, force_dc_prob=0.0)
    key = jax.random.PRNGKey(args.eval_seed)

    print(f"experiment  : {args.experiment or pd_dir.parent}")
    print(f"control rate: {1/dt:.0f} Hz   g_crit = {gc:.2f}")
    print(f"episodes    : {args.episodes}, eval seed {args.eval_seed}\n")

    # ---- all seeds, for the per-seed points on the summary bars ----
    per_seed_rmse = {a: [] for a in ARMS}
    traces = draw = None
    for sname, dirs in seed_dirs:
        _, ag_pd, pr_pd = load_policy(dirs["pd"], args.checkpoint)
        _, ag_pid, pr_pid = load_policy(dirs["pid"], args.checkpoint)
        tr, dw = rollout_all(cfg.env, ag_pd, pr_pd, ag_pid, pr_pid,
                             key, args.episodes, ranges)
        for arm in ARMS:
            *_, mask, pe, pt, alive = tr[arm]
            per_seed_rmse[arm].append(
                np.sqrt(np.mean(pe ** 2, axis=0)))
        if traces is None:
            traces, draw = tr, dw
        print(f"  {sname} done")

    g = draw["kf"] * draw["arm"] / draw["k"]
    fmag = np.linalg.norm(draw["force"], axis=-1)

    # ---------------- per-episode metrics ----------------
    metrics = {}
    for arm in ARMS:
        x_true, x_nom, x_des, motor, act, mask, pe, pt, alive = traces[arm]
        rmse = np.sqrt(np.mean(pe ** 2, axis=0))
        hf = hp_component(motor[:, :, 0], mask, dt, args.hf_cutoff)
        n = mask.sum(axis=0)
        metrics[arm] = {
            "rmse": rmse, "terminated": 1.0 - alive.astype(float),
            "hf": hf,
            "sat": np.array([float(np.mean(np.abs(motor[:int(n[e]), e]).max(axis=1)
                                           >= 0.999 * float(MAX_MOTOR_THRUST)))
                             for e in range(args.episodes)]),
            "effort": np.sqrt(np.mean(np.sum(act ** 2, axis=-1), axis=0)),
        }

    rows = []
    for e in range(args.episodes):
        for arm in ARMS:
            rows.append({"episode": e, "arm": arm, "label": LABEL[arm],
                         "k": float(draw["k"][e]), "force": float(fmag[e]),
                         "kf": float(draw["kf"][e]), "arm_scale": float(draw["arm"][e]),
                         "g": float(g[e]), "inside_margin": bool(g[e] < gc),
                         **{k2: float(v[e]) for k2, v in metrics[arm].items()}})
    with (out / "per_episode.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for arm in ARMS:
        m = metrics[arm]
        summary.append({
            "arm": arm, "label": LABEL[arm],
            "rmse_mean": float(m["rmse"].mean()),
            "rmse_median": float(np.median(m["rmse"])),
            "rmse_p90": float(np.percentile(m["rmse"], 90)),
            "terminated_pct": float(100 * m["terminated"].mean()),
            "motor_hf_N": float(m["hf"].mean()),
            "saturation_pct": float(100 * m["sat"].mean()),
            "effort_rms": float(m["effort"].mean()),
        })
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # ================= 01 SUMMARY =================
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ARMS))
    means = [np.mean(np.concatenate(per_seed_rmse[a])) for a in ARMS]
    ax.bar(x, means, color=[COLOR[a] for a in ARMS])
    for i, a in enumerate(ARMS):
        for sd in per_seed_rmse[a]:
            ax.plot(i, sd.mean(), "k.", ms=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[a] for a in ARMS], fontsize=9)
    ax.set_ylabel("Position RMSE [m]")
    ax.set_title(f"Tracking accuracy, {args.episodes} paired episodes\n"
                 f"{1/dt:.0f} Hz, dots = individual seeds")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "01_summary" / "arm_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.boxplot([metrics[a]["rmse"] for a in ARMS], tick_labels=[LABEL[a] for a in ARMS],
               showfliers=True, medianprops=dict(color="k", lw=2))
    ax.set_ylabel("Position RMSE [m]")
    ax.set_yscale("log")
    ax.set_title("Per-episode distribution\n"
                 "the mean is pulled by the failed-episode tail; read the median")
    ax.grid(axis="y", alpha=0.3, which="both")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "01_summary" / "rmse_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    for arm in ARMS:
        *_, mask, pe, pt, alive = traces[arm]
        t = np.arange(pe.shape[0]) * dt
        ax.plot(t, pe.mean(axis=1), lw=1.5, color=COLOR[arm], label=LABEL[arm])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean ‖x − x_des‖ [m]")
    ax.set_title(f"Tracking error over time, mean of {args.episodes} episodes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "01_summary" / "tracking_error_vs_time.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, field, title in ((axes[0], "terminated_pct", "Termination rate [%]"),
                             (axes[1], "saturation_pct", "Actuator saturation [%]"),
                             (axes[2], "effort_rms", "Residual command RMS")):
        vals = [next(r[field] for r in summary if r["arm"] == a) for a in ARMS]
        ax.bar(x, vals, color=[COLOR[a] for a in ARMS])
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[a] for a in ARMS], fontsize=8, rotation=12)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Robustness and control cost")
    fig.tight_layout()
    fig.savefig(out / "01_summary" / "termination_and_effort.png", dpi=150)
    plt.close(fig)

    # ---- pick cases spanning the uncertainty space ----
    cases = [("benign", int(np.argmin(g + fmag))),
             ("max_force", int(np.argmax(fmag))),
             ("max_mass_error", int(np.argmax(np.abs(draw["k"] - 1.0)))),
             ("max_gain", int(np.argmax(g)))]
    seen, picks = set(), []
    for lab, c in cases:
        if c not in seen:
            seen.add(c)
            picks.append((lab, c))
    picks = picks[:args.n_cases]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sc = ax.scatter(g, fmag, c=metrics["pid_res"]["rmse"], cmap="viridis",
                    s=22, alpha=0.85)
    ax.axvline(gc, color="r", ls="--", lw=1.5)
    ax.axvspan(gc, max(g.max(), gc * 1.05), color="red", alpha=0.07)
    ax.text(gc, fmag.max(), f" g_crit = {gc:.2f}", color="r", fontsize=9,
            rotation=90, va="top")
    for lab, c in picks:
        ax.scatter([g[c]], [fmag[c]], s=170, facecolors="none",
                   edgecolors="k", lw=1.8)
        ax.annotate(lab, (g[c], fmag[c]), fontsize=8,
                    textcoords="offset points", xytext=(9, 5))
    plt.colorbar(sc, ax=ax, label="PID+residual RMSE [m]")
    ax.set_xlabel("effective attitude gain  g = kf·arm / k")
    ax.set_ylabel("|F_ext| [N]")
    ax.set_title(f"The evaluation bank: {args.episodes} episodes\n"
                 "circled points are the cases plotted individually")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "01_summary" / "uncertainty_coverage.png", dpi=150)
    plt.close(fig)

    # ================= 02 CASES =================
    case_rows = []
    for lab, c in picks:
        header = f"{lab}  —  episode {c}   |   {case_label(draw, g, gc, c)}"
        plot_trajectory(c, traces, draw,
                        out / "02_cases" / f"case_{lab}_trajectory3d.png", header)
        plot_actuators(c, traces, out / "02_cases" / f"case_{lab}_actuators.png",
                       dt, header, args.zoom_s)
        plot_errors(c, traces, out / "02_cases" / f"case_{lab}_errors.png",
                    dt, header)
        row = {"case": lab, "episode": c, "k": float(draw["k"][c]),
               "force": float(fmag[c]), "kf": float(draw["kf"][c]),
               "arm": float(draw["arm"][c]), "g": float(g[c]),
               "inside_margin": bool(g[c] < gc)}
        for arm in ARMS:
            row[f"rmse_{arm}"] = float(metrics[arm]["rmse"][c])
            row[f"hf_{arm}"] = float(metrics[arm]["hf"][c])
        case_rows.append(row)
        print(f"  case {lab:<16} ep {c:>3}  g={g[c]:.2f}")
    with (out / "cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(case_rows[0]))
        w.writeheader()
        w.writerows(case_rows)

    # ================= 03 ACTUATORS =================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, field, title in (
            (axes[0], "motor_hf_N", f"Rotor command RMS above {args.hf_cutoff:g} Hz [N]"),
            (axes[1], "saturation_pct", "Saturation [%]"),
            (axes[2], "rmse_median", "Median RMSE [m]")):
        vals = [next(r[field] for r in summary if r["arm"] == a) for a in ARMS]
        ax.bar(x, vals, color=[COLOR[a] for a in ARMS])
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[a] for a in ARMS], fontsize=8, rotation=12)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Actuator behaviour "
                 f"(mean rotor command ≈ {np.mean([traces['pid'][3][:, :, 0].mean()]):.1f} N)")
    fig.tight_layout()
    fig.savefig(out / "03_actuators" / "actuator_summary.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    nfft = 1024
    for arm in ARMS:
        *_, motor, act, mask, pe, pt, alive = (None,) + traces[arm][1:]
        motor = traces[arm][3]
        acc, cnt = None, 0
        for e in range(args.episodes):
            nn = max(int(traces[arm][5][:, e].sum()), 16)
            xx = motor[:nn, e, 0] - motor[:nn, e, 0].mean()
            xx = np.pad(xx, (0, nfft - nn)) if nn < nfft else xx[:nfft]
            pw = np.abs(np.fft.rfft(xx * np.hanning(len(xx)))) ** 2
            acc = pw if acc is None else acc + pw
            cnt += 1
        freqs = np.fft.rfftfreq(nfft, d=dt)
        ax.semilogy(freqs[1:], acc[1:] / cnt, lw=1.4, color=COLOR[arm],
                    label=LABEL[arm])
    ax.axvline(args.hf_cutoff, color="grey", ls=":", lw=1)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("mean power, rotor 0 command")
    ax.set_title("Rotor command spectra, averaged over episodes")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "03_actuators" / "spectra.png", dpi=150)
    plt.close(fig)

    # ================= TABLES =================
    L = [f"# Evaluation — {args.experiment or pd_dir.parent}\n",
         f"Generated {datetime.now().isoformat(timespec='seconds')}",
         "",
         f"- Control rate **{1/dt:.0f} Hz**, g_crit = **{gc:.2f}**",
         f"- {args.episodes} paired episodes, frozen eval seed {args.eval_seed}",
         f"- External-force frequency {args.force_freq:g} Hz",
         f"- Episodes beyond the stability margin: "
         f"**{int((g >= gc).sum())} / {args.episodes}** ({100*(g >= gc).mean():.1f}%)",
         "",
         "## Summary", "",
         "| Arm | RMSE mean | RMSE median | RMSE p90 | Terminated | "
         "Rotor HF [N] | Saturation | Effort |",
         "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        L.append(f"| {r['label']} | {r['rmse_mean']:.4f} | {r['rmse_median']:.4f} "
                 f"| {r['rmse_p90']:.4f} | {r['terminated_pct']:.1f}% "
                 f"| {r['motor_hf_N']:.4f} | {r['saturation_pct']:.2f}% "
                 f"| {r['effort_rms']:.3f} |")
    L += ["",
          "Read the **median** for typical tracking and the termination column "
          "for robustness. The mean folds both together and is dominated by "
          "the failed-episode tail.",
          "",
          "## Uncertainty cases plotted individually", "",
          "| case | ep | k | \\|F\\| [N] | kf | arm | g | in margin | "
          + " | ".join(f"RMSE {LABEL[a]}" for a in ARMS) + " |",
          "|---" * (8 + len(ARMS)) + "|"]
    for r in case_rows:
        L.append(f"| {r['case']} | {r['episode']} | {r['k']:.2f} "
                 f"| {r['force']:.2f} | {r['kf']:.2f} | {r['arm']:.2f} "
                 f"| {r['g']:.2f} | {'yes' if r['inside_margin'] else 'NO'} | "
                 + " | ".join(f"{r[f'rmse_{a}']:.4f}" for a in ARMS) + " |")
    L += ["", "## Figures", "",
          "**01_summary** — `arm_comparison` (per-seed dots), "
          "`rmse_distribution` (full spread), `tracking_error_vs_time`, "
          "`termination_and_effort`, `uncertainty_coverage` (the bank itself, "
          "with g_crit marked and the plotted cases circled).",
          "",
          "**02_cases** — for each case: `trajectory3d` (desired / nominal twin "
          "/ true, per arm), `errors` (tracking error, twin discrepancy, "
          "residual command), `actuators` (full episode plus zoom). Every "
          "title carries the full uncertainty description.",
          "",
          "**03_actuators** — `actuator_summary` (HF content, saturation, "
          "median RMSE), `spectra` (rotor-command spectra, all arms).",
          "",
          "## Note on the twin", "",
          "The 3-D figures show the desired path, the disturbance-free nominal "
          "twin, and the true disturbed plant separately. Desired-vs-twin is "
          "what the BASE controller cannot do; twin-vs-true is what the "
          "DISTURBANCE does and what the residual is trained to remove. "
          "Plotting only desired-vs-true conflates the two."]
    (out / "TABLES.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    write_manifest(out, {"experiment": str(args.experiment or pd_dir.parent),
                         "episodes": args.episodes, "eval_seed": args.eval_seed,
                         "control_hz": 1 / dt, "g_crit": gc,
                         "force_freq": args.force_freq,
                         "cases": {lab: c for lab, c in picks}})
    print(f"\nwrote {out}")
    print("  TABLES.md, 01_summary/, 02_cases/, 03_actuators/")


if __name__ == "__main__":
    main()
