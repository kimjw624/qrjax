"""Complete figure and table set for the final study.

Reproduces everything the earlier per-experiment evaluation produced, but on
the survivor bank and across all six arms:

    01_summary/   arm_comparison, rmse_distribution, tracking_error_vs_time,
                  termination_and_effort, uncertainty_coverage
    02_cases/     trajectory3d / errors / actuators for four cases chosen to
                  span the uncertainty space
    03_actuators/ actuator_summary, spectra
    figures/      fig2 presets and fig3_summary
    per_episode.csv, paired_tests.csv

Every case figure carries its full uncertainty description -- k, |F|, kf, arm
and the effective attitude gain g = kf*arm/k -- because a residual-induced
problem and a baseline gain-margin failure look alike in a raw actuator trace
and can only be told apart by g.

Axis limits are shared across panels within a figure. Autoscaling each panel to
its own data gives a badly-tracking arm a zoomed-out view that makes it look
comparable to a good one, which is the opposite of what these figures are for.
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

J_NOM, KR, KOMEGA = 0.022, 8.81, 2.54
ROTOR_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
COLOR = {"PD": "#999999", "PID": "#444444",
         "PD + RRL": "#8fce8f", "PID + RRL": "#8fb4e0",
         "PD + RRL + LPF": "#2ca02c", "PID + RRL + LPF": "#1f77b4"}
# Rows pair a base controller with its residual variants.
CASE_ORDER = ["PD", "PD + RRL", "PD + RRL + LPF",
              "PID", "PID + RRL", "PID + RRL + LPF"]


def critical_gain(dt):
    return 2 * J_NOM / (dt * (KOMEGA - dt * KR / 2))


def _paired_p(x, y, rng, n=20000):
    d = np.asarray(y) - np.asarray(x)
    idx = rng.integers(0, d.size, size=(n, d.size))
    b = d[idx].mean(axis=1)
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))


def pick_cases(draw, g, n_cases=4):
    fmag = np.linalg.norm(draw["force"], axis=-1)
    cand = [("benign", int(np.argmin(g + fmag))),
            ("max_force", int(np.argmax(fmag))),
            ("max_mass_error", int(np.argmax(np.abs(draw["k"] - 1.0)))),
            ("max_gain", int(np.argmax(g)))]
    seen, out = set(), []
    for lab, c in cand:
        if c not in seen:
            seen.add(c)
            out.append((lab, c))
    return out[:n_cases]


def case_header(lab, c, draw, g, gc):
    f = np.linalg.norm(draw["force"][c])
    return (f"{lab.replace('_', ' ')} — episode {c}   |   "
            f"k={draw['k'][c]:.2f}  |F|={f:.2f} N  kf={draw['kf'][c]:.2f}  "
            f"arm={draw['arm'][c]:.2f}  g={g[c]:.2f} "
            f"({'inside' if g[c] < gc else 'BEYOND'} margin)")


def plot_case_traj(keep, M, c, header, path, arms=CASE_ORDER):
    fig = plt.figure(figsize=(17, 10))
    lim = [[np.inf, -np.inf] for _ in range(3)]
    for nm in arms:
        xt, xn = keep[nm][7], keep[nm][8]
        for j, sgn in enumerate((1, 1, -1)):
            for series in (xt, xn):
                v = sgn * series[:, c, j]
                lim[j][0] = min(lim[j][0], float(v.min()))
                lim[j][1] = max(lim[j][1], float(v.max()))
    pads = [0.08 * (hi - lo + 1e-6) for lo, hi in lim]
    for i, nm in enumerate(arms):
        xt, xn = keep[nm][7], keep[nm][8]
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.plot(xn[:, c, 0], xn[:, c, 1], -xn[:, c, 2], color="tab:blue",
                ls="--", lw=1.7, label="nominal")
        ax.plot(xt[:, c, 0], xt[:, c, 1], -xt[:, c, 2], color="tab:red",
                ls="-", lw=1.7, label="true")
        ax.set_title(f"{nm}\nRMSE {M[nm]['rmse'][c]:.4f} m", fontsize=13)
        ax.set_xlim(lim[0][0] - pads[0], lim[0][1] + pads[0])
        ax.set_ylim(lim[1][0] - pads[1], lim[1][1] + pads[1])
        ax.set_zlim(lim[2][0] - pads[2], lim[2][1] + pads[2])
        for setter, t in ((ax.set_xlabel, "N [m]"), (ax.set_ylabel, "E [m]"),
                          (ax.set_zlabel, "up [m]")):
            setter(t, fontsize=11, labelpad=6)
        ax.tick_params(labelsize=9)
        for a in "xyz":
            ax.locator_params(axis=a, nbins=4)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=14, frameon=True)
    fig.suptitle(header, fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_case_errors(keep, c, dt, header, path, arms=CASE_ORDER):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for nm in arms:
        pe, pt, act = keep[nm][0], keep[nm][1], keep[nm][4]
        t = np.arange(pe.shape[0]) * dt
        axes[0].plot(t, pe[:, c], lw=1.5, color=COLOR[nm], label=nm)
        axes[1].plot(t, pt[:, c], lw=1.5, color=COLOR[nm], label=nm)
        if "RRL" in nm:
            axes[2].plot(t, act[:, c, 0], lw=1.3, color=COLOR[nm], label=nm)
    axes[0].set_ylabel("‖x − x_des‖ [m]", fontsize=12)
    axes[0].set_title("tracking error vs the reference", fontsize=13)
    axes[1].set_ylabel("‖x_nom − x_true‖ [m]", fontsize=12)
    axes[1].set_title("twin discrepancy — what the residual removes", fontsize=13)
    axes[2].set_ylabel("normalized thrust residual", fontsize=12)
    axes[2].set_title("residual thrust command", fontsize=13)
    axes[2].set_xlabel("time [s]", fontsize=12)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, ncol=2)
        ax.tick_params(labelsize=10)
    fig.suptitle(header, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_case_actuators(keep, c, dt, header, path, zoom_s=1.0, arms=CASE_ORDER):
    n = keep[arms[0]][2].shape[0]
    nz = min(int(zoom_s / dt), n)
    z0 = max(0, n // 2 - nz // 2)
    ylo = min(float(keep[nm][2][:, c].min()) for nm in arms)
    yhi = max(float(keep[nm][2][:, c].max()) for nm in arms)
    zlo = min(float(keep[nm][2][z0:z0 + nz, c].min()) for nm in arms)
    zhi = max(float(keep[nm][2][z0:z0 + nz, c].max()) for nm in arms)
    pad, zpad = 0.05 * (yhi - ylo + 1e-6), 0.08 * (zhi - zlo + 1e-9)

    fig, axes = plt.subplots(len(arms), 2, figsize=(15, 2.6 * len(arms)))
    t = np.arange(n) * dt
    for i, nm in enumerate(arms):
        motor = keep[nm][2]
        ax = axes[i, 0]
        for r in range(4):
            ax.plot(t, motor[:, c, r], lw=0.6, color=ROTOR_COLORS[r])
        ax.axvspan(t[z0], t[z0 + nz - 1], color="grey", alpha=0.15)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_ylabel(f"{nm}\nrotor [N]", fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)
        if i == 0:
            ax.set_title("full episode", fontsize=13)
        ax = axes[i, 1]
        for r in range(4):
            ax.plot(t[z0:z0 + nz], motor[z0:z0 + nz, c, r], lw=1.0,
                    marker=".", ms=1.8, color=ROTOR_COLORS[r])
        ax.set_ylim(zlo - zpad, zhi + zpad)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)
        if i == 0:
            ax.set_title(f"{zoom_s:g} s zoom (shared scale)", fontsize=13)
    axes[-1, 0].set_xlabel("time [s]", fontsize=11)
    axes[-1, 1].set_xlabel("time [s]", fontsize=11)
    handles = [plt.Line2D([], [], color=cc, lw=2) for cc in ROTOR_COLORS]
    fig.legend(handles, [f"rotor {r}" for r in range(4)], loc="lower center",
               ncol=4, fontsize=12, frameon=True)
    fig.suptitle(header, fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_full_set(root, keep, draw, M, stats, dt, arms, episodes, eval_seed):
    """Write every summary figure, case figure, actuator figure and CSV."""
    for sub in ("01_summary", "02_cases", "03_actuators"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    gc = critical_gain(dt)
    g = draw["kf"] * draw["arm"] / draw["k"]
    fmag = np.linalg.norm(draw["force"], axis=-1)
    rng = np.random.default_rng(eval_seed)
    names = [nm for nm in CASE_ORDER if nm in M]

    # ---------------- per-episode csv ----------------
    with (root / "per_episode.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "arm", "k", "force", "kf", "arm_scale", "g",
                    "inside_margin", "pos_rmse", "steady", "sat_pct",
                    "roughness", "motor_hf_N", "effort"])
        for e in range(episodes):
            for nm in names:
                m = M[nm]
                w.writerow([e, nm, f"{draw['k'][e]:.6f}", f"{fmag[e]:.6f}",
                            f"{draw['kf'][e]:.6f}", f"{draw['arm'][e]:.6f}",
                            f"{g[e]:.6f}", int(g[e] < gc),
                            f"{m['rmse'][e]:.6f}", f"{m['steady'][e]:.6f}",
                            f"{m['sat_pct'][e]:.6f}", f"{m['roughness'][e]:.6f}",
                            f"{m['motor_hf'][e]:.6f}", f"{m['effort'][e]:.6f}"])

    # ---------------- paired tests ----------------
    pairs = [("PD", "PD + RRL + LPF"), ("PID", "PID + RRL + LPF"),
             ("PD", "PD + RRL"), ("PID", "PID + RRL"),
             ("PD + RRL", "PD + RRL + LPF"), ("PID + RRL", "PID + RRL + LPF"),
             ("PD + RRL + LPF", "PID + RRL + LPF")]
    prows = []
    for a, b in pairs:
        if a not in M or b not in M:
            continue
        x, y = M[a]["rmse"], M[b]["rmse"]
        prows.append({"arm_a": a, "arm_b": b,
                      "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                      "median_a": float(np.median(x)),
                      "median_b": float(np.median(y)),
                      "delta_mean_pct": float(100 * (y.mean() - x.mean()) / x.mean()),
                      "b_wins_pct": float(100 * np.mean(y < x)),
                      "p_value": _paired_p(x, y, rng)})
    with (root / "paired_tests.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prows[0]))
        w.writeheader()
        w.writerows(prows)

    # ---------------- 01 summary ----------------
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x, [np.median(M[nm]["rmse"]) for nm in names],
           color=[COLOR[nm] for nm in names])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=11)
    ax.set_ylabel("Median position RMSE [m]", fontsize=12)
    ax.set_title(f"Tracking accuracy, {episodes} paired non-terminated episodes",
                 fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "01_summary" / "arm_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.boxplot([M[nm]["rmse"] for nm in names], tick_labels=names,
               showfliers=True, medianprops=dict(color="k", lw=2))
    ax.set_ylabel("Position RMSE [m]", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Per-episode distribution (log scale)", fontsize=13)
    ax.grid(axis="y", alpha=0.3, which="both")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=10)
    fig.tight_layout()
    fig.savefig(root / "01_summary" / "rmse_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for nm in names:
        pe = keep[nm][0]
        ax.plot(np.arange(pe.shape[0]) * dt, pe.mean(axis=1), lw=1.6,
                color=COLOR[nm], label=nm)
    ax.set_xlabel("time [s]", fontsize=12)
    ax.set_ylabel("mean ‖x − x_des‖ [m]", fontsize=12)
    ax.set_title(f"Tracking error over time, mean of {episodes} episodes",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "01_summary" / "tracking_error_vs_time.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, key, title in ((axes[0], "sat_pct", "Actuator saturation [%]"),
                           (axes[1], "roughness", "Input roughness RMS"),
                           (axes[2], "effort", "Residual command RMS")):
        ax.bar(x, [M[nm][key].mean() for nm in names],
               color=[COLOR[nm] for nm in names])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Control cost (means)", fontsize=13)
    fig.tight_layout()
    fig.savefig(root / "01_summary" / "termination_and_effort.png", dpi=150)
    plt.close(fig)

    cases = pick_cases(draw, g)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    sc = ax.scatter(g, fmag, c=M["PID + RRL + LPF"]["rmse"], cmap="viridis",
                    s=24, alpha=0.85)
    if gc < g.max():
        ax.axvline(gc, color="r", ls="--", lw=1.5)
        ax.axvspan(gc, g.max(), color="red", alpha=0.07)
    ax.text(min(gc, g.max() * 0.98), fmag.max(), f" g_crit = {gc:.2f}",
            color="r", fontsize=10, rotation=90, va="top")
    for lab, c in cases:
        ax.scatter([g[c]], [fmag[c]], s=180, facecolors="none",
                   edgecolors="k", lw=1.8)
        ax.annotate(lab, (g[c], fmag[c]), fontsize=9,
                    textcoords="offset points", xytext=(9, 5))
    plt.colorbar(sc, ax=ax, label="PID + RRL + LPF RMSE [m]")
    ax.set_xlabel("effective attitude gain  g = kf·arm / k", fontsize=12)
    ax.set_ylabel("|F_ext| [N]", fontsize=12)
    ax.set_title(f"Evaluation bank: {episodes} non-terminated episodes\n"
                 "circled points are plotted individually", fontsize=13)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "01_summary" / "uncertainty_coverage.png", dpi=150)
    plt.close(fig)

    # ---------------- 02 cases ----------------
    for lab, c in cases:
        hdr = case_header(lab, c, draw, g, gc)
        plot_case_traj(keep, M, c, hdr,
                       root / "02_cases" / f"case_{lab}_trajectory3d.png", names)
        plot_case_errors(keep, c, dt, hdr,
                         root / "02_cases" / f"case_{lab}_errors.png", names)
        plot_case_actuators(keep, c, dt, hdr,
                            root / "02_cases" / f"case_{lab}_actuators.png",
                            arms=names)

    # ---------------- 03 actuators ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, key, title in (
            (axes[0], "motor_hf", "Rotor command RMS above 10 Hz [N]"),
            (axes[1], "sat_pct", "Saturation [%]"),
            (axes[2], "roughness", "Input roughness RMS")):
        ax.bar(x, [M[nm][key].mean() for nm in names],
               color=[COLOR[nm] for nm in names])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Actuator behaviour (means)", fontsize=13)
    fig.tight_layout()
    fig.savefig(root / "03_actuators" / "actuator_summary.png", dpi=150)
    plt.close(fig)

    nfft = 1024
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for nm in names:
        motor = keep[nm][2]
        acc = None
        for e in range(episodes):
            xx = motor[:, e, 0] - motor[:, e, 0].mean()
            xx = np.pad(xx, (0, nfft - len(xx))) if len(xx) < nfft else xx[:nfft]
            pw = np.abs(np.fft.rfft(xx * np.hanning(len(xx)))) ** 2
            acc = pw if acc is None else acc + pw
        freqs = np.fft.rfftfreq(nfft, d=dt)
        ax.semilogy(freqs[1:], acc[1:] / episodes, lw=1.4, color=COLOR[nm],
                    label=nm)
    ax.axvline(10.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("frequency [Hz]", fontsize=12)
    ax.set_ylabel("mean power, rotor 0 command", fontsize=12)
    ax.set_title("Rotor command spectra, averaged over episodes", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(root / "03_actuators" / "spectra.png", dpi=150)
    plt.close(fig)

    return cases, prows
