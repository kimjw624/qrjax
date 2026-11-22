"""Stage 7 -- figures and tables for the KSAS talk.

Scope: the method as submitted (paper reward tau_pos 0.5, residual authority
0.2, history 10, thrust LPF beta 0.2), retrained with every bug fix. No new
training is needed for sections 1-2; sections 3-4 appear once the group H / M
runs of stage 2 exist.

  1  Tracking: PD, PID and their residual-RL versions on the common bank of
     512 random conditions (3 training seeds); where the gain comes from
     (error over time, PID's integrator limit) and two example flights.
  2  Thrust filter: the same residual RL trained with and without the filter
     -- the actuator commands of the condition where the filter helps most,
     the result table (roughness, thrust rate, rotor jitter, saturation,
     tracking) and a diagram of the reason.
  3  History length (group H): 1, 10, 20 steps.
  4  Which channels to filter (group M): thrust only vs thrust + torques.
  B  Backup for questions: PID without its integrator limit; the nominal
     vehicle (no uncertainty at all).

Figures are in English with plain-word labels for slides; the titles state the
takeaway. Writes <results>/06_talk/.
"""

import argparse
from pathlib import Path

import numpy as np

from . import common as C
from .s2_train import HISTORIES, finished_run
from .s5_figures import (Arms, BASE_GRAY, GRID, INK, INK2, MUTED, SLOT, SURFACE, fly_positions,
                         plain_log, save, table_stats)
from . import s6_diagnose as D

import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch   # noqa: E402

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11})

DT = 1.0 / C.PAPER_HZ
W0 = int(0.5 / DT)                       # steady flight starts after 0.5 s
BLUE, ORANGE = SLOT[0], SLOT[1]
LIGHT = {"PID": "#86b6ef", "PD": "#f4b48e"}            # "no filter" tints
FAMILY = {"PID": BLUE, "PD": ORANGE}
RRL = "Residual RL"
BOX = ("mass & inertia, rotor thrust and torque constants, arm length ×0.7–1.3; "
       "constant external force up to 3 N per axis")
NAME = {"PD_fixed": "PD", "PID_fixed": "PID",
        "PD_fixed_lpf": f"PD + {RRL}", "PID_fixed_lpf": f"PID + {RRL}",
        "PD_fixed_nolpf": f"PD + {RRL}, no filter", "PID_fixed_nolpf": f"PID + {RRL}, no filter",
        "PID_fixed_noclip": "PID, no integrator limit",
        "PD_lpfall": f"PD + {RRL}, filter on thrust + torques",
        "PID_lpfall": f"PID + {RRL}, filter on thrust + torques"}
for _b in ("PD", "PID"):
    for _h in HISTORIES:
        NAME[f"{_b}_h{_h}"] = f"{_b} + {RRL}, history {_h}"
NOMINAL = dict(k_min=1.0, k_max=1.0, external_force_max=0.0, motor_coeff_min=1.0,
               motor_coeff_max=1.0, moment_coeff_min=1.0, moment_coeff_max=1.0,
               arm_length_min=1.0, arm_length_max=1.0)
LIGHT_FIELDS = ("motor_cmd", "residual", "saturated", "terminated")


def extend(A):
    fx = C.all_fixes()
    A._base("PID_fixed_noclip", NAME["PID_fixed_noclip"], "pid",
            C.paper_env(base_controller="pid", pid_ei_limit=1e6, **fx))
    for base in ("pid", "pd"):
        B = base.upper()
        for h in HISTORIES:
            runs = {s: finished_run(A.a.results, f"H_{base}_lpf_h{h}_s{s}") for s in A.seeds}
            A._pol(f"{B}_h{h}", NAME[f"{B}_h{h}"], base, runs, {}, "fixed")
        runs = {s: finished_run(A.a.results, f"M_{base}_lpfall_s{s}") for s in A.seeds}
        A._pol(f"{B}_lpfall", NAME[f"{B}_lpfall"], base, runs, {}, "fixed")
    return A


def footer(fig, text, y=-0.02):
    fig.text(0.01, y, text, fontsize=9.5, color=INK2, ha="left", va="top", wrap=True)


def cm(v):
    return f"{100 * v:.2f}"


def pm_cm(v, s):
    return f"{100 * v:.2f} ± {100 * s:.2f}" if np.isfinite(s) and s > 0 else f"{100 * v:.2f}"


# ------------------------------------------------------------------ 1. tracking
def t1_tracking(A, out):
    keys = ["PD_fixed", "PD_fixed_lpf", "PID_fixed", "PID_fixed_lpf", "PID_fixed_noclip"]
    rows, _ = table_stats(A, keys)
    R = {r["key"]: r for r in rows}
    tab = []
    for k in keys:
        if k not in R:
            continue
        r = R[k]
        red = "–" if k in ("PD_fixed", "PID_fixed") else \
            f"{r['red_median']:.0f}% / {r['red_p90']:.0f}%"
        tab.append({"controller": NAME[k], "mean ± SD [cm]": pm_cm(r["mean"], r["sd"]),
                    "median [cm]": cm(r["median"]), "P90 [cm]": cm(r["p90"]),
                    "reduction vs own baseline (median / P90)": red,
                    "saturated steps [%]": f"{r['sat_pct']:.2f}",
                    "terminated [%]": f"{r['term_pct']:.2f}",
                    "training seeds": r["seeds"] if A.spec[k]["policy"] else "–"})
    C.write_csv(out / "table1_tracking.csv", tab)
    cols = list(tab[0].keys()) if tab else []
    md = C.md_table(tab, [(c, c) for c in cols]) if tab else "_no data_"
    return R, ["## 1. Tracking", "",
               "Position RMSE over the 10 s 3-D figure-8 on 512 random conditions "
               f"({BOX}). Statistics on the conditions every controller in the table "
               "survives; residual RL = mean ± SD over 3 training seeds (the baselines have "
               "no seed). 'PID, no integrator limit' is a reference: the same PID with its "
               "integral clip removed.", "", md, ""]


def f1_tracking(R, out):
    keys = ["PD_fixed", "PD_fixed_lpf", "PID_fixed", "PID_fixed_lpf"]
    if not all(k in R for k in keys):
        return None
    col = {"PD_fixed": BASE_GRAY, "PID_fixed": BASE_GRAY,
           "PD_fixed_lpf": ORANGE, "PID_fixed_lpf": BLUE}
    xt = ["PD", f"PD +\n{RRL}", "PID", f"PID +\n{RRL}"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (stat, title) in zip(axes, (("median", "Typical condition (median)"),
                                        ("p90", "Hard conditions (worst 10 %, P90)"))):
        x = np.array([0, 1, 2.4, 3.4])
        vals = [100 * R[k][stat] for k in keys]
        sds = [100 * R[k][stat + "_sd"] for k in keys]
        ax.bar(x, vals, 0.8, color=[col[k] for k in keys],
               yerr=[s if np.isfinite(s) else 0 for s in sds],
               error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
        top = max(vals)
        for xi, v, k in zip(x, vals, keys):
            ax.text(xi, v + 0.02 * top, f"{v:.1f} cm", ha="center", va="bottom",
                    fontsize=12, color=INK, fontweight="bold" if "lpf" in k else None)
        for b, rr in ((0, 1), (2, 3)):
            red = 100 * (1 - vals[rr] / vals[b])
            ax.text(x[rr], vals[rr] + 0.11 * top, f"−{red:.0f} %", ha="center", va="bottom",
                    fontsize=14, color=col[keys[rr]], fontweight="bold")
        ax.set_ylim(0, top * 1.18)
        ax.set_xticks(x, xt)
        ax.set_ylabel("position RMSE [cm]")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"{RRL} cuts the tracking error of both controllers", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    footer(fig, f"10 s 3-D figure-8, 512 random conditions ({BOX}). {RRL} = model-reference "
                "residual RL with the thrust filter, mean over 3 training seeds (whiskers: SD).")
    return save(fig, out, "fig1_tracking_summary")


def f2_distribution(A, out):
    keys = ["PD_fixed", "PD_fixed_lpf", "PID_fixed", "PID_fixed_lpf"]
    data = {k: [] for k in keys}
    for s in A.seeds_of(keys):
        mm = {k: A.m(k, s) for k in keys}
        j = C.joint_survivors(mm)
        for k in keys:
            data[k].append(100 * mm[k]["rmse_alive"][j])
    if not data[keys[0]]:
        return None
    vals = [np.concatenate(data[k]) for k in keys][::-1]
    labs = [NAME[k] for k in keys][::-1]
    cols = [BASE_GRAY, ORANGE, BASE_GRAY, BLUE][::-1]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bp = ax.boxplot(vals, vert=False, widths=0.55, patch_artist=True, whis=(5, 95),
                    showfliers=True, medianprops={"color": INK, "lw": 2},
                    flierprops={"marker": "o", "ms": 3, "alpha": 0.35, "mec": "none"})
    for patch, fl, c in zip(bp["boxes"], bp["fliers"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)
        patch.set_edgecolor(SURFACE)
        fl.set_markerfacecolor(c)
    for i, v in enumerate(vals):
        ax.text(np.median(v), i + 1 + 0.33, f"median {np.median(v):.1f} cm", ha="center",
                va="bottom", fontsize=11, color=INK, fontweight="bold")
    ax.set_xscale("log")
    plain_log(ax, "x")
    ax.set_yticks(range(1, len(labs) + 1), labs)
    ax.set_xlabel("position RMSE per condition [cm]  (log scale)")
    ax.set_ylim(0.5, len(vals) + 0.85)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{RRL} removes the long tail of bad conditions", loc="left", fontsize=15)
    footer(fig, "Box: middle 50 % of conditions; whiskers: 5–95 %; dots: the rest. "
                f"512 conditions × 3 training seeds for {RRL}.")
    fig.tight_layout()
    return save(fig, out, "fig2_error_distribution")


def _err_series(A, keys):
    seeds = A.seeds_of(keys)
    ids = list(range(A.a.episodes))
    ser, base_cache = {k: [] for k in keys}, {}
    for s in seeds:
        tr = {}
        for k in keys:
            if not A.spec[k]["policy"]:
                if k not in base_cache:
                    base_cache[k] = fly_positions(A, k, s, ids)
                tr[k] = base_cache[k]
            else:
                tr[k] = fly_positions(A, k, s, ids)
        alive = np.ones(len(ids), bool)
        for k in keys:
            alive &= ~tr[k]["term"].astype(bool).any(axis=0)
        for k in keys:
            e = tr[k]["err"]
            e = np.linalg.norm(e, axis=-1) if e.ndim == 3 else e
            ser[k].append(100 * e[:, alive])
    return {k: np.concatenate(v, axis=1) for k, v in ser.items() if v}


def f3_error_time(A, out, clip_frac=None):
    rows = (("PID", "PID_fixed", "PID_fixed_lpf"), ("PD", "PD_fixed", "PD_fixed_lpf"))
    keys = [k for _, b, r in rows for k in (b, r)]
    if not all(A.has(k) for k in keys):
        return None, {}
    S = _err_series(A, keys)
    t = np.arange(next(iter(S.values())).shape[0]) * DT
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    info = {}
    for i, (B, kb, kr) in enumerate(rows):
        for j, (q, lab) in enumerate(((50, "typical condition (median)"),
                                      (90, "hard conditions (90th percentile)"))):
            ax = axes[i, j]
            yb, yr = np.percentile(S[kb], q, axis=1), np.percentile(S[kr], q, axis=1)
            ax.plot(t, yb, color=BASE_GRAY, lw=2.4)
            ax.plot(t, yr, color=FAMILY[B], lw=2.4)
            top = yb.max() * 1.15
            lb, lr = yb[-1], yr[-1]
            if abs(lb - lr) < 0.12 * top:                    # keep the end labels apart
                mid = (lb + lr) / 2
                lb, lr = (mid + 0.07 * top, mid - 0.07 * top) if yb[-1] >= yr[-1] else \
                    (mid - 0.07 * top, mid + 0.07 * top)
            ax.text(t[-1] * 1.01, lb, B, color=BASE_GRAY, va="center", fontsize=11.5,
                    fontweight="bold")
            ax.text(t[-1] * 1.01, lr, f"{B} +\n{RRL}", color=FAMILY[B], va="center",
                    fontsize=11.5, fontweight="bold")
            ib, ir = int(np.argmax(yb)), int(np.argmax(yr))
            if B == "PID":
                ax.annotate(f"peak {yb[ib]:.1f} cm", (t[ib], yb[ib]), (t[ib] + 0.6, yb[ib]),
                            fontsize=10.5, color=INK2, va="center",
                            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
                ax.annotate(f"peak {yr[ir]:.1f} cm", (t[ir], yr[ir]),
                            (t[ir] + 0.9, yr[ir] * 0.45), fontsize=10.5, color=FAMILY[B],
                            va="center", arrowprops={"arrowstyle": "-", "color": FAMILY[B],
                                                     "lw": 0.8})
            late = t >= 5.0
            info[(B, q)] = {"peak_base": float(yb[ib]), "peak_rrl": float(yr[ir]),
                            "late_base": float(yb[late].mean()), "late_rrl": float(yr[late].mean())}
            ax.set_ylim(0, top)
            ax.set_xlim(0, t[-1])
            ax.set_title(f"{B}: {lab}", loc="left")
            if j == 0:
                ax.set_ylabel("position error [cm]")
            if i == 1:
                ax.set_xlabel("time [s]")
            if B == "PID" and q == 90:
                ax.text(6.1, (yb[late].mean() + yr[late].mean()) / 2,
                        "PID's integral term is at its limit\nin most of these conditions "
                        "→ the error stays", fontsize=10.5, color=INK2, ha="center",
                        va="center")
            if B == "PD" and q == 50:
                ax.text(5.0, (yb[late].mean() + yr[late].mean()) / 2,
                        "PD has no integral term:\nits error never goes away",
                        fontsize=10.5, color=INK2, ha="center", va="center")
    fig.suptitle(f"{RRL} cuts the error burst when the uncertainty hits, and the lasting "
                 "error in hard conditions", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.95, 0.95))
    footer(fig, "At each instant, the median (left) and the 90th percentile (right) of the "
                "position error over all surviving conditions. Every flight starts on the "
                "reference with the uncertainty switched on at t = 0.")
    return save(fig, out, "fig3_error_over_time"), info


def pid_integrator_clip(A):
    env = A.spec["PID_fixed"]["envs"][A.seeds[0]]
    tr = D.fly_instr(env, None, A.a.eval_seed, list(range(A.a.episodes)), None,
                     chunk=A.a.chunk, fields=("ctrl_ei", "terminated"))
    lim = float(getattr(env, "pid_ei_limit", 1.0))
    clip = (np.abs(tr["ctrl_ei"]) >= lim - 1e-6).any(axis=2)
    return clip.mean(axis=0)


def f4_where(A, out, frac):
    kb, kr = "PID_fixed", "PID_fixed_lpf"
    seeds = A.seeds_of([kb, kr])
    if not seeds:
        return None, []
    mb = A.m(kb, seeds[0])
    rr, alive = [], mb["terminated"] < 0.5
    for s in seeds:
        m = A.m(kr, s)
        rr.append(m["rmse_alive"])
        alive &= m["terminated"] < 0.5
    rb, rr = 100 * mb["rmse_alive"], 100 * np.mean(rr, axis=0)
    hit = frac > 0.25
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    lo = 0.5
    hi = float(np.percentile(np.r_[rb[alive], rr[alive]], 99.7)) * 1.4
    off = int(np.sum(alive & ((rb > hi) | (rr > hi))))
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls="--")
    xm = np.sqrt(lo * hi) * 1.4
    ax.text(xm, xm * 1.18, "no improvement", rotation=45, rotation_mode="anchor",
            color=MUTED, fontsize=10.5, ha="center", va="bottom")
    groups = [(~hit & alive, "#9e9d98", "PID's integral term stays within its limit",
               "within limit"),
              (hit & alive, ORANGE, "PID's integral term is at its limit > 25 % of the time",
               "at its limit")]
    rows = []
    for msk, c, lab, short in groups:
        ax.scatter(np.minimum(rb[msk], hi), np.minimum(rr[msk], hi), s=16, color=c, alpha=0.75,
                   edgecolor="none", label=f"{lab} ({int(msk.sum())})")
        rows.append({"group": lab, "short": short, "color": c, "conditions": int(msk.sum()),
                     "PID median [cm]": float(np.median(rb[msk])),
                     "PID + RRL median [cm]": float(np.median(rr[msk])),
                     "reduction [%]": float(100 * (1 - np.median(rr[msk]) / np.median(rb[msk])))})
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log(ax, "x")
    plain_log(ax, "y")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("PID alone: position RMSE [cm]")
    ax.set_ylabel(f"PID + {RRL}: position RMSE [cm]")
    ax.legend(loc="upper left", fontsize=10.5, markerscale=1.6)
    for i, r in enumerate(rows[::-1]):
        ax.text(0.98, 0.04 + 0.07 * i, f"{r['short']}: median {r['PID median [cm]']:.1f} → "
                f"{r['PID + RRL median [cm]']:.1f} cm", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=11.5, color=r["color"] if r["short"] != "within limit"
                else INK2, fontweight="bold")
    if off:
        ax.text(0.98, 0.98, f"{off} condition{'s' if off > 1 else ''} off the scale "
                "(drawn at the edge)", transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color=MUTED)
    ax.set_title("The residual helps most where PID's\nintegral action runs out", loc="left",
                 fontsize=14)
    footer(fig, f"One dot per condition (512, terminated ones removed); {RRL}: mean over 3 "
                "training seeds. Below the dashed line = the residual helps.")
    fig.tight_layout()
    for r in rows:
        r.pop("color", None)
        r.pop("short", None)
    C.write_csv(out / "table_integrator_groups.csv", rows)
    return save(fig, out, "fig4_where_it_helps"), rows


def _cond_text(m, e):
    return (f"mass & inertia ×{m['k'][e]:.2f} · rotor thrust ×{m['kf'][e]:.2f} · "
            f"arm ×{m['arm'][e]:.2f} · external force {m['force'][e]:.1f} N")


def f5_cases(A, out, frac):
    s = A.seeds[0]
    keys = ["PD_fixed", "PD_fixed_lpf", "PID_fixed", "PID_fixed_lpf"]
    if not all(A.has(k, s) for k in keys):
        return [], {}
    mm = {k: A.m(k, s) for k in keys}
    j = C.joint_survivors(mm)
    rb = mm["PID_fixed"]["rmse_alive"]
    med, p90 = np.median(rb[j]), np.percentile(rb[j], 90)
    typ = int(np.argmin(np.where(j, np.abs(rb - med), np.inf)))
    hard = int(np.argmin(np.where(j & (frac > 0.25), np.abs(rb - p90), np.inf)))
    names, picked = [], {"typical": typ, "hard": hard}
    for tag, e in picked.items():
        tr = {k: fly_positions(A, k, s, [e]) for k in keys}
        for B in ("PID", "PD"):
            kb, kr = f"{B}_fixed", f"{B}_fixed_lpf"
            fig = plt.figure(figsize=(12.5, 4.8))
            ax3 = fig.add_subplot(1, 2, 1, projection="3d")
            xd = tr[kb]["xd"][:, 0]
            ax3.plot(xd[:, 0], xd[:, 1], -xd[:, 2], color=INK, lw=1.3, ls="--", label="reference")
            for k, c, lab in ((kb, BASE_GRAY, B), (kr, FAMILY[B], f"{B} + {RRL}")):
                x = tr[k]["x"][:, 0]
                ax3.plot(x[:, 0], x[:, 1], -x[:, 2], color=c, lw=2, label=lab)
            ax3.set_xlabel("north [m]", labelpad=6)
            ax3.set_ylabel("east [m]", labelpad=6)
            ax3.set_zlabel("up [m]", labelpad=6)
            ax3.legend(loc="upper left", fontsize=10.5)
            ax = fig.add_subplot(1, 2, 2)
            t = np.arange(xd.shape[0]) * DT
            for k, c, lab in ((kb, BASE_GRAY, B), (kr, FAMILY[B], f"{B} + {RRL}")):
                err = tr[k]["err"][:, 0]
                err = np.linalg.norm(err, axis=-1) if err.ndim == 2 else err
                ax.plot(t, 100 * err, color=c, lw=2.2)
                ax.text(t[-1] * 1.01, 100 * err[-1], lab.replace(" + ", " +\n"), color=c,
                        fontsize=11, va="center", fontweight="bold")
            ax.set_xlim(0, t[-1])
            ax.set_ylim(bottom=0)
            ax.set_xlabel("time [s]")
            ax.set_ylabel("position error [cm]")
            r_b, r_r = 100 * mm[kb]["rmse_alive"][e], 100 * mm[kr]["rmse_alive"][e]
            ax.set_title(f"RMSE {r_b:.1f} cm → {r_r:.1f} cm", loc="left")
            word = "Typical" if tag == "typical" else "Hard"
            fig.suptitle(f"{word} condition: {_cond_text(mm[kb], e)}", x=0.01, ha="left",
                         fontsize=13.5, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 0.94, 0.94))
            names.append(save(fig, out, f"fig5_case_{tag}_{B.lower()}"))
    return names, picked


# ------------------------------------------------------------------ 2. thrust filter
_STEADY = {}


def steady(A, key):
    """Per-seed, per-condition rotor jitter in steady flight (t > 0.5 s)."""
    if key in _STEADY:
        return _STEADY[key]
    res = _STEADY[key] = {}
    sp = A.spec[key]
    for s in A.seeds:
        if not A.has(key, s):
            continue
        tr = D.fly_instr(sp["envs"][s], sp["runs"].get(s), A.a.eval_seed,
                         list(range(A.a.episodes)), None, chunk=A.a.chunk, fields=LIGHT_FIELDS)
        res[s] = {"alive": ~tr["terminated"].astype(bool).any(axis=0),
                  "jitter": D.hp(tr["motor_cmd"][W0:], DT).mean(axis=-1)}
    return res


# (row label, table_stats field, scale, format, "smaller is better")
FILTER_ROWS = [("tracking error, median [cm]", "median", 100, "{:.2f}"),
               ("tracking error, worst 10 % (P90) [cm]", "p90", 100, "{:.2f}"),
               ("input roughness (command change per step)", "roughness", 1, "{:.3f}"),
               ("residual-thrust rate of change [N/s]", "res_f_slew", 1, "{:.2f}"),
               ("rotor jitter above 10 Hz [N]", "motor_hf", 1, "{:.3f}"),
               ("rotor jitter, worst condition [N]", "motor_hf_max", 1, "{:.3f}"),
               ("saturated steps [%]", "sat_pct", 1, "{:.2f}")]


def t2_filter(A, out):
    pairs = [(B, f"{B}_fixed_nolpf", f"{B}_fixed_lpf") for B in ("PD", "PID")]
    pairs = [p for p in pairs if A.has(p[1]) and A.has(p[2])]
    if not pairs:
        return None, ["## 2. Thrust filter", "", "_filter arms missing_", ""]
    rows, _ = table_stats(A, [k for _, kn, kf in pairs for k in (kn, kf)])
    R = {r["key"]: r for r in rows}
    tab = []
    for lab, f, sc, fm in FILTER_ROWS:
        row = {"metric": lab}
        for B, kn, kf in pairs:
            a_, b_ = sc * R[kn][f], sc * R[kf][f]
            row[f"{B}: no filter"] = fm.format(a_)
            row[f"{B}: thrust filter"] = fm.format(b_)
            row[f"{B}: change"] = f"{100 * (b_ / a_ - 1):+.0f} %" if a_ > 0 else "–"
        tab.append(row)
    C.write_csv(out / "table2_thrust_filter.csv", tab)
    L = ["## 2. Thrust filter (authority 0.2)", "",
         "Residual RL trained with and without the thrust filter (3 training seeds each), "
         "on the 512 conditions; statistics on the conditions every arm survives, mean over "
         "seeds. Input roughness = RMS change per control step of the total command sent to "
         "the vehicle (thrust and the three torques). Residual-thrust rate of change = mean "
         "|change per step| / dt. Rotor jitter = RMS of the rotor-command content above 10 Hz "
         "over the whole flight; 'worst condition' = the largest of any condition.", "",
         C.md_table(tab, [(c, c) for c in tab[0]]), ""]
    return (R, pairs, tab), L


def f7_filter_table(T, out):
    R, pairs, tab = T
    fig = plt.figure(figsize=(12, 0.62 * (len(tab) + 2) + 1.4))
    ax = fig.add_axes((0.02, 0.07, 0.96, 0.78))
    ax.axis("off")
    cols = ["metric"] + [c for B, _, _ in pairs for c in
                         (f"{B}: no filter", f"{B}: thrust filter", f"{B}: change")]
    xs = [0.0, 0.43, 0.53, 0.63, 0.76, 0.86, 0.96]
    n = len(tab) + 1
    yrow = lambda i: 1 - (i + 0.5) / (n + 0.6)
    for j, (B, _, _) in enumerate(pairs):
        x0 = xs[1 + 3 * j] - 0.045
        ax.add_patch(plt.Rectangle((x0, yrow(-0.6) - 0.02), 0.29, 0.08, color=FAMILY[B],
                                   alpha=0.12, transform=ax.transAxes, lw=0))
        ax.text(x0 + 0.145, yrow(-0.6) + 0.02, f"{B} + {RRL}", transform=ax.transAxes,
                ha="center", va="center", fontsize=12.5, fontweight="bold", color=FAMILY[B])
    heads = ["", "no filter", "thrust filter", "change"] * 1 + ["no filter", "thrust filter",
                                                                "change"]
    for x, h in zip(xs, heads[:len(xs)]):
        ax.text(x, yrow(0), h, transform=ax.transAxes, ha="left" if x == 0 else "center",
                va="center", fontsize=11.5, color=INK2, fontweight="bold")
    ax.plot([0, 1.02], [yrow(0.5)] * 2, color=MUTED, lw=1, transform=ax.transAxes,
            clip_on=False)
    for i, row in enumerate(tab, start=1):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((-0.01, yrow(i) - 0.5 / (n + 0.6)), 1.04, 1 / (n + 0.6),
                                       color="#f1f0ec", transform=ax.transAxes, lw=0))
        for x, c in zip(xs, cols):
            v = row.get(c, "")
            kw = {}
            if c.endswith("change"):
                try:
                    pct = float(v.split()[0])
                except ValueError:
                    pct = 0.0
                track = row["metric"].startswith("tracking")
                kw = {"fontweight": "bold",
                      "color": INK2 if (track or abs(pct) < 5) else
                      ("#1a8a5a" if pct < 0 else "#c0392b")}
            ax.text(x, yrow(i), v, transform=ax.transAxes,
                    ha="left" if c == "metric" else "center", va="center", fontsize=11.5,
                    color=kw.pop("color", INK), **kw)
    fig.text(0.02, 0.97, "The thrust filter makes the commands smoother without changing the "
             "tracking", ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.025, "Authority 0.2, paper reward; 512 conditions, 3 training seeds per "
             "arm. Green = smoother with the filter; gray = essentially unchanged.",
             fontsize=9.5, color=INK2, ha="left", va="bottom")
    fig.savefig(out / "fig7_filter_table.png", dpi=200)
    fig.savefig(out / "fig7_filter_table.pdf")
    plt.close(fig)
    return "fig7_filter_table.png"


def _helping_case(A, B):
    kn, kf = f"{B}_fixed_nolpf", f"{B}_fixed_lpf"
    if not (A.has(kn) and A.has(kf)):
        return None
    best = None
    for s in A.seeds:
        if not (A.has(kn, s) and A.has(kf, s)):
            continue
        stn, stf = steady(A, kn)[s], steady(A, kf)[s]
        rn, rf = A.m(kn, s)["rmse_alive"], A.m(kf, s)["rmse_alive"]
        j = stn["alive"] & stf["alive"]
        normal = j & (rn <= 2 * np.median(rn[j])) & (rf <= 2 * np.median(rf[j]))
        gain = np.where(normal, stn["jitter"] - stf["jitter"], -np.inf)
        e = int(np.argmax(gain))
        if best is None or gain[e] > best[2]:
            best = (s, e, gain[e])
    return best


def f6_filter_case(A, out):
    names, info = [], {}
    for B in ("PID", "PD"):
        best = _helping_case(A, B)
        if best is None:
            continue
        s, e, _ = best
        kn, kf = f"{B}_fixed_nolpf", f"{B}_fixed_lpf"
        trs = {k: D.fly_instr(A.spec[k]["envs"][s], A.spec[k]["runs"][s], A.a.eval_seed, [e],
                              None, chunk=1) for k in (kn, kf)}
        mot = trs[kn]["motor_cmd"][:, 0]
        act = np.convolve(np.abs(np.diff(mot, 2, axis=0)).sum(axis=1), np.ones(40), "same")
        act[:2 * W0] = 0
        c = int(np.clip(np.argmax(act), 2 * W0 + 40, len(act) - 41))
        w = slice(c - 40, c + 40)                                 # 0.4 s window
        t = np.arange(mot.shape[0])[w] * DT
        lo = min(trs[k]["motor_cmd"][w, 0].min() for k in trs)
        hi = max(trs[k]["motor_cmd"][w, 0].max() for k in trs)
        pad = 0.15 * (hi - lo + 1e-6)
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
        mets = {}
        for ax, k, title, c_ in ((axes[0], kn, "trained without the filter", ORANGE),
                                 (axes[1], kf, "trained with the thrust filter (proposed)",
                                  FAMILY[B])):
            for r in range(4):
                ax.plot(t, trs[k]["motor_cmd"][w, 0, r], color=SLOT[r], lw=1.6)
            m = D.case_metrics(trs[k], A.spec[k]["envs"][s])
            mets[k] = m
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_ylabel("rotor commands [N]")
            ax.set_title(title, loc="left")
            ax.text(0.99, 0.94, f"rotor jitter {m['motor_hf']:.3f} N", transform=ax.transAxes,
                    ha="right", va="top", fontsize=12, fontweight="bold", color=INK)
        axes[1].set_xlabel("time [s]")
        fig.suptitle("Trained with the thrust filter, the rotor commands stay smooth", x=0.01,
                     ha="left", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        footer(fig, f"{B} + {RRL}, authority 0.2, training seed {s}; condition: "
                    f"{_cond_text(A.m(kn, s), e)}. A 0.4 s window in steady flight, the 4 "
                    "rotors. This is the condition where the filter helps most; averaged over "
                    "all conditions both are smooth in steady flight (see the table). Tracking "
                    f"{mets[kn]['rmse_cm']:.2f} vs {mets[kf]['rmse_cm']:.2f} cm.")
        names.append(save(fig, out, f"fig6_filter_case_{B.lower()}"))
        info[B] = {"seed": s, "condition": e, "no filter": mets[kn], "filter": mets[kf]}
    return names, info


def _wave(kind, n=40):
    x = np.linspace(0, 1, n)
    if kind == "jagged":
        rng = np.random.default_rng(3)
        return x, 0.85 * np.where(np.arange(n) % 2 == 0, 1, -1) * (0.6 + 0.4 * rng.random(n))
    return x, 0.35 * np.sin(2 * np.pi * 1.1 * x + 0.4)


def f8_reason_diagram(out):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    nodes = {"pol": (1.55, 3.6, "Residual RL\npolicy", False, 2.5),
             "f": (5.6, 6.35, "residual thrust", True, 2.7),
             "A": (9.35, 4.85, "thrust-direction\ncommand", False, 2.6),
             "w": (9.35, 2.35, "desired body rate", True, 2.6),
             "M": (5.6, 0.85, "rotor commands", True, 2.7)}
    edges = [("f", "A"), ("A", "w"), ("w", "M"), ("M", "pol")]
    labels = {("f", "A"): ("tilts the\nthrust vector", (8.55, 6.45)),
              ("A", "w"): ("its rate of\nchange", (10.75, 3.6)),
              ("M", "pol"): ("attitude & rate error\n= what the policy sees", (1.7, 1.45))}
    for ax, filt in zip(axes, (False, True)):
        ax.set_xlim(0, 11.2)
        ax.set_ylim(-0.3, 7.4)
        ax.axis("off")
        col = ORANGE if not filt else BLUE
        kind = "jagged" if not filt else "smooth"

        def box(k):
            x, y, txt, spark, wbox = nodes[k]
            return x, y, wbox, (1.45 if spark else 1.05)

        for k, (x, y, txt, spark, wbox) in nodes.items():
            _, _, wb, hb = box(k)
            fc = "#ecebe7" if k == "pol" else SURFACE
            ax.add_patch(FancyBboxPatch((x - wb / 2, y - hb / 2), wb, hb,
                                        boxstyle="round,pad=0.02,rounding_size=0.22", fc=fc,
                                        ec=INK2, lw=1.4, zorder=2))
            ax.text(x, y + (0.3 if spark else 0), txt, ha="center", va="center", fontsize=12.5,
                    color=INK, zorder=3, fontweight="bold" if k == "pol" else None)
            if spark:
                xx, yy = _wave(kind)
                ax.plot(x - 1.0 + 2.0 * xx, y - 0.33 + 0.25 * yy, color=col, lw=1.9, zorder=3)

        def arrow(p0, p1, rad=-0.2):
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=20, lw=2.4,
                                         color=col, connectionstyle=f"arc3,rad={rad}",
                                         zorder=1))
        # policy -> (filter) -> residual thrust
        p_top = (1.55, 3.6 + 0.55)
        f_left = (5.6 - 1.37, 6.35)
        if filt:
            fx, fy = 2.45, 5.75
            ax.add_patch(FancyBboxPatch((fx - 0.8, fy - 0.42), 1.6, 0.84,
                                        boxstyle="round,pad=0.02,rounding_size=0.2",
                                        fc=BLUE, ec=BLUE, zorder=4))
            ax.text(fx, fy, "thrust\nfilter", ha="center", va="center", fontsize=12.5,
                    color="white", fontweight="bold", zorder=5)
            arrow(p_top, (fx - 0.3, fy - 0.43), rad=-0.1)
            arrow((fx + 0.81, fy + 0.15), f_left, rad=-0.15)
        else:
            arrow(p_top, f_left, rad=-0.3)
        ends = {("f", "A"): ((5.6 + 1.37, 6.2), (9.35 - 0.6, 4.85 + 0.54)),
                ("A", "w"): ((9.35 + 0.2, 4.85 - 0.54), (9.35 + 0.2, 2.35 + 0.74)),
                ("w", "M"): ((9.35 - 0.6, 2.35 - 0.74), (5.6 + 1.37, 0.95)),
                ("M", "pol"): ((5.6 - 1.37, 0.75), (1.55 - 0.1, 3.6 - 0.54))}
        for e in edges:
            arrow(*ends[e], rad=-0.25 if e != ("A", "w") else -0.35)
            if e in labels:
                txt, (lx, ly) = labels[e]
                ax.text(lx, ly, txt, ha="center", va="center", fontsize=11, color=INK2,
                        style="italic")
        if not filt:
            ax.text(5.75, 3.6, "The policy reacts to\nthe jitter it caused\none step earlier\n"
                    "→ the loop keeps going", ha="center", va="center", fontsize=12.5,
                    color=ORANGE, fontweight="bold", linespacing=1.35)
            ax.set_title("Without the filter", loc="left", fontsize=15, color=ORANGE)
        else:
            ax.text(5.75, 3.6, "The thrust can only\nchange slowly\n→ nothing jumps "
                    "downstream\n→ the loop dies out", ha="center", va="center",
                    fontsize=12.5, color=BLUE, fontweight="bold", linespacing=1.35)
            ax.set_title("With the thrust filter", loc="left", fontsize=15, color=BLUE)
    fig.suptitle("Why the thrust filter helps", x=0.01, ha="left", fontsize=18,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    footer(fig, "A jump in the residual thrust tilts the thrust vector the geometric controller "
                "commands; its rate of change becomes the desired body rate, which the "
                "attitude loop turns into rotor commands. The resulting attitude and rate errors "
                "are what the policy observes, so it answers its own jump.", y=0.02)
    fig.savefig(out / "fig8_filter_reason_diagram.svg", bbox_inches="tight")  # editable
    return save(fig, out, "fig8_filter_reason_diagram")


# ------------------------------------------------------------------ 3-4. ablations
def t_ablation(A, out, name, title, sets, note, need):
    """sets: list of (base, [(key, label)]); ``need``: base -> key that must exist."""
    present = [(B, [(k, l) for k, l in items if A.has(k)]) for B, items in sets
               if A.has(need(B))]
    present = [(B, it) for B, it in present if len(it) >= 2]
    if not present:
        return [f"## {title}", "", f"_not trained yet: {note}_", ""], None
    tab, R = [], {}
    for B, items in present:
        rows, _ = table_stats(A, [k for k, _ in items])
        R.update({r["key"]: r for r in rows})
        for k, l in items:
            if k in R:
                r = R[k]
                tab.append({"base": B, "variant": l.replace("\n", " "), "seeds": r["seeds"],
                            "mean ± SD [cm]": pm_cm(r["mean"], r["sd"]),
                            "median [cm]": pm_cm(r["median"], r["median_sd"]),
                            "P90 [cm]": pm_cm(r["p90"], r["p90_sd"]),
                            "saturated steps [%]": f"{r['sat_pct']:.2f}",
                            "terminated [%]": f"{r['term_pct']:.2f}"})
    C.write_csv(out / f"{name}.csv", tab)
    fig, axes = plt.subplots(1, len(present), figsize=(5.6 * len(present), 4.3), squeeze=False)
    for ax, (B, items) in zip(axes[0], present):
        items = [(k, l) for k, l in items if k in R]
        x = np.arange(len(items))
        for i, (k, l) in enumerate(items):
            r = R[k]
            ax.bar(i - 0.2, 100 * r["median"], 0.38, color=FAMILY[B],
                   yerr=100 * r["median_sd"], error_kw={"ecolor": INK2, "capsize": 3})
            ax.bar(i + 0.2, 100 * r["p90"], 0.38, color=LIGHT[B],
                   yerr=100 * r["p90_sd"], error_kw={"ecolor": INK2, "capsize": 3})
            ax.text(i - 0.2, 100 * r["median"], f"{100 * r['median']:.1f}", ha="center",
                    va="bottom", fontsize=10.5)
            ax.text(i + 0.2, 100 * r["p90"], f"{100 * r['p90']:.1f}", ha="center", va="bottom",
                    fontsize=10.5, color=INK2)
        ax.set_xticks(x, [l for _, l in items])
        ax.set_ylabel("position RMSE [cm]")
        ax.set_title(f"{B} + {RRL}", loc="left")
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED),
                               plt.Rectangle((0, 0), 1, 1, color="#d6d5d1")],
                      labels=["typical (median)", "hard (P90)"], loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fn = save(fig, out, name)
    md = C.md_table(tab, [(c, c) for c in tab[0]])
    return [f"## {title}", "", note, "", md, "", f"![]({fn})", ""], fn


# ------------------------------------------------------------------ backup
def t_backup(A, out):
    L = ["## B. Backup for questions", ""]
    rows = []
    for k in ("PD_fixed", "PD_fixed_lpf", "PID_fixed", "PID_fixed_lpf"):
        if not A.has(k):
            continue
        sp = A.spec[k]
        vals = []
        for s in A.seeds:
            if s not in sp["envs"]:
                continue
            env = C.EnvConfig(**{**sp["envs"][s].__dict__, **NOMINAL})
            tr = D.fly_instr(env, sp["runs"].get(s), A.a.eval_seed, [0], None, chunk=1,
                             fields=("pos_err_desired", "residual"))
            e = tr["pos_err_desired"][:, 0]
            e = np.linalg.norm(e, axis=-1) if e.ndim == 2 else e
            vals.append((np.sqrt(np.mean(e ** 2)), np.mean(np.abs(tr["residual"][:, 0, 0]))))
            if not sp["policy"]:
                break
        v = np.asarray(vals)
        rows.append({"controller": NAME[k], "RMSE on the nominal vehicle [cm]":
                     " / ".join(f"{100 * x:.2f}" for x in v[:, 0]),
                     "mean |residual thrust| [N]": " / ".join(f"{x:.2f}" for x in v[:, 1])})
    C.write_csv(out / "table_backup_nominal.csv", rows)
    L += ["**The nominal vehicle (no uncertainty at all)**, one flight per training seed "
          "(seeds separated by '/'). The residual policy was never trained on the exact "
          "nominal vehicle (it is a single point of the box); it does not switch itself off "
          "there.", "", C.md_table(rows, [(c, c) for c in rows[0]]) if rows else "", ""]
    return L


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="study_results")
    p.add_argument("--paper_dir", default="auto")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--parts", default="1,2,3,4,B")
    a = p.parse_args()
    out = C.results_dir(Path(a.results) / "06_talk")
    A = extend(Arms(a))
    parts = set(a.parts.split(","))
    L = ["# 06 — Talk figures and tables", "",
         "The method as submitted (paper reward, residual authority 0.2, history 10, thrust "
         "filter β 0.2), retrained with every bug fix; 512 random conditions "
         f"({BOX}); 3 training seeds. Figures: PNG (200 dpi) + PDF.", ""]
    figs = []
    if "1" in parts:
        R1, md = t1_tracking(A, out)
        L += md
        figs += [f1_tracking(R1, out), f2_distribution(A, out)]
        frac = pid_integrator_clip(A)
        f3, info = f3_error_time(A, out, frac)
        figs.append(f3)
        if info:
            L += ["Error over time (all surviving conditions, pooled seeds), peak and mean "
                  "after 5 s [cm]:", "",
                  C.md_table([{"base": B, "statistic": "median" if q == 50 else "P90",
                               "peak, baseline": f"{v['peak_base']:.2f}",
                               "peak, + RRL": f"{v['peak_rrl']:.2f}",
                               "after 5 s, baseline": f"{v['late_base']:.2f}",
                               "after 5 s, + RRL": f"{v['late_rrl']:.2f}"}
                              for (B, q), v in info.items()],
                             [(c, c) for c in ("base", "statistic", "peak, baseline",
                                               "peak, + RRL", "after 5 s, baseline",
                                               "after 5 s, + RRL")]), ""]
        f4, grp = f4_where(A, out, frac)
        figs.append(f4)
        if grp:
            L += ["Where the residual helps (PID base): conditions split by how long PID's "
                  "integral term sits at its limit.", "",
                  C.md_table(grp, [(c, c) for c in grp[0]],
                             {"PID median [cm]": "{:.2f}", "PID + RRL median [cm]": "{:.2f}",
                              "reduction [%]": "{:.0f}"}), ""]
        names, picked = f5_cases(A, out, frac)
        figs += names
        if picked:
            L += [f"Case figures: typical = bank condition {picked['typical']}, hard = "
                  f"{picked['hard']} (seed {A.seeds[0]}).", ""]
    if "2" in parts:
        T2, md = t2_filter(A, out)
        L += md
        if T2 is not None:
            names, info = f6_filter_case(A, out)
            figs += names
            for B, v in info.items():
                L += [f"Example ({B}): seed {v['seed']}, condition {v['condition']} — the "
                      "condition where the filter helps most in steady flight; rotor jitter "
                      f"{v['no filter']['motor_hf']:.3f} N without vs "
                      f"{v['filter']['motor_hf']:.3f} N with the filter.", ""]
            figs.append(f7_filter_table(T2, out))
        figs.append(f8_reason_diagram(out))
    if "3" in parts:
        hs = HISTORIES
        md, fn = t_ablation(
            A, out, "fig9_history", "Does the history help?",
            [(B, [(f"{B}_h{hs[0]}", f"history {hs[0]}\n(current error only)"),
                  (f"{B}_fixed_lpf", "history 10\n(proposed)"),
                  (f"{B}_h{hs[-1]}", f"history {hs[-1]}")]) for B in ("PD", "PID")],
            "History = how many past steps of the error (vs. the nominal model) and of the "
            "policy's own past residual the policy sees; 1 = only the current error. Run "
            "`TRAIN_GROUPS=H` (12 runs).",
            need=lambda B: f"{B}_h{hs[0]}" if A.has(f"{B}_h{hs[0]}") else f"{B}_h{hs[-1]}")
        L += md
    if "4" in parts:
        md, fn = t_ablation(
            A, out, "fig10_filter_channels", "Which channels should be filtered?",
            [(B, [(f"{B}_fixed_nolpf", "no filter"), (f"{B}_fixed_lpf", "thrust only\n(proposed)"),
                  (f"{B}_lpfall", "thrust +\ntorques")]) for B in ("PD", "PID")],
            "Same filter (β 0.2) on the thrust only (proposed) or on the thrust and the three "
            "torques. Run `TRAIN_GROUPS=M` (6 runs).",
            need=lambda B: f"{B}_lpfall")
        L += md
    if "B" in parts:
        L += t_backup(A, out)
    L += ["## Figures", ""] + [f"![]({f})" for f in figs if f]
    (out / "TALK.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
