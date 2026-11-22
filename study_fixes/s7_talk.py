"""Stage 7 -- figures and tables for the KSAS talk.

Scope: the method as submitted (paper reward tau_pos 0.5, residual authority
0.2, history 10, thrust LPF beta 0.2), retrained with every bug fix. No new
training is needed for sections 1-2; sections 3-4 appear once the group H / M
runs of stage 2 exist.

  1  Tracking: PD, PID and their residual-RL versions on the common bank of
     512 random conditions (3 training seeds); error over time, where the gain
     comes from (PID's integrator limit) and example flights.
  2  Thrust filter: the actuator commands of the condition where the filter
     helps most, the result table, and a diagram of the reason.
  3  History length (group H): 1, 10, 20 steps.
  4  Which channels to filter (group M): thrust only vs thrust + torques.
  B  Backup for questions: PID without its integrator limit, where the residual
     helps, the nominal vehicle.

One figure style throughout: stacked panels on a shared scale, one panel per
controller, the key number in the panel's top-right corner, a one-line takeaway
as the title. Tables are rendered like LaTeX booktabs tables (Computer Modern,
horizontal rules only) as PNG + PDF, and written as .tex (needs booktabs).
Writes <results>/06_talk/.
"""

import argparse
import re
import textwrap
from pathlib import Path

import numpy as np

from . import common as C
from .s2_train import HISTORIES, finished_run
from .s5_figures import (Arms, BASE_GRAY, INK, INK2, MUTED, SLOT, SURFACE, fly_positions,
                         plain_log, save, table_stats)
from . import s6_diagnose as D

import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch   # noqa: E402

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
                     # serif text to sit with a serif (신명조) deck; STIX ships with matplotlib
                     "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
SLIDE = False      # True: figures without their own title / footnote, tables without caption /
                   # notes (the slide carries those)

DT = 1.0 / C.PAPER_HZ
W0 = int(0.5 / DT)                       # steady flight starts after 0.5 s
BLUE, ORANGE, GREEN = SLOT[0], SLOT[1], SLOT[2]
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
VARIANT_GRAYS = ("#b9b8b3", "#5f5e5a")   # the two non-proposed variants of an ablation
HISTORY_TITLE = "Without history the residual loses much of its gain; 20 steps adds little"
CHANNEL_TITLE = "Filtering the thrust smooths the commands without changing the tracking"


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


def corner(ax, text):
    """The key number of a panel, top-right, as in the actuator figure."""
    ax.text(0.99, 0.94, text, transform=ax.transAxes, ha="right", va="top", fontsize=12,
            fontweight="bold", color=INK, linespacing=1.45)


def headline(fig, text, legend=None, ncol=None):
    """Takeaway title; an optional legend on its own line underneath."""
    if not SLIDE:
        fig.suptitle(text, x=0.01, ha="left", fontsize=15, fontweight="bold")
    if legend:
        h, l_ = legend
        fig.legend(h, l_, loc="upper left", bbox_to_anchor=(0.005, 0.995 if SLIDE else 0.945),
                   frameon=False, ncol=ncol or len(l_), fontsize=12 if SLIDE else 11.5,
                   handlelength=2.2)


def finish(fig, out, name, note, legend=False):
    if SLIDE:
        fig.tight_layout(rect=(0, 0, 1, 0.94 if legend else 1.0))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.905 if legend else 0.94))
        footer(fig, note)
    return save(fig, out, name)


def headroom(axes, lo=None):
    """Shared y-range with room at the top for the corner label."""
    y0, y1 = axes[0].get_ylim()
    for ax in axes[1:]:
        a, b = ax.get_ylim()
        y0, y1 = min(y0, a), max(y1, b)
    if lo is not None:
        y0 = lo
    axes[0].set_ylim(y0, y1 + 0.42 * (y1 - y0))


def line_handle(c, ls="-"):
    return Line2D([0], [0], color=c, lw=2.4, ls=ls)


def cm(v):
    return f"{100 * v:.2f}"


def pm_cm(v, s):
    return f"{100 * v:.2f} ± {100 * s:.2f}" if np.isfinite(s) and s > 0 else f"{100 * v:.2f}"


def _cond_text(m, e):
    return (f"mass & inertia ×{m['k'][e]:.2f} · rotor thrust ×{m['kf'][e]:.2f} · "
            f"arm ×{m['arm'][e]:.2f} · external force {m['force'][e]:.1f} N")


# ------------------------------------------------------------------ LaTeX-style tables
TEX_R, TEX_B = "cmr10", "cmb10"          # Computer Modern, shipped with matplotlib
TABLE_MATH = "cm"


def use_deck_font(font_dir):
    """Figures and tables in the deck font (KoPub돋움체 Medium, Bold for bold text).

    font_dir holds KoPubDotum-Medium.ttf and KoPubDotum-Bold.ttf (npm package font-kopub).
    Each weight is its own family in those files, so the Bold file is also registered as the
    bold face of the Medium family: fontweight="bold" then picks the real Bold.
    """
    global TEX_R, TEX_B, TABLE_MATH
    from matplotlib import font_manager as fm
    d = Path(font_dir)
    med, bold = d / "KoPubDotum-Medium.ttf", d / "KoPubDotum-Bold.ttf"
    for f in (med, bold):
        fm.fontManager.addfont(str(f))
    fam_m = fm.FontProperties(fname=str(med)).get_name()
    fam_b = fm.FontProperties(fname=str(bold)).get_name()
    fm.fontManager.ttflist.append(fm.FontEntry(fname=str(bold), name=fam_m, style="normal",
                                               variant="normal", weight=700, stretch="normal",
                                               size="scalable"))
    try:
        fm.fontManager._findfont_cached.cache_clear()
    except AttributeError:
        pass
    plt.rcParams.update({"font.family": [fam_m, "DejaVu Sans"], "mathtext.fontset": "stixsans"})
    TEX_R, TEX_B, TABLE_MATH = fam_m, fam_b, "stixsans"
# the Computer Modern TTFs use the TeX (OT1) layout: "{" is the en dash, "|" the em dash,
# "<" / ">" are inverted marks -> route those through math mode
_MATH = (("²", r"$^{2}$"), ("|", r"$|$"), (">", r"$>$"), ("<", r"$<$"), ("±", r"$\pm$"),
         ("→", r"$\rightarrow$"), ("×", r"$\times$"), ("τ", r"$\tau$"), ("β", r"$\beta$"),
         ("≈", r"$\approx$"), ("−", r"$-$"), ("–", "{"), ("·", r"$\cdot$"))
_TEX = (("\\", r"\textbackslash{}"), ("²", r"$^{2}$"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("_", r"\_"),
        ("|", r"$|$"), (">", r"$>$"), ("<", r"$<$"),
        ("±", r"$\pm$"), ("→", r"$\rightarrow$"), ("×", r"$\times$"), ("τ", r"$\tau$"),
        ("β", r"$\beta$"), ("≈", r"$\approx$"), ("−", r"$-$"), ("–", "--"), ("·", r"$\cdot$"))
_NEG = re.compile(r"(^|[\s(/])-(?=\d)")


_GREEK = (("Δ", r"$\Delta$"), ("Ω", r"$\Omega$"), ("Σ", r"$\Sigma$"), ("γ", r"$\gamma$"),
          ("ω", r"$\omega$"), ("‖", r"$\|$"), ("≤", r"$\leq$"), ("⁻⁴", r"$^{-4}$"),
          ("⁵", r"$^{5}$"), ("⁶", r"$^{6}$"))


def _segments(s, table):
    """Apply the replacements outside $...$ only (math passes through)."""
    parts = str(s).split("$")
    for i in range(0, len(parts), 2):
        for a, b in table:
            parts[i] = parts[i].replace(a, b)
        parts[i] = _NEG.sub(lambda m: m.group(1) + r"$-$", parts[i])
    return "$".join(parts)


def _famlist(fam):
    """Deck font: fall back to DejaVu Sans for the few glyphs KoPub lacks (en dash)."""
    return fam if fam.startswith("cm") else [fam, "DejaVu Sans"]


def _cm_text(s):
    if TEX_R.startswith("cm"):
        return _segments(s, _GREEK + _MATH)
    return _segments(s, _GREEK + tuple(m for m in _MATH if m[0] != "–"))   # CM-only remap


def _tex(s):
    return _segments(s, _TEX + _GREEK)


def latex_table(out, name, number, caption, cols, rows, groups=(), notes="", fs=12.0,
                bold_cols=()):
    """A booktabs-style table: PNG + PDF rendered in Computer Modern, plus the .tex.

    cols: [(header, "l" | "c")]; rows: [{"cells": [...], "bold": bool}] or "mid" for a
    \\midrule; groups: [(label, first col, last col)] spanning headers with \\cmidrule.
    """
    _mt = plt.rcParams["mathtext.fontset"]
    plt.rcParams["mathtext.fontset"] = TABLE_MATH           # LaTeX look inside tables
    pad, mar = 18.0, 14.0                                   # points
    tmp = plt.figure()
    rend = tmp.canvas.get_renderer()

    def width(s, fam):
        t = tmp.text(0, 0, _cm_text(s), fontsize=fs, family=_famlist(fam))
        w = t.get_window_extent(rend).width * 72.0 / tmp.dpi
        t.remove()
        return w

    W = [width(h, TEX_B if j in bold_cols else TEX_R) for j, (h, _) in enumerate(cols)]
    for r in rows:
        if r == "mid":
            continue
        for j, c in enumerate(r["cells"]):
            W[j] = max(W[j], width(c, TEX_B if (r.get("bold") or j in bold_cols) else TEX_R))
    colw = [w + pad for w in W]
    for lab, a, b in groups:
        need, have = width(lab, TEX_R) + pad, sum(colw[a:b + 1])
        if need > have:
            for j in range(a, b + 1):
                colw[j] += (need - have) / (b - a + 1)
    plt.close(tmp)
    tw = sum(colw)
    line = fs * 1.85
    per_char = fs * 0.47
    cap = textwrap.wrap(f"Table {number}: {caption}", width=max(30, int(tw / per_char)))
    nts = textwrap.wrap(notes, width=max(30, int(tw / (0.85 * per_char)))) if notes else []
    if SLIDE:
        cap, nts = [], []
    nbody = sum(1 for r in rows if r != "mid")
    nmid = sum(1 for r in rows if r == "mid")
    H = (2 * mar + len(cap) * fs * 1.4 + 10 + (line if groups else 0) + line + 5 +
         nbody * line + nmid * 5 + 8 + len(nts) * fs * 0.85 * 1.4)
    fig = plt.figure(figsize=((tw + 2 * mar) / 72.0, H / 72.0))
    fig.patch.set_facecolor("white")
    T = fig.dpi_scale_trans

    def put(x, y, s, fam=TEX_R, size=fs, ha="center"):
        fig.text(x / 72.0, y / 72.0, _cm_text(s), transform=T, fontsize=size, family=_famlist(fam),
                 ha=ha, va="baseline", color="black")

    def rule(y, x0=mar, x1=mar + tw, lw=0.6):
        fig.add_artist(Line2D([x0 / 72.0, x1 / 72.0], [y / 72.0] * 2, transform=T,
                              color="black", lw=lw, solid_capstyle="butt"))

    left = [mar + sum(colw[:j]) for j in range(len(cols))]
    y = H - mar
    for l_ in cap:
        y -= fs * 1.4
        put(mar + tw / 2, y + 0.25 * fs, l_)
    y -= 10
    rule(y, lw=1.1)                                         # \toprule
    if groups:
        y -= line
        for lab, a, b in groups:
            xa, xb = left[a], left[b] + colw[b]
            put((xa + xb) / 2, y + 0.42 * line, lab)
            rule(y + 0.12 * line, xa + 5, xb - 5, lw=0.5)  # \cmidrule(lr)
    y -= line
    for j, (h, al) in enumerate(cols):
        put(left[j] + (pad / 2 if al == "l" else colw[j] / 2), y + 0.32 * line, h,
            fam=TEX_B if j in bold_cols else TEX_R, ha="left" if al == "l" else "center")
    y -= 5
    rule(y, lw=0.6)                                         # \midrule
    for r in rows:
        if r == "mid":
            y -= 5
            rule(y, lw=0.5)
            continue
        y -= line
        for j, (c, (_, al)) in enumerate(zip(r["cells"], cols)):
            fam = TEX_B if (r.get("bold") or j in bold_cols) else TEX_R
            put(left[j] + (pad / 2 if al == "l" else colw[j] / 2), y + 0.32 * line, c, fam=fam,
                ha="left" if al == "l" else "center")
    y -= 8
    rule(y, lw=1.1)                                         # \bottomrule
    for l_ in nts:
        y -= fs * 0.85 * 1.4
        put(mar, y, l_, size=fs * 0.85, ha="left")
    fig.savefig(out / f"{name}.png", dpi=220, facecolor="white")
    fig.savefig(out / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    plt.rcParams["mathtext.fontset"] = _mt
    # the same table as LaTeX source
    L = ["% needs \\usepackage{booktabs}", r"\begin{table}[t]", r"\centering",
         rf"\caption{{{_tex(caption)}}}", rf"\label{{tab:{name}}}",
         rf"\begin{{tabular}}{{{''.join(al for _, al in cols)}}}", r"\toprule"]
    if groups:
        cells, cmr, j = [], [], 0
        for lab, a, b in groups:
            cells += [""] * (a - j)
            cells.append(rf"\multicolumn{{{b - a + 1}}}{{c}}{{{_tex(lab)}}}")
            cmr.append(rf"\cmidrule(lr){{{a + 1}-{b + 1}}}")
            j = b + 1
        cells += [""] * (len(cols) - j)
        L += [" & ".join(cells) + r" \\", " ".join(cmr)]
    L += [" & ".join(_tex(h) for h, _ in cols) + r" \\", r"\midrule"]
    for r in rows:
        if r == "mid":
            L.append(r"\midrule")
            continue
        cs = [_tex(c) for c in r["cells"]]
        if r.get("bold"):
            cs = [rf"\textbf{{{c}}}" if c else c for c in cs]
        else:
            cs = [rf"\textbf{{{c}}}" if (c and j in bold_cols) else c for j, c in enumerate(cs)]
        L.append(" & ".join(cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        L += [r"\par\smallskip", rf"{{\footnotesize {_tex(notes)}\par}}"]
    L.append(r"\end{table}")
    (out / f"{name}.tex").write_text("\n".join(L) + "\n")
    md = C.md_table([dict(zip([h for h, _ in cols], r["cells"])) for r in rows if r != "mid"],
                    [(h, h) for h, _ in cols])
    return f"{name}.png", md


# ------------------------------------------------------------------ 1. tracking
def t1_tracking(A, out):
    keys = ["PD_fixed", "PD_fixed_lpf", "PID_fixed", "PID_fixed_lpf", "PID_fixed_noclip"]
    rows, _ = table_stats(A, keys)
    R = {r["key"]: r for r in rows}
    body = []
    for grp in (("PD_fixed", "PD_fixed_lpf"), ("PID_fixed", "PID_fixed_lpf")):
        if body:
            body.append("mid")
        for k in grp:
            if k not in R:
                continue
            r = R[k]
            pol = A.spec[k]["policy"]
            body.append({"cells": [NAME[k], pm_cm(r["mean"], r["sd"]), cm(r["median"]),
                                   cm(r["p90"]),
                                   f"−{r['red_median']:.0f}% / −{r['red_p90']:.0f}%" if pol
                                   else "–", f"{r['sat_pct']:.2f}"], "bold": pol})
    fn, md = latex_table(
        out, "table1_tracking", 1, "Tracking error (position RMSE) over 512 random conditions.",
        [("Controller", "l"), ("Mean ± SD [cm]", "c"), ("Median [cm]", "c"),
         ("P90 [cm]", "c"), ("Reduction (median / P90)", "c"), ("Saturated [%]", "c")], body,
        notes=f"10 s 3-D figure-8; conditions: {BOX}. Statistics on the conditions every "
              f"controller completes. {RRL} = model-reference residual RL with the thrust "
              "filter, mean over 3 training seeds; reduction against the same controller "
              "alone. Mean ± SD over conditions.")
    return R, fn, ["## 1. Tracking", "", md, ""]


def f1_tracking(R, out):
    if not all(k in R for k in ("PD_fixed", "PD_fixed_lpf", "PID_fixed", "PID_fixed_lpf")):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, B in zip(axes, ("PD", "PID")):
        kb, kr = f"{B}_fixed", f"{B}_fixed_lpf"
        x, w = np.array([0.0, 1.0]), 0.36
        top = 100 * max(R[kb]["p90"], R[kr]["p90"])
        for i, (k, c, lab) in enumerate(((kb, BASE_GRAY, f"{B} alone"),
                                         (kr, FAMILY[B], f"{B} + {RRL}"))):
            v = [100 * R[k]["median"], 100 * R[k]["p90"]]
            sd = [100 * np.nan_to_num(R[k]["median_sd"]), 100 * np.nan_to_num(R[k]["p90_sd"])]
            xs = x + (i - 0.5) * w
            ax.bar(xs, v, w * 0.94, color=c, yerr=sd, label=lab,
                   error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
            for xi, vi, si in zip(xs, v, sd):
                ax.text(xi, vi + si + 0.02 * top, f"{vi:.1f}", ha="center", va="bottom",
                        fontsize=11.5)
        ax.set_ylim(0, top * 1.5)
        ax.set_xticks(x, ["typical (median)", "hard (P90)"])
        ax.set_ylabel("position RMSE [cm]")
        ax.set_title(f"{B} alone vs {B} + {RRL}", loc="left")
        ax.grid(axis="x", visible=False)
        corner(ax, f"median −{R[kr]['red_median']:.0f} %\nP90 −{R[kr]['red_p90']:.0f} %")
        ax.legend(loc="upper left", frameon=False)
    headline(fig, f"{RRL} cuts the tracking error of both controllers")
    return finish(fig, out, "fig1_tracking_summary",
                  f"10 s 3-D figure-8, 512 random conditions ({BOX}). {RRL}: mean over 3 "
                  "training seeds (whiskers: SD).")


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
    V = {k: np.concatenate(v) for k, v in data.items()}
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    for ax, B in zip(axes, ("PD", "PID")):
        kb, kr = f"{B}_fixed", f"{B}_fixed_lpf"
        bp = ax.boxplot([V[kr], V[kb]], vert=False, widths=0.6, patch_artist=True,
                        whis=(5, 95), medianprops={"color": INK, "lw": 2},
                        flierprops={"marker": "o", "ms": 3, "alpha": 0.35, "mec": "none"})
        for patch, fl, c in zip(bp["boxes"], bp["fliers"], (FAMILY[B], BASE_GRAY)):
            patch.set_facecolor(c)
            patch.set_edgecolor(SURFACE)
            fl.set_markerfacecolor(c)
        ax.set_yticks([1, 2], [f"{B} + {RRL}", f"{B} alone"])
        ax.set_xscale("log")
        plain_log(ax, "x")
        ax.set_title(B, loc="left")
        ax.grid(axis="y", visible=False)
        corner(ax, f"median {np.median(V[kb]):.1f} → {np.median(V[kr]):.1f} cm")
    axes[-1].set_xlim(right=max(v.max() for v in V.values()) * 2.6)
    axes[-1].set_xlabel("position RMSE per condition [cm]  (log scale)")
    headline(fig, f"{RRL} removes the long tail of bad conditions")
    return finish(fig, out, "fig2_error_distribution",
                  "Box: middle 50 % of the conditions; whiskers: 5–95 %; dots: the rest. "
                  f"512 conditions (× 3 training seeds for {RRL}).")


def _err_series(A, keys):
    """Position error over time for every surviving condition: key -> (T, N), ids, seeds."""
    seeds = A.seeds_of(keys)
    ids = np.arange(A.a.episodes)
    raw, base_cache = {k: [] for k in keys}, {}
    for s in seeds:
        tr = {}
        for k in keys:
            if not A.spec[k]["policy"]:
                if k not in base_cache:
                    base_cache[k] = fly_positions(A, k, s, list(ids))
                tr[k] = base_cache[k]
            else:
                tr[k] = fly_positions(A, k, s, list(ids))
        alive = np.ones(len(ids), bool)
        for k in keys:
            alive &= ~tr[k]["term"].astype(bool).any(axis=0)
        for k in keys:
            e = tr[k]["err"]
            e = np.linalg.norm(e, axis=-1) if e.ndim == 3 else e
            raw[k].append((100 * e[:, alive], ids[alive], np.full(alive.sum(), s)))
    return {k: tuple(np.concatenate([p[i] for p in v], axis=-1) for i in range(3))
            for k, v in raw.items() if v}


def f3_error_time(A, out, S):
    names, info = [], {}
    for B in ("PID", "PD"):
        kb, kr = f"{B}_fixed", f"{B}_fixed_lpf"
        if kb not in S or kr not in S:
            continue
        t = np.arange(S[kb][0].shape[0]) * DT
        late = t >= 5.0
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, sharey=True)
        for ax, k, title in ((axes[0], kb, f"{B} alone"), (axes[1], kr, f"{B} + {RRL}")):
            y50, y90 = np.percentile(S[k][0], 50, axis=1), np.percentile(S[k][0], 90, axis=1)
            ax.plot(t, y50, color=BLUE, lw=2.2)
            ax.plot(t, y90, color=ORANGE, lw=2.2)
            ax.set_title(title, loc="left")
            ax.set_ylabel("position error [cm]")
            info[(B, k)] = {"peak typical": float(y50.max()), "peak hard": float(y90.max()),
                            "after 5 s typical": float(y50[late].mean()),
                            "after 5 s hard": float(y90[late].mean())}
            corner(ax, f"typical: peak {y50.max():.1f} cm\nhard: {y90[late].mean():.1f} cm "
                       "after 5 s")
        headroom(axes, lo=0)
        axes[0].set_xlim(0, t[-1])
        axes[1].set_xlabel("time [s]")
        headline(fig, f"{RRL} cuts the error burst and the error that stays",
                 ([line_handle(BLUE), line_handle(ORANGE)],
                  ["typical condition (median)", "hard conditions (90th percentile)"]))
        names.append(finish(fig, out, f"fig3_error_over_time_{B.lower()}",
                            "At each instant, the median and the 90th percentile of the position "
                            "error over all surviving conditions (512 × 3 training seeds). The "
                            "uncertainty is switched on at t = 0.", legend=True))
    return names, info


def pid_integrator_clip(A):
    env = A.spec["PID_fixed"]["envs"][A.seeds[0]]
    tr = D.fly_instr(env, None, A.a.eval_seed, list(range(A.a.episodes)), None,
                     chunk=A.a.chunk, fields=("ctrl_ei", "terminated"))
    lim = float(getattr(env, "pid_ei_limit", 1.0))
    clip = (np.abs(tr["ctrl_ei"]) >= lim - 1e-6).any(axis=2)
    return clip.mean(axis=0)


def integrator_groups(A, frac):
    kb, kr = "PID_fixed", "PID_fixed_lpf"
    seeds = A.seeds_of([kb, kr])
    mb = A.m(kb, seeds[0])
    alive, rr = mb["terminated"] < 0.5, []
    for s in seeds:
        m = A.m(kr, s)
        rr.append(m["rmse_alive"])
        alive &= m["terminated"] < 0.5
    rb, rr = 100 * mb["rmse_alive"], 100 * np.mean(rr, axis=0)
    hit = frac > 0.25
    out = {}
    for tag, msk in (("within", ~hit & alive), ("limit", hit & alive)):
        out[tag] = {"n": int(msk.sum()), "pid": float(np.median(rb[msk])),
                    "rrl": float(np.median(rr[msk]))}
    return out, hit


def f4_integral(A, out, S, frac):
    kb, kr = "PID_fixed", "PID_fixed_lpf"
    if kb not in S or kr not in S:
        return None, None
    G, hit = integrator_groups(A, frac)
    t = np.arange(S[kb][0].shape[0]) * DT
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, sharey=True)
    for ax, k, title, who in ((axes[0], kb, "PID alone", "pid"),
                              (axes[1], kr, f"PID + {RRL}", "rrl")):
        E, ids, _ = S[k]
        for msk, c in ((~hit[ids], GREEN), (hit[ids], ORANGE)):
            ax.plot(t, np.median(E[:, msk], axis=1), color=c, lw=2.2)
        ax.set_title(title, loc="left")
        ax.set_ylabel("position error [cm]")
        corner(ax, f"within its limit: RMSE {G['within'][who]:.1f} cm\n"
                   f"at its limit: RMSE {G['limit'][who]:.1f} cm")
    headroom(axes, lo=0)
    axes[0].set_xlim(0, t[-1])
    axes[1].set_xlabel("time [s]")
    headline(fig, "The residual helps most where PID's integral action runs out",
             ([line_handle(GREEN), line_handle(ORANGE)],
              [f"PID's integral term stays within its limit ({G['within']['n']} conditions)",
               f"reaches its limit ({G['limit']['n']} conditions)"]))
    return finish(fig, out, "fig4_integral_limit",
                  "Median position error over time in each group of conditions. 'At its "
                  "limit' = PID's translational integral is clipped for more than a quarter of "
                  "the flight (the clip caps the integral force at 3.6 N per axis). RMSE = "
                  "median over the group.", legend=True), G


def f5_cases(A, out, frac):
    s = A.seeds[0]
    keys = ["PD_fixed", "PD_fixed_lpf", "PID_fixed", "PID_fixed_lpf"]
    if not all(A.has(k, s) for k in keys):
        return [], {}
    mm = {k: A.m(k, s) for k in keys}
    j = C.joint_survivors(mm)
    rb = mm["PID_fixed"]["rmse_alive"]
    med, p90 = np.median(rb[j]), np.percentile(rb[j], 90)
    picked = {"typical": int(np.argmin(np.where(j, np.abs(rb - med), np.inf))),
              "hard": int(np.argmin(np.where(j & (frac > 0.25), np.abs(rb - p90), np.inf)))}
    names = []
    comp = (("north", BLUE), ("east", ORANGE), ("up", GREEN))
    for tag, e in picked.items():
        tr = {k: fly_positions(A, k, s, [e]) for k in keys}
        word = "Typical" if tag == "typical" else "Hard"
        for B in ("PID", "PD"):
            kb, kr = f"{B}_fixed", f"{B}_fixed_lpf"
            t = np.arange(tr[kb]["x"].shape[0]) * DT
            r_b, r_r = 100 * mm[kb]["rmse_alive"][e], 100 * mm[kr]["rmse_alive"][e]
            # error components over time
            fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, sharey=True)
            for ax, k, title, rr in ((axes[0], kb, f"{B} alone", r_b),
                                     (axes[1], kr, f"{B} + {RRL}", r_r)):
                d = 100 * (tr[k]["x"][:, 0] - tr[k]["xd"][:, 0])
                d[:, 2] *= -1                                   # NED down -> up
                for i, (lab, c) in enumerate(comp):
                    ax.plot(t, d[:, i], color=c, lw=1.8)
                ax.axhline(0, color=MUTED, lw=0.8)
                ax.set_title(title, loc="left")
                ax.set_ylabel("position error [cm]")
                corner(ax, f"RMSE {rr:.1f} cm")
            headroom(axes)
            axes[0].set_xlim(0, t[-1])
            axes[1].set_xlabel("time [s]")
            headline(fig, f"{word} condition: {RRL} keeps the error {r_b / r_r:.0f}× smaller",
                     ([line_handle(c) for _, c in comp], [f"{lab} error" for lab, _ in comp]))
            names.append(finish(fig, out, f"fig5_case_{tag}_{B.lower()}",
                                f"{B} base, training seed {s}; condition: {_cond_text(mm[kb], e)}.",
                                legend=True))
            # the flown path
            fig = plt.figure(figsize=(12, 5.4))
            for i, (k, title, rr) in enumerate(((kb, f"{B} alone", r_b),
                                                (kr, f"{B} + {RRL}", r_r))):
                ax = fig.add_subplot(1, 2, i + 1, projection="3d")
                xd, x = tr[k]["xd"][:, 0], tr[k]["x"][:, 0]
                ax.plot(xd[:, 0], xd[:, 1], -xd[:, 2], color=INK, lw=1.3, ls="--")
                ax.plot(x[:, 0], x[:, 1], -x[:, 2], color=BLUE, lw=2)
                ax.set_xlabel("north [m]", labelpad=6)
                ax.set_ylabel("east [m]", labelpad=6)
                ax.set_zlabel("up [m]", labelpad=6)
                ax.set_title(title, loc="left")
                ax.text2D(0.98, 0.95, f"RMSE {rr:.1f} cm", transform=ax.transAxes, ha="right",
                          va="top", fontsize=12, fontweight="bold", color=INK)
            headline(fig, f"{word} condition: with {RRL} the flown path stays on the reference",
                     ([line_handle(INK, "--"), line_handle(BLUE)], ["reference", "flown"]))
            names.append(finish(fig, out, f"fig5_path_{tag}_{B.lower()}",
                                f"{B} base, training seed {s}; condition: {_cond_text(mm[kb], e)}.",
                                legend=True))
    return names, picked


# (row label, table_stats field, scale, format)
FILTER_ROWS = [("Tracking error, median [cm]", "median", 100, "{:.2f}"),
               ("Tracking error, P90 [cm]", "p90", 100, "{:.2f}"),
               "mid",
               ("Input roughness", "roughness", 1, "{:.3f}"),
               ("Residual-thrust rate [N/s]", "res_f_slew", 1, "{:.2f}"),
               ("Rotor jitter above 10 Hz [N]", "motor_hf", 1, "{:.3f}"),
               ("Rotor jitter, worst condition [N]", "motor_hf_max", 1, "{:.3f}"),
               ("Saturated steps [%]", "sat_pct", 1, "{:.2f}")]


def t2_filter(A, out):
    pairs = [(B, f"{B}_fixed_nolpf", f"{B}_fixed_lpf") for B in ("PD", "PID")]
    pairs = [p for p in pairs if A.has(p[1]) and A.has(p[2])]
    if not pairs:
        return None, None, ["## 2. Thrust filter", "", "_filter arms missing_", ""]
    rows, _ = table_stats(A, [k for _, kn, kf in pairs for k in (kn, kf)])
    R = {r["key"]: r for r in rows}
    body = []
    for spec in FILTER_ROWS:
        if spec == "mid":
            body.append("mid")
            continue
        lab, f, sc, fm = spec
        cells = [lab]
        for B, kn, kf in pairs:
            a_, b_ = sc * R[kn][f], sc * R[kf][f]
            cells += [fm.format(a_), fm.format(b_),
                      f"{100 * (b_ / a_ - 1):+.0f}%".replace("-", "−") if a_ > 0 else "–"]
        body.append({"cells": cells})
    cols = [("Metric", "l")] + [c for _ in pairs for c in
                                (("No filter", "c"), ("Thrust filter", "c"), ("Change", "c"))]
    groups = [(f"{B} + {RRL}", 1 + 3 * i, 3 + 3 * i) for i, (B, _, _) in enumerate(pairs)]
    fn, md = latex_table(
        out, "table2_thrust_filter", 2, "Effect of the thrust filter on the residual RL.",
        cols, body, groups=groups,
        notes="Authority 0.2, paper reward; each residual RL trained with and without the "
              "filter (3 training seeds each), 512 conditions. Input roughness: RMS change per "
              "control step of the total command (thrust and torques). Residual-thrust rate: "
              "mean absolute change per second. Rotor jitter: RMS of the rotor-command content "
              "above 10 Hz; worst condition: the largest of any condition.")
    return R, fn, ["## 2. Thrust filter (authority 0.2)", "", md, ""]


# ------------------------------------------------------------------ 2. thrust filter (figures)
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
    if SLIDE:                                   # the slide carries title and message
        fig.tight_layout()
    else:
        fig.suptitle("Why the thrust filter helps", x=0.01, ha="left", fontsize=18,
                     fontweight="bold")
        fig.tight_layout(rect=(0, 0.03, 1, 0.94))
        footer(fig, "A jump in the residual thrust tilts the thrust vector the geometric "
                    "controller commands; its rate of change becomes the desired body rate, which "
                    "the attitude loop turns into rotor commands. The resulting attitude and rate "
                    "errors are what the policy observes, so it answers its own jump.", y=0.02)
    fig.savefig(out / "fig8_filter_reason_diagram.svg", bbox_inches="tight")  # editable
    return save(fig, out, "fig8_filter_reason_diagram")


# ------------------------------------------------------------------ 3-4. ablations
def ablation(A, out, name, number, title, caption, sets, note, need, extra=None,
             fig_kind="rmse"):
    """sets: [(base, [(key, label, proposed?)])]; figure in the style of fig1 + Table."""
    present = [(B, [it for it in items if A.has(it[0])]) for B, items in sets if A.has(need(B))]
    present = [(B, it) for B, it in present if len(it) >= 2]
    if not present:
        return [f"## {title}", "", f"_not trained yet: {note}_", ""], [], None
    R = {}
    for B, items in present:
        rows, _ = table_stats(A, [k for k, _, _ in items])
        R.update({r["key"]: r for r in rows})
    md, names = ablation_from_stats(out, name, number, title, caption, present, R, note, extra,
                                    fig_kind)
    return md, names, R


def ablation_from_stats(out, name, number, title, caption, present, R, note, extra=None,
                        fig_kind="rmse"):
    """Table + figure from per-arm statistics (table_stats rows keyed by arm).

    extra: [(header, field, format)] columns shown instead of 'Mean ± SD' (e.g. smoothness).
    """
    body = []
    for B, items in present:
        if body:
            body.append("mid")
        for k, lab, prop in items:
            if k not in R:
                continue
            r = R[k]
            mid = ([fm.format(r[f]) for _, f, fm in extra] if extra
                   else [pm_cm(r["mean"], r["sd"])])
            body.append({"cells": [f"{B} + {RRL}", lab.replace("\n", " "),
                                   pm_cm(r["median"], r["median_sd"]),
                                   pm_cm(r["p90"], r["p90_sd"])] + mid +
                                  [f"{r['sat_pct']:.2f}"], "bold": prop})
    mid_cols = [(h, "c") for h, _, _ in extra] if extra else [("Mean ± SD [cm]", "c")]
    tfn, md = latex_table(out, f"table{number}_{name}", number, caption,
                          [("Controller", "l"), ("Variant", "l"), ("Median [cm]", "c"),
                           ("P90 [cm]", "c")] + mid_cols + [("Saturated [%]", "c")],
                          body, notes=note)
    if fig_kind == "smooth":
        ffn = smoothness_figure(out, f"fig_{name}", title, present, R, note)
        return [f"## {title}", "", md, ""], [ffn, tfn]
    fig, axes = plt.subplots(1, len(present), figsize=(5.8 * len(present), 4.9), squeeze=False)
    for ax, (B, items) in zip(axes[0], present):
        items = [it for it in items if it[0] in R]
        x, w = np.array([0.0, 1.0]), 0.8 / len(items)
        top = 100 * max(R[k]["p90"] for k, _, _ in items)
        grays = iter(VARIANT_GRAYS)
        for i, (k, lab, prop) in enumerate(items):
            c = FAMILY[B] if prop else next(grays)
            v = [100 * R[k]["median"], 100 * R[k]["p90"]]
            sd = [100 * np.nan_to_num(R[k]["median_sd"]), 100 * np.nan_to_num(R[k]["p90_sd"])]
            xs = x + (i - (len(items) - 1) / 2) * w
            ax.bar(xs, v, w * 0.92, color=c, yerr=sd, label=lab.replace("\n", " "),
                   error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
            for xi, vi, si in zip(xs, v, sd):
                ax.text(xi, vi + si + 0.02 * top, f"{vi:.1f}", ha="center", va="bottom",
                        fontsize=11)
        ax.set_ylim(0, top * 1.55)
        ax.set_xticks(x, ["typical (median)", "hard (P90)"])
        ax.set_ylabel("position RMSE [cm]")
        ax.set_title(f"{B} + {RRL}", loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    headline(fig, title)
    ffn = finish(fig, out, f"fig_{name}", note)
    return [f"## {title}", "", md, ""], [ffn, tfn]


SMOOTH_METRICS = [("input\nroughness", "roughness"), ("residual-thrust\nrate", "res_f_slew"),
                  ("rotor jitter\n(> 10 Hz)", "motor_hf"), ("rotor jitter,\nworst condition",
                                                         "motor_hf_max"),
                  ("tracking error\n(median)", "median")]


def smoothness_figure(out, name, title, present, R, note):
    """Each metric as % of the first (reference) variant: smaller = smoother."""
    fig, axes = plt.subplots(1, len(present), figsize=(6.6 * len(present), 5.0), squeeze=False,
                             sharey=True)
    ymax = 140.0
    for ax, (B, items) in zip(axes[0], present):
        items = [it for it in items if it[0] in R]
        ref, others = items[0], items[1:]
        x = np.arange(len(SMOOTH_METRICS), dtype=float)
        w = 0.8 / max(1, len(others))
        grays = iter(VARIANT_GRAYS[1:])
        for i, (k, lab, prop) in enumerate(others):
            c = FAMILY[B] if prop else next(grays)
            v = np.array([100 * R[k][f] / R[ref[0]][f] for _, f in SMOOTH_METRICS])
            sd = np.array([100 * np.nan_to_num(R[k].get(f + "_sd", 0.0)) / R[ref[0]][f]
                           for _, f in SMOOTH_METRICS])
            xs = x + (i - (len(others) - 1) / 2) * w
            ax.bar(xs, v, w * 0.92, color=c, yerr=sd, label=lab,
                   error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
            for xi, vi, si in zip(xs, v, sd):
                ax.text(xi, vi + si + 2, f"{vi - 100:+.0f} %".replace("-", "−"),
                        ha="center", va="bottom",
                        fontsize=10.5, fontweight="bold" if prop else None)
            ymax = max(ymax, float(np.max(v + sd)) + 18)
        ax.axhline(100, color=INK2, lw=1.2, ls="--")
        ax.text(-0.45, 101.5, f"{ref[1]} = 100 %", ha="left", va="bottom", fontsize=10.5,
                color=INK2)
        ax.set_xticks(x, [m for m, _ in SMOOTH_METRICS], fontsize=10.5)
        ax.set_title(f"{B} + {RRL}", loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    axes[0, 0].set_ylabel(f"relative to {present[0][1][0][1]} [%]")
    axes[0, 0].set_ylim(0, ymax)
    axes[0, 0].set_xlim(-0.55, len(SMOOTH_METRICS) - 0.45)
    headline(fig, title)
    return finish(fig, out, name, note + " Smaller = smoother; the last group shows that the "
                                         "tracking is unchanged.")


# ------------------------------------------------------------------ backup
def t_backup(A, out, R1, G):
    L, names = ["## B. Backup for questions", ""], []
    if R1 and all(k in R1 for k in ("PID_fixed", "PID_fixed_noclip", "PID_fixed_lpf")):
        body = [{"cells": [NAME[k], cm(R1[k]["median"]), cm(R1[k]["p90"]),
                           pm_cm(R1[k]["mean"], R1[k]["sd"])], "bold": k == "PID_fixed_lpf"}
                for k in ("PID_fixed", "PID_fixed_noclip", "PID_fixed_lpf")]
        fn, md = latex_table(out, "tableB1_integrator_limit", "B1",
                             "Is it only PID's integrator limit?",
                             [("Controller", "l"), ("Median [cm]", "c"), ("P90 [cm]", "c"),
                              ("Mean ± SD [cm]", "c")], body,
                             notes="The reference PID clips its translational integral "
                                   "(anti-windup), capping the integral force at 3.6 N per "
                                   "axis. Removing the clip helps PID's hard conditions, not "
                                   "its typical ones.")
        names.append(fn)
        L += [md, ""]
    if G:
        body = [{"cells": [lab, str(G[t]["n"]), f"{G[t]['pid']:.2f}", f"{G[t]['rrl']:.2f}",
                           f"−{100 * (1 - G[t]['rrl'] / G[t]['pid']):.0f}%"]}
                for t, lab in (("within", "Integral term within its limit"),
                               ("limit", "Integral term at its limit"))]
        fn, md = latex_table(out, "tableB2_where_it_helps", "B2",
                             "Where the residual helps (PID base).",
                             [("Conditions", "l"), ("Count", "c"), ("PID [cm]", "c"),
                              (f"PID + {RRL} [cm]", "c"), ("Reduction", "c")], body,
                             notes="Median position RMSE. 'At its limit': PID's translational "
                                   "integral is clipped for more than a quarter of the flight.")
        names.append(fn)
        L += [md, ""]
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
        rows.append({"cells": [NAME[k], " / ".join(f"{100 * x:.2f}" for x in v[:, 0]),
                               " / ".join(f"{x:.2f}" for x in v[:, 1])]})
    if rows:
        fn, md = latex_table(out, "tableB3_nominal_vehicle", "B3",
                             "The exactly nominal vehicle (no uncertainty).",
                             [("Controller", "l"), ("RMSE [cm]", "c"),
                              ("Mean |residual thrust| [N]", "c")], rows,
                             notes="One flight per training seed (seeds separated by '/'). The "
                                   "policy was never trained on the exact nominal vehicle (a "
                                   "single point of the box) and does not switch itself off "
                                   "there.")
        names.append(fn)
        L += [md, ""]
    return L, names


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
         f"({BOX}); 3 training seeds. Figures: PNG (200 dpi) + PDF; tables: PNG + PDF + .tex "
         "(booktabs).", ""]
    figs, R1, G = [], None, None
    if "1" in parts or "B" in parts:
        R1, t1, md = t1_tracking(A, out)
        frac = pid_integrator_clip(A)
    if "1" in parts:
        L += md
        figs += [t1, f1_tracking(R1, out), f2_distribution(A, out)]
        S = _err_series(A, ["PD_fixed", "PD_fixed_lpf", "PID_fixed", "PID_fixed_lpf"])
        names, info = f3_error_time(A, out, S)
        figs += names
        if info:
            L += ["Error over time [cm] (median = typical, P90 = hard):", "",
                  C.md_table([{"controller": NAME[k], **{kk: f"{vv:.2f}" for kk, vv in v.items()}}
                              for (B, k), v in info.items()],
                             [(c, c) for c in ("controller", "peak typical", "peak hard",
                                               "after 5 s typical", "after 5 s hard")]), ""]
        fn, G = f4_integral(A, out, S, frac)
        figs.append(fn)
        names, picked = f5_cases(A, out, frac)
        figs += names
        if picked:
            L += [f"Case figures: typical = bank condition {picked['typical']}, hard = "
                  f"{picked['hard']} (seed {A.seeds[0]}).", ""]
    if "2" in parts:
        R2, t2, md = t2_filter(A, out)
        L += md
        if R2 is not None:
            figs.append(t2)
            names, info = f6_filter_case(A, out)
            figs += names
            for B, v in info.items():
                L += [f"Example ({B}): seed {v['seed']}, condition {v['condition']} — the "
                      "condition where the filter helps most in steady flight; rotor jitter "
                      f"{v['no filter']['motor_hf']:.3f} N without vs "
                      f"{v['filter']['motor_hf']:.3f} N with the filter.", ""]
        figs.append(f8_reason_diagram(out))
    if "3" in parts:
        hs = HISTORIES
        md, names, _ = ablation(
            A, out, "history", 3, HISTORY_TITLE,
            "Effect of the observation history length.",
            [(B, [(f"{B}_h{hs[0]}", f"history {hs[0]} (current error only)", False),
                  (f"{B}_fixed_lpf", "history 10 (proposed)", True),
                  (f"{B}_h{hs[-1]}", f"history {hs[-1]}", False)]) for B in ("PD", "PID")],
            "History = how many past steps of the error against the nominal model, and of the "
            "policy's own past residual, the policy sees; 1 = only the current error. 3 "
            "training seeds per variant, 512 conditions.",
            need=lambda B: f"{B}_h{hs[0]}" if A.has(f"{B}_h{hs[0]}") else f"{B}_h{hs[-1]}")
        L += md
        figs += names
    if "4" in parts:
        md, names, _ = ablation(
            A, out, "filter_channels", 4, CHANNEL_TITLE,
            "Filtering the thrust only, or the thrust and the torques.",
            [(B, [(f"{B}_fixed_nolpf", "no filter", False),
                  (f"{B}_fixed_lpf", "thrust only (proposed)", True),
                  (f"{B}_lpfall", "thrust + torques", False)]) for B in ("PD", "PID")],
            "Same first-order filter (β 0.2) on the residual thrust only, or on the residual "
            "thrust and the three residual torques. 3 training seeds per variant, 512 "
            "conditions. Input roughness: RMS change per control step of the total command; "
            "rotor jitter: RMS of the rotor-command content above 10 Hz.",
            need=lambda B: f"{B}_lpfall",
            extra=[("Input roughness", "roughness", "{:.3f}"),
                   ("Residual-thrust rate [N/s]", "res_f_slew", "{:.2f}"),
                   ("Rotor jitter [N]", "motor_hf", "{:.3f}")],
            fig_kind="smooth")
        L += md
        figs += names
    if "B" in parts:
        if G is None:
            G = integrator_groups(A, frac)[0]
        md, names = t_backup(A, out, R1, G)
        L += md
        figs += names
    L += ["## Figures and tables", ""] + [f"![]({f})" for f in figs if f]
    (out / "TALK.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
