"""Stage 8b -- GIFs for the deck: 3-D flight with base vs residual thrust (fixed, chosen
conditions).

    gif0a_problem_pid       problem slide 1: PID only; payload +30 %, downdraft 3 N
    gif0b_problem_residual  problem slide 2: same flight with the residual (PID-only error ghosted)
    gif1_opener       PID only vs PID + residual RL; payload +30 %, downdraft 3 N
    gif2_uncertainty  PID + residual RL: no uncertainty vs condition 339
    gif3_downdraft    PID only vs PID + residual RL; 3 N downdraft, descents shaded

Each also as MP4 and as first / final frame PNG. Needs ffmpeg. Writes <results>/07_deck/gif/.
    python -m study_fixes.s8_gifs --results study_results [--parts problem,opener,qa1,qa2]
"""
import argparse
import subprocess
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402
from matplotlib.patches import Patch                                    # noqa: E402

from . import common as C                                               # noqa: E402
from .s2_train import finished_run                                      # noqa: E402
from .s8_deck import fly_fixed                                          # noqa: E402

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--results", default="study_results")
_p.add_argument("--seed", type=int, default=0, help="training seed of the policy shown")
_p.add_argument("--parts", default="problem,opener,qa1,qa2")
_p.add_argument("--font_dir", default="", help="folder with KoPubDotum-Medium/Bold.ttf")
_A, _ = _p.parse_known_args()
R, OUT = _A.results, Path(_A.results) / "07_deck" / "gif"
OUT.mkdir(parents=True, exist_ok=True)
FIELDS = ("x_true", "x_des", "R_true", "u_base", "residual")
DT = 1.0 / 200
INK, INK2, MUTED = "#1f1e1d", "#5f5e5a", "#a8a7a2"
GRAY, GRAY_FILL = "#7a7975", "#d3d2cd"
BLUE, ORANGE, WIND = "#2a78d6", "#eb6834", "#7fb2e5"
BG = "#fcfcfb"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 12})
if _A.font_dir:
    from study_fixes.s7_talk import use_deck_font
    use_deck_font(_A.font_dir)
    plt.rcParams["font.size"] = 12

ENV_B = C.paper_env(base_controller="pid", **C.all_fixes())
RUNS = {s: str(finished_run(R, f"A_pid_lpf_s{s}")) for s in (_A.seed,)}
ENV_R = {s: C.env_from_run(r) for s, r in RUNS.items()}

COND = {
    "opener": dict(k=1.3, force=(0.0, 0.0, 3.0)),
    "nominal": dict(),
    "c339": dict(k=1.069, kf=0.789, km=1.026, arm=0.950, force=(-2.535, 2.502, 0.817)),
    "down0": dict(), "down1": dict(force=(0, 0, 1.0)), "down2": dict(force=(0, 0, 2.0)),
    "down3": dict(force=(0, 0, 3.0)),
}
NAMES = list(COND)


def fly_all():
    L = [COND[n] for n in NAMES]
    tr = {"pid": fly_fixed(ENV_B, None, L, FIELDS)}
    for s in RUNS:
        tr["rrl"] = fly_fixed(ENV_R[s], RUNS[s], L, FIELDS)
    return tr


def one(tr, arm, name):
    i = NAMES.index(name)
    return {k: v[:, i] for k, v in tr[arm].items()}


# ------------------------------------------------------------------ drawing helpers
def ned(x):
    """NED -> plot coordinates (north, east, up)."""
    return x[..., 0], x[..., 1], -x[..., 2]


def quad(ax, x, Rm, color, arm=0.26, rotor=0.085):
    b1, b2 = Rm[:, 0], Rm[:, 1]
    c = np.asarray(x)
    th = np.linspace(0, 2 * np.pi, 24)
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        p = c + arm / np.sqrt(2) * (sx * b1 + sy * b2)
        ax.plot(*ned(np.stack([c, p])), color=INK, lw=2.0, zorder=6)
        ring = p[None] + rotor * (np.cos(th)[:, None] * b1 + np.sin(th)[:, None] * b2)
        ax.plot(*ned(ring), color=color, lw=2.0, zorder=7)
    ax.scatter(*[np.array([v]) for v in ned(c)], color=INK, s=18, zorder=8, depthshade=False)


def wind_arrows(ax, force, t, lims):
    f = np.asarray(force, float)
    n = np.linalg.norm(f)
    if n < 1e-6:
        return
    d = f / n                                                        # NED unit direction
    dn, de, du = d[0], d[1], -d[2]
    L = 0.42 if abs(du) > 0.7 else 0.75
    (n0, n1), (e0, e1), (u0, u1) = lims
    if abs(du) > 0.7:                                                # mostly vertical: rain of arrows
        pts = [(a, b) for a in np.linspace(n0 + 0.2, n1 - 0.2, 5) for b in (e0 + 0.15, e1 - 0.15)]
        span = (u1 - u0) * 0.42
        for j, (a, b) in enumerate(pts):
            ph = ((t * 0.9 + 0.37 * j) % 1.0)
            top = u1 - ph * span
            ax.quiver(a, b, top, dn * L, de * L, du * L, color=WIND, lw=2.2,
                      arrow_length_ratio=0.35, alpha=0.95)
    else:                                     # mostly horizontal: the compass shows it
        return
        for j, uu in enumerate(np.linspace(u0 + 0.35, u1 - 0.35, 4)):
            ph = ((t * 0.9 + 0.29 * j) % 1.0)
            a = n0 + 0.1 + ph * 0.5 * (n1 - n0) if dn >= 0 else n1 - 0.1 - ph * 0.5 * (n1 - n0)
            b = e0 + 0.1 if de >= 0 else e1 - 0.1
            ax.quiver(a, b, uu, dn * L, de * L, du * L, color=WIND, lw=2.2,
                      arrow_length_ratio=0.35, alpha=0.95)


def compass(fig, rect, force):
    """Top view of the wind: north up, east right; the down part as text."""
    f = np.asarray(force, float)
    cx = fig.add_axes(rect)
    cx.set_xlim(-1.35, 1.35); cx.set_ylim(-1.35, 1.35); cx.set_aspect("equal"); cx.axis("off")
    th = np.linspace(0, 2 * np.pi, 80)
    cx.plot(np.cos(th), np.sin(th), color=MUTED, lw=1.0)
    cx.text(0, 1.18, "N", ha="center", va="bottom", fontsize=13, color=INK2)
    cx.text(1.18, 0, "E", ha="left", va="center", fontsize=13, color=INK2)
    h = np.hypot(f[0], f[1])
    if h > 0.3:
        u = np.array([f[1], f[0]]) / h * min(1.0, 0.35 + h / 4.0)
        cx.annotate("", xy=(u[0], u[1]), xytext=(-u[0], -u[1]),
                    arrowprops=dict(arrowstyle="-|>", color=WIND, lw=3.0, mutation_scale=18))
    if f[2] > 0.3 and h <= 0.3:
        cx.plot([-0.32, 0.32], [-0.32, 0.32], color=WIND, lw=3)
        cx.plot([-0.32, 0.32], [0.32, -0.32], color=WIND, lw=3)
        cx.plot(0.45 * np.cos(th), 0.45 * np.sin(th), color=WIND, lw=2.5)
    lab = f"wind {np.linalg.norm(f):.1f} N"
    sub = []
    if h > 0.3:
        sub.append(f"{h:.1f} N horizontal")
    if abs(f[2]) > 0.3:
        sub.append(f"{abs(f[2]):.1f} N {'down' if f[2] > 0 else 'up'}")
    cx.text(0, -1.45, lab, ha="center", va="top", fontsize=15, fontweight="bold", color=INK)
    cx.text(0, -2.2, "\n".join(sub), ha="center", va="top", fontsize=12.5, color=INK2)


def style3d(ax, lims):
    (n0, n1), (e0, e1), (u0, u1) = lims
    ax.set_xlim(n0, n1); ax.set_ylim(e0, e1); ax.set_zlim(u0, u1)
    ax.set_box_aspect((n1 - n0, (e1 - e0), (u1 - u0)), zoom=1.0)
    ax.view_init(elev=27, azim=-38)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.96, 0.96, 0.95, 1.0))
        a.pane.set_edgecolor("#dddcd7")
        a._axinfo["grid"]["color"] = (0.86, 0.86, 0.84, 1)
        a._axinfo["grid"]["linewidth"] = 0.6
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-0.5, 0.5]); ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=11, colors=INK2, pad=0)
    ax.set_xlabel("north [m]", fontsize=13, color=INK2, labelpad=0)
    ax.set_ylabel("east [m]", fontsize=13, color=INK2, labelpad=-2)
    ax.set_zlabel("up [m]", fontsize=13, color=INK2, labelpad=-2)


LIMS = ((-1.3, 1.3), (-0.75, 0.75), (-1.35, 1.75))


def frame(fig, cols, i, cond_line, shade, ymax, show_wind):
    fig.clf()
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.962, cond_line, ha="center", va="center", fontsize=18, color=INK2)
    T = cols[0]["x"].shape[0]
    t = np.arange(1, T + 1) * DT
    n = i + 1
    for c, col in enumerate(cols):
        x0 = 0.02 + 0.5 * c
        fig.text(x0 + 0.235, 0.9, col["title"], ha="center", va="center", fontsize=22,
                 fontweight="bold", color=INK)
        # ---- 3-D flight
        ax = fig.add_axes([x0 + 0.03, 0.385, 0.45, 0.5], projection="3d")
        ax.set_facecolor(BG)
        style3d(ax, LIMS)
        xd, x = col["xd"], col["x"]
        ax.plot(*ned(xd), color=INK, lw=1.3, ls=(0, (4, 3)), alpha=0.75)
        ax.plot(*ned(x[:n]), color=col["color"], lw=2.6)
        if show_wind:
            wind_arrows(ax, col["force"], t[i], LIMS)
        if show_wind and np.linalg.norm(col["force"]) > 0:
            compass(fig, [x0 + 0.005, 0.68, 0.085, 0.15], col["force"])
        ax.plot(*ned(np.stack([x[i], xd[i]])), color=INK, lw=1.0, alpha=0.6)
        ax.scatter(*[np.array([v]) for v in ned(xd[i])], s=55, facecolors="white",
                   edgecolors=INK, linewidths=1.6, depthshade=False, zorder=9)
        quad(ax, x[i], col["R"][i], col["color"])
        err = 100 * np.linalg.norm(x[i] - xd[i])
        rm = 100 * np.sqrt(np.mean(np.sum((x[:n] - xd[:n]) ** 2, 1)))
        fig.text(x0 + 0.012, 0.565, "position error", ha="left", va="top", fontsize=13,
                 color=INK2)
        fig.text(x0 + 0.012, 0.535, f"{err:4.1f} cm", ha="left", va="top", fontsize=24,
                 fontweight="bold", color=INK, family="DejaVu Sans Mono")
        fig.text(x0 + 0.012, 0.47, f"RMSE so far {rm:.1f} cm", ha="left", va="top",
                 fontsize=12.5, color=INK2)
        # ---- thrust: base + residual stacked
        bx = fig.add_axes([x0 + 0.05, 0.115, 0.41, 0.175])
        bx.set_facecolor(BG)
        fb = col["base"]
        fr = col["res"] if col["res"] is not None else np.zeros_like(fb)
        if shade:
            for a0, a1, lab in shade:
                bx.axvspan(a0, a1, color="#e6eef8", zorder=0, lw=0)
                bx.text((a0 + a1) / 2, ymax * 0.05, lab, ha="center", va="bottom", fontsize=13,
                        color=INK, zorder=7)
        bx.fill_between(t[:n], 0, fb[:n], color=GRAY_FILL, lw=0, zorder=2)
        bx.plot(t[:n], fb[:n], color=GRAY, lw=1.6, zorder=3)
        if col["res"] is not None:
            bx.fill_between(t[:n], fb[:n], fb[:n] + fr[:n], color=ORANGE, alpha=0.9, lw=0,
                            zorder=4)
        bx.axvline(t[i], color=INK, lw=1.0, zorder=5)
        bx.set_xlim(0, 10); bx.set_ylim(0, ymax)
        for s in ("top", "right"):
            bx.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            bx.spines[s].set_color(MUTED)
        bx.tick_params(labelsize=13, colors=INK2)
        bx.grid(axis="y", color="#e4e3de", lw=0.7)
        bx.set_xlabel("time [s]", fontsize=14, color=INK2, labelpad=0)
        bx.set_ylabel("[N]", fontsize=14, color=INK2)
        lab = f"base {fb[i]:4.1f} N"
        if col["res"] is not None:
            lab += f"   residual {fr[i]:+4.1f} N"
        bx.text(1.0, 1.05, lab, transform=bx.transAxes, ha="right", va="bottom",
                fontsize=16, fontweight="bold", color=INK, family="DejaVu Sans Mono")
        bx.text(0.0, 1.05, "thrust command", transform=bx.transAxes, ha="left", va="bottom",
                fontsize=15, color=INK2)
    h = [Line2D([0], [0], color=INK, lw=1.3, ls=(0, (4, 3))),
         Line2D([0], [0], ls="none", marker="o", mfc="white", mec=INK, mew=1.6, ms=8),
         Patch(color=GRAY_FILL), Patch(color=ORANGE)]
    lab = ["reference path", "where it should be now", "base controller", "residual RL"]
    if show_wind and any(np.linalg.norm(col["force"]) > 0 for col in cols):
        h.append(Line2D([0], [0], ls="none", marker=r"$\downarrow$", color=WIND, ms=14))
        lab.append("wind (external force)")
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(h), frameon=False,
               fontsize=15, handlelength=2.0)


def render(name, cols, cond_line, shade=None, ymax=32, stride=10, hold=2.0, show_wind=True):
    tmp = OUT / f"_{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    fig = plt.figure(figsize=(16, 8.4), dpi=100)
    T = cols[0]["x"].shape[0]
    idx = list(range(20, T, stride)) + [T - 1]      # from 0.1 s: past the first-step spike
    for j, i in enumerate(idx):
        frame(fig, cols, i, cond_line, shade, ymax, show_wind)
        fig.savefig(tmp / f"f{j:04d}.png", dpi=100, facecolor=BG)
    fig.savefig(OUT / f"{name}_final.png", dpi=150, facecolor=BG)
    plt.close(fig)
    fps = 1.0 / (stride * DT)
    nhold = int(hold * fps)
    last = len(idx) - 1
    for k in range(nhold):
        shutil.copy(tmp / f"f{last:04d}.png", tmp / f"f{last + 1 + k:04d}.png")
    pat = str(tmp / "f%04d.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps}", "-i", pat,
                    "-vf", "scale=1600:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128"
                    ":stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
                    ":diff_mode=rectangle", "-loop", "0", str(OUT / f"{name}.gif")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps}", "-i", pat,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-vf", "scale=1920:-2", str(OUT / f"{name}.mp4")], check=True)
    shutil.copy(tmp / "f0000.png", OUT / f"{name}_first.png")
    shutil.rmtree(tmp)
    print(name, (OUT / f"{name}.gif").stat().st_size / 1e6, "MB")


def col(tr, arm, cond, title, color):
    d = one(tr, arm, cond)
    return {"x": d["x_true"], "xd": d["x_des"], "R": d["R_true"], "base": d["u_base"][:, 0],
            "res": None if arm == "pid" else d["residual"][:, 0], "force": COND[cond].get(
                "force", (0, 0, 0)), "title": title, "color": color}


# ---------------------------------------------------------------- single-flight GIFs (problem slides)
EMAX, YMAX1 = 22.0, 32.0          # shared scales so the two problem slides compare directly


def frame_single(fig, c, i, ghost=None):
    fig.clf()
    fig.patch.set_facecolor(BG)
    x, xd = c["x"], c["xd"]
    t = np.arange(1, x.shape[0] + 1) * DT
    n = i + 1
    err = 100 * np.linalg.norm(x - xd, axis=1)
    fig.text(0.255, 0.955, c["title"], ha="center", va="center", fontsize=22, fontweight="bold",
             color=INK)
    ax = fig.add_axes([0.08, 0.1, 0.42, 0.82], projection="3d")
    ax.set_facecolor(BG)
    style3d(ax, LIMS)
    ax.plot(*ned(xd), color=INK, lw=1.3, ls=(0, (4, 3)), alpha=0.75)
    ax.plot(*ned(x[:n]), color=c["color"], lw=2.8)
    wind_arrows(ax, c["force"], t[i], LIMS)
    ax.plot(*ned(np.stack([x[i], xd[i]])), color=INK, lw=1.0, alpha=0.6)
    ax.scatter(*[np.array([v]) for v in ned(xd[i])], s=60, facecolors="white", edgecolors=INK,
               linewidths=1.6, depthshade=False, zorder=9)
    quad(ax, x[i], c["R"][i], c["color"])
    compass(fig, [0.015, 0.66, 0.075, 0.2], c["force"])
    fig.text(0.02, 0.44, "position error", ha="left", va="top", fontsize=14, color=INK2)
    fig.text(0.02, 0.405, f"{err[i]:4.1f} cm", ha="left", va="top", fontsize=27,
             fontweight="bold", color=INK)
    rm = 100 * np.sqrt(np.mean(np.sum((x[:n] - xd[:n]) ** 2, 1)))
    fig.text(0.02, 0.31, f"RMSE so far {rm:.1f} cm", ha="left", va="top", fontsize=13.5,
             color=INK2)

    def style(a, ylab):
        a.set_facecolor(BG)
        for sname in ("top", "right"):
            a.spines[sname].set_visible(False)
        for sname in ("left", "bottom"):
            a.spines[sname].set_color(MUTED)
        a.tick_params(labelsize=13, colors=INK2)
        a.grid(axis="y", color="#e4e3de", lw=0.7)
        a.set_xlim(0, 10)
        a.set_ylabel(ylab, fontsize=14, color=INK2)
    ex = fig.add_axes([0.585, 0.575, 0.395, 0.29])
    style(ex, "[cm]")
    if ghost is not None:
        ex.plot(t, ghost, color=GRAY, lw=1.6, ls=(0, (3, 2)), alpha=0.9)
    ex.plot(t[:n], err[:n], color=c["color"], lw=2.4)
    ex.axvline(t[i], color=INK, lw=1.0)
    ex.set_ylim(0, EMAX)
    ex.set_xticklabels([])
    ex.text(0.0, 1.05, "position error", transform=ex.transAxes, fontsize=15, color=INK2,
            va="bottom")
    bx = fig.add_axes([0.585, 0.15, 0.395, 0.29])
    style(bx, "[N]")
    fb = c["base"]
    fr = c["res"] if c["res"] is not None else np.zeros_like(fb)
    bx.fill_between(t[:n], 0, fb[:n], color=GRAY_FILL, lw=0)
    bx.plot(t[:n], fb[:n], color=GRAY, lw=1.6)
    if c["res"] is not None:
        bx.fill_between(t[:n], fb[:n], fb[:n] + fr[:n], color=ORANGE, alpha=0.9, lw=0)
    bx.axvline(t[i], color=INK, lw=1.0)
    bx.set_ylim(0, YMAX1)
    bx.set_xlabel("time [s]", fontsize=14, color=INK2, labelpad=1)
    bx.text(0.0, 1.05, "thrust command", transform=bx.transAxes, fontsize=15, color=INK2,
            va="bottom")
    lab = f"base {fb[i]:4.1f} N"
    if c["res"] is not None:
        lab += f"    residual {fr[i]:+4.1f} N"
    bx.text(1.0, 1.05, lab, transform=bx.transAxes, ha="right", va="bottom", fontsize=16,
            fontweight="bold", color=INK)
    h = [Line2D([0], [0], color=INK, lw=1.3, ls=(0, (4, 3))),
         Line2D([0], [0], ls="none", marker="o", mfc="white", mec=INK, mew=1.6, ms=8),
         Patch(color=GRAY_FILL), Line2D([0], [0], ls="none", marker=r"$\downarrow$", color=WIND,
                                         ms=15)]
    labs = ["reference path", "where it should be now", "base controller", "wind"]
    if c["res"] is not None:
        h.insert(3, Patch(color=ORANGE)); labs.insert(3, "residual RL")
    if ghost is not None:
        h.append(Line2D([0], [0], color=GRAY, lw=1.6, ls=(0, (3, 2))))
        labs.append("PID only (previous slide)")
    fig.legend(h, labs, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=len(h),
               frameon=False, fontsize=14, handlelength=1.8, columnspacing=1.4)


def render_single(name, c, ghost=None, stride=10, hold=2.0):
    tmp = OUT / f"_{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    fig = plt.figure(figsize=(16, 6.8), dpi=100)
    T = c["x"].shape[0]
    idx = list(range(20, T, stride)) + [T - 1]
    for j, i in enumerate(idx):
        frame_single(fig, c, i, ghost)
        fig.savefig(tmp / f"f{j:04d}.png", dpi=100, facecolor=BG)
    fig.savefig(OUT / f"{name}_final.png", dpi=150, facecolor=BG)
    plt.close(fig)
    fps = 1.0 / (stride * DT)
    last = len(idx) - 1
    for k in range(int(hold * fps)):
        shutil.copy(tmp / f"f{last:04d}.png", tmp / f"f{last + 1 + k:04d}.png")
    pat = str(tmp / "f%04d.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps}", "-i", pat,
                    "-vf", "scale=1600:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128"
                    ":stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
                    ":diff_mode=rectangle", "-loop", "0", str(OUT / f"{name}.gif")], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps}", "-i", pat,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-vf", "scale=1920:-2", str(OUT / f"{name}.mp4")], check=True)
    shutil.copy(tmp / "f0000.png", OUT / f"{name}_first.png")
    shutil.rmtree(tmp)
    print(name, (OUT / f"{name}.gif").stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    tr = fly_all()
    np.savez(OUT / "flights.npz", **{f"{a}__{k}": v for a, d in tr.items() for k, v in d.items()})
    parts = _A.parts.split(",")
    if "problem" in parts:
        pid = col(tr, "pid", "opener", "Geometric PID only", GRAY)
        render_single("gif0a_problem_pid", pid)
        render_single("gif0b_problem_residual",
                      col(tr, "rrl", "opener", "Geometric PID + residual RL", BLUE),
                      ghost=100 * np.linalg.norm(pid["x"] - pid["xd"], axis=1))
    if "opener" in parts:
        render("gif1_opener", [col(tr, "pid", "opener", "Geometric PID only", GRAY),
                               col(tr, "rrl", "opener", "Geometric PID + residual RL", BLUE)],
               "Same vehicle, same wind:  payload +30 %  ·  downdraft 3 N")
    if "qa1" in parts:
        render("gif2_uncertainty", [col(tr, "rrl", "nominal", "No uncertainty", BLUE),
                                    col(tr, "rrl", "c339", "Large uncertainty", BLUE)],
               "PID + residual RL in both  ·  right: mass ×1.07, rotor thrust ×0.79, arm ×0.95, wind 3.7 N")
    if "qa2" in parts:
        render("gif3_downdraft", [col(tr, "pid", "down3", "Geometric PID only", GRAY),
                                  col(tr, "rrl", "down3", "Geometric PID + residual RL", BLUE)],
               "Downdraft 3 N on the nominal vehicle  ·  shaded: descending",
               shade=[(0, 2.5, "descending"), (7.5, 10, "descending")])
