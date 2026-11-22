"""Stage 8 -- figures and tables for the KSAS talk, deck v4 (PID only).

The practice talk's feedback: PD is dropped; the three controllers compared are

    PID                    geometric SE(3) PID alone
    PID + Residual         + model-reference residual RL, trained without the filter
    PID + Residual + LPF   + the thrust low-pass filter (proposed)

Same style as stage 7 (stacked panels on a shared scale, the key number top-right,
a one-line takeaway as the title; LaTeX booktabs tables as PNG + PDF + .tex).
New here: the setup tables (controllers, domain randomization, training vs
evaluation), a small condition table for every example flight instead of a
'hard condition' label, the history ablation marked as run with the thrust LPF,
and the backup for the question 'does the residual only do what the base
cannot?' (residual thrust vs the thrust the model is missing, over the bank).
Writes <results>/07_deck/.
"""

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import common as C
from . import s6_diagnose as D
from . import s7_talk as T
from .s5_figures import (Arms, BASE_GRAY, INK, INK2, MUTED, SLOT, SURFACE, fly_positions,
                         plain_log, save, table_stats)

import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

latex_table, headline, corner, finish = T.latex_table, T.headline, T.corner, T.finish
headroom, line_handle, cm, pm_cm = T.headroom, T.line_handle, T.cm, T.pm_cm
DT, W0, BLUE, ORANGE, GREEN = T.DT, T.W0, T.BLUE, T.ORANGE, T.GREEN

KB, KN, KF = "PID_fixed", "PID_fixed_nolpf", "PID_fixed_lpf"
KEYS = (KB, KN, KF)
LAB = {KB: "PID", KN: "PID + Residual", KF: "PID + Residual + LPF"}
COL = {KB: BASE_GRAY, KN: "#8db6e8", KF: BLUE}
LIGHT_BLUE = COL[KN]
UNC = ("mass & inertia, rotor thrust and torque constants, arm length ×0.7–1.3; constant "
       "external force up to ±3 N per axis")
G = 9.81
M_NOM = 2.0


# ------------------------------------------------------------------ helpers
def bank_conditions(env_cfg, eval_seed, ids):
    """The exact vehicle and wind the bank draws for episodes ``ids``."""
    env = C.VecEnv(env_cfg, len(ids))
    ranges = env.broadcast_ranges(C.flat_ranges(env_cfg, force_freq=0.0, force_dc_prob=0.0))
    st = env._reset(C.bank_keys(eval_seed, list(ids)), ranges, jnp.zeros(len(ids), jnp.int32))
    return {"k": np.asarray(st.k), "force": np.asarray(st.external_force),
            "kf": np.asarray(st.mixer_true.kf_scale)[:, 0],
            "km": np.asarray(st.mixer_true.moment_scale)[:, 0],
            "arm": np.asarray(st.mixer_true.arm_scale)[:, 0]}


def needed_thrust(c):
    """Extra thrust command the vehicle needs at hover beyond the nominal model's [N]."""
    return (c["k"] * M_NOM * G + c["force"][..., 2]) / c["kf"] - M_NOM * G


def cond_rows(c, i):
    f = c["force"][i]
    return [{"cells": ["Mass & inertia", f"×{c['k'][i]:.2f}"]},
            {"cells": ["Rotor thrust constant", f"×{c['kf'][i]:.2f}"]},
            {"cells": ["Rotor torque constant", f"×{c['km'][i]:.2f}"]},
            {"cells": ["Arm length", f"×{c['arm'][i]:.2f}"]},
            {"cells": ["External force (N, E, D)",
                       f"({f[0]:+.1f}, {f[1]:+.1f}, {f[2]:+.1f}) N".replace("-", "−")]},
            {"cells": ["External force magnitude", f"{np.linalg.norm(f):.1f} N"]}]


def cond_table(out, name, number, caption, c, i, note=""):
    return latex_table(out, name, number, caption, [("Uncertainty", "l"), ("Value", "c")],
                       cond_rows(c, i), notes=note or "Scales are relative to the nominal "
                       "model; the force is constant over the flight (down = +D).", fs=13)


def bars(ax, R, keys, top=None):
    x, w = np.array([0.0, 1.0]), 0.8 / len(keys)
    top = top or 100 * max(R[k]["p90"] for k in keys)
    for i, k in enumerate(keys):
        v = [100 * R[k]["median"], 100 * R[k]["p90"]]
        sd = [100 * np.nan_to_num(R[k]["median_sd"]), 100 * np.nan_to_num(R[k]["p90_sd"])]
        xs = x + (i - (len(keys) - 1) / 2) * w
        ax.bar(xs, v, w * 0.92, color=COL[k], yerr=sd, label=LAB[k],
               error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
        for xi, vi, si in zip(xs, v, sd):
            ax.text(xi, vi + si + 0.02 * top, f"{vi:.1f}", ha="center", va="bottom",
                    fontsize=11.5, fontweight="bold" if k == KF else None)
    ax.set_ylim(0, top * 1.5)
    ax.set_xticks(x, ["typical condition (median)", "hard conditions (P90)"])
    ax.set_ylabel("position RMSE [cm]")
    ax.grid(axis="x", visible=False)


# ------------------------------------------------------------------ setup tables
def t_setup(out):
    names = []
    body = [{"cells": ["PID", "Geometric PID", "–", "–", "–"]},
            {"cells": ["PID + Residual", "Geometric PID", "SAC, authority 0.2", "–",
                       "3 seeds"]},
            {"cells": ["PID + Residual + LPF (proposed)", "Geometric PID",
                       "SAC, authority 0.2", "β = 0.2", "3 seeds"], "bold": True}]
    names.append(latex_table(
        out, "tableS1_controllers", "S1", "Controllers compared.",
        [("Controller", "l"), ("Base controller", "c"), ("Residual RL", "c"),
         ("Thrust LPF", "c"), ("Trained policies", "c")], body,
        notes="Geometric PID on SE(3) with the gains of the nominal 2 kg vehicle; the same "
              "base, gains and trajectory for all three. Authority 0.2: the residual can add "
              "at most 20 % of the maximum thrust and torques. Residual policies observe the "
              "error against the nominal twin over the last 10 steps.", fs=13)[0])
    body = [{"cells": ["Mass & inertia", "nominal × U(0.7, 1.3)"]},
            {"cells": ["Rotor thrust constant", "nominal × U(0.7, 1.3)"]},
            {"cells": ["Rotor torque constant", "nominal × U(0.7, 1.3)"]},
            {"cells": ["Arm length", "nominal × U(0.7, 1.3)"]},
            {"cells": ["External force (wind)", "U(−3, 3) N per axis, constant"]}]
    names.append(latex_table(
        out, "tableS2_randomization", "S2", "Uncertainty: what is randomized and how.",
        [("Quantity", "l"), ("True vehicle (new draw every episode)", "c")], body,
        notes="The controller and the nominal twin always use the nominal model (Table A1). "
              "Scales are shared by the four rotors.", fs=14)[0])
    names += t_appendix(out)
    body = [{"cells": ["Trajectory", "10 s 3-D figure-8 (±1 m north / up, ±0.5 m east)",
                       "same"]},
            {"cells": ["Control rate", "200 Hz", "same"]},
            {"cells": ["Uncertainty", "Table S2, new draw every episode",
                       "Table S2, 512 fixed draws"]},
            {"cells": ["Same for all controllers", "–",
                       "the 512 conditions, start, trajectory"]},
            {"cells": ["Start", "t = 0, on the reference (first episodes staggered)",
                       "t = 0, on the reference"]},
            {"cells": ["Amount", "2 M steps, 256 parallel envs", "512 flights per policy"]},
            {"cells": ["Seeds", "3 training seeds per policy", "every seed evaluated"]}]
    names.append(latex_table(
        out, "tableS3_train_eval", "S3", "Training and evaluation settings.",
        [("", "l"), ("Training", "c"), ("Evaluation", "c")], body,
        notes="Typical condition = the median of the 512 per-condition RMSEs; hard conditions "
              "= their 90th percentile (P90).", fs=13)[0])
    return names


def t_appendix(out):
    """Appendix: nominal vehicle and controller, residual RL and reward, training."""
    names = []
    body = [{"cells": ["Mass $m$", "2.0 kg"]},
            {"cells": ["Inertia $J$", "diag(0.022, 0.022, 0.040) kg m²"]},
            {"cells": ["Arm length", "0.174 m"]},
            {"cells": ["Max. rotor thrust", "8.55 N (four rotors: 34.2 N)"]},
            {"cells": ["Rotor torque / thrust", "0.016 N m / N"]},
            {"cells": ["Gravity", "9.807 m/s²"]}, "mid",
            {"cells": ["Position gains $k_x$, $k_v$", "36.0, 20.0"]},
            {"cells": ["Position integral $k_i$, $c_1$", "3.6, 5.0  (clipped at ±1, i.e. ±3.6 N)"]},
            {"cells": ["Attitude gains $k_R$, $k_\\Omega$", "8.81, 2.54"]},
            {"cells": ["Attitude integral $k_I$, $c_2$", "0.881, 0.5"]},
            {"cells": ["Control rate", "200 Hz (Δt = 5 ms), RK4 integration"]}]
    names.append(latex_table(out, "tableA1_nominal", "A1",
                             "Nominal vehicle and geometric PID (the model the controller and "
                             "the nominal twin use).",
                             [("Parameter", "l"), ("Value", "l")], body,
                             notes="Gains of the reference geometric PID for the nominal 2 kg "
                                   "vehicle.", fs=13)[0])
    body = [{"cells": ["Action", "$\\Delta u = [\\Delta f, \\Delta M]$, tanh-squashed"]},
            {"cells": ["Residual authority", "0.2 × max. wrench ($\\Delta f \\leq 6.84$ N, $\\Delta M_{x,y} \\leq 0.60$ N m)"]},
            {"cells": ["Thrust LPF", "$\\Delta f_t = (1-\\beta)\\,\\Delta f_{t-1} + \\beta\\, a_t$,  $\\beta = 0.2$"]},
            {"cells": ["Observation", "error vs nominal twin ($p, v, R, \\omega$) and residual, 10 steps"]},
            "mid",
            {"cells": ["Tracking terms $k$", "position, velocity, attitude, body rate"]},
            {"cells": ["Weights $w_k$", "1.0, 0.5, 0.5, 0.2"]},
            {"cells": ["Length scales $\\tau_k$", "0.125 m, 0.15 m/s, 0.25, 0.30 rad/s"]},
            {"cells": ["Effort / smoothness $w_1$, $w_2$", "0.01, 0.01"]}]
    names.append(latex_table(out, "tableA2_residual_reward", "A2",
                             "Residual policy and reward.",
                             [("Item", "l"), ("Value", "l")], body,
                             notes="Reward: tracking terms on the error against the nominal twin "
                                   "(Gaussian kernels, normalized by the weight sum) minus "
                                   "effort and smoothness penalties on the action.", fs=13)[0])
    body = [{"cells": ["Algorithm", "Soft Actor-Critic (SAC)"]},
            {"cells": ["Networks", "MLP 256 × 256 (actor, twin critics with LayerNorm)"]},
            {"cells": ["Learning rates", "$3 \\times 10^{-4}$ (actor, critic, temperature)"]},
            {"cells": ["Discount $\\gamma$, target update $\\tau$", "0.99, 0.005"]},
            {"cells": ["Batch, replay buffer", "256, $3 \\times 10^{5}$"]},
            {"cells": ["Target entropy", "$-0.5 \\times \\dim(a)$"]},
            {"cells": ["Policy / target update", "every 2 / 1 gradient steps, grad. clip 1.0"]},
            "mid",
            {"cells": ["Training steps", "$2 \\times 10^{6}$ (256 parallel environments)"]},
            {"cells": ["Episode", "2,000 steps (10 s at 200 Hz)"]},
            {"cells": ["Checkpoint", "lowest position RMSE, mean of 3 consecutive evaluations (128 fixed episodes)"]},
            {"cells": ["Seeds", "3 per policy"]}]
    names.append(latex_table(out, "tableA3_training", "A3", "Training settings.",
                             [("Item", "l"), ("Value", "l")], body, fs=13)[0])
    return names


def f_reference(out):
    """The reference figure-8, as a small picture for the setup slide."""
    t = np.linspace(0, 10, 1001)
    w = 2 * np.pi / 10
    n, e, u = np.sin(w * t), 0.5 * np.sin(2 * w * t), -np.sin(w * t)
    fig = plt.figure(figsize=(4.6, 4.0))
    ax = fig.add_subplot(projection="3d")
    ax.plot(n, e, u, color=BLUE, lw=2.4)
    ax.scatter([0], [0], [0], color=INK, s=30, depthshade=False)
    ax.text(0.1, 0.0, 0.12, "start", fontsize=12, color=INK)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.7, 0.7); ax.set_zlim(-1.2, 1.2)
    ax.set_box_aspect((2.4, 1.4, 2.4), zoom=0.86)
    ax.view_init(elev=27, azim=-38)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-0.5, 0.5]); ax.set_zticks([-1, 0, 1])
    ax.set_xlabel("north [m]", labelpad=2); ax.set_ylabel("east [m]", labelpad=2)
    ax.set_zlabel("up [m]", labelpad=2)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.97, 0.97, 0.96, 1))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out / "fig_reference.png", dpi=220)
    plt.close(fig)
    return "fig_reference.png"


# ------------------------------------------------------------------ 1. tracking
def t1(A, out):
    rows, _ = table_stats(A, list(KEYS))
    R = {r["key"]: r for r in rows}
    body = []
    for k in KEYS:
        if k not in R:
            continue
        r = R[k]
        red = "–" if k == KB else f"−{r['red_median']:.0f}% / −{r['red_p90']:.0f}%"
        cells = [LAB[k], pm_cm(r["mean"], r["sd"]), cm(r["median"]), cm(r["p90"]), red]
        if not T.SLIDE:                       # rotor saturation: paper table only
            cells.append(f"{r['sat_pct']:.2f}")
        body.append({"cells": cells, "bold": k == KF})
    cols = [("Controller", "l"), ("Mean ± SD [cm]", "c"), ("Median [cm]", "c"),
            ("P90 [cm]", "c"), ("Reduction vs PID (median / P90)", "c")]
    if not T.SLIDE:
        cols.append(("Saturated [%]", "c"))
    fn, md = latex_table(
        out, "table1_tracking", 1, "Tracking error (position RMSE) over 512 conditions.",
        cols, body,
        notes=f"10 s 3-D figure-8; uncertainty: {UNC}. Statistics on the conditions all "
              "three controllers complete; residual policies: mean over 3 training seeds. "
              "Mean ± SD over conditions.")
    return R, fn, md


def f1(R, out):
    if not all(k in R for k in KEYS):
        return None
    fig, ax = plt.subplots(figsize=(11, 4.9))
    bars(ax, R, KEYS)
    corner(ax, f"PID + Residual + LPF vs PID\nmedian −{R[KF]['red_median']:.0f} %   "
               f"P90 −{R[KF]['red_p90']:.0f} %")
    ax.legend(loc="upper left", frameon=False)
    headline(fig, "The residual halves the typical error and cuts the hard cases by 3/4")
    return finish(fig, out, "fig1_tracking_summary",
                  f"512 conditions ({UNC}). Residual policies: mean over 3 training seeds "
                  "(whiskers: SD across seeds).")


def f2(A, out):
    data = {k: [] for k in KEYS}
    for s in A.seeds_of(list(KEYS)):
        mm = {k: A.m(k, s) for k in KEYS}
        j = C.joint_survivors(mm)
        for k in KEYS:
            data[k].append(100 * mm[k]["rmse_alive"][j])
    if not data[KB]:
        return None
    V = {k: np.concatenate(v) for k, v in data.items()}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    order = [KF, KN, KB]
    bp = ax.boxplot([V[k] for k in order], vert=False, widths=0.6, patch_artist=True,
                    whis=(5, 95), medianprops={"color": INK, "lw": 2},
                    flierprops={"marker": "o", "ms": 3, "alpha": 0.35, "mec": "none"})
    for patch, fl, k in zip(bp["boxes"], bp["fliers"], order):
        patch.set_facecolor(COL[k])
        patch.set_edgecolor(SURFACE)
        fl.set_markerfacecolor(COL[k])
    ax.set_yticks([1, 2, 3], [LAB[k] for k in order])
    ax.set_xscale("log")
    plain_log(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=max(v.max() for v in V.values()) * 2.6)
    ax.set_xlabel("position RMSE per condition [cm]  (log scale)")
    corner(ax, f"median {np.median(V[KB]):.1f} → {np.median(V[KF]):.1f} cm")
    headline(fig, "The residual removes the long tail of bad conditions")
    return finish(fig, out, "fig2_error_distribution",
                  "Box: middle 50 % of the conditions; whiskers: 5–95 %; dots: the rest. 512 "
                  "conditions (× 3 training seeds for the residual policies).")


def f_typical_hard(A, out):
    """What 'typical' and 'hard' mean: the 512 conditions sorted by error (PID + Residual + LPF,
    quantile curve averaged over the training seeds, so the two marks equal the bar values)."""
    pct = np.linspace(0, 100, 401)
    curves = []
    for s in A.seeds_of([KF]):
        m = A.m(KF, s)
        r = 100 * m["rmse_alive"][m["terminated"] < 0.5]
        curves.append(np.percentile(r, pct))
    if not curves:
        return None
    q = np.mean(curves, axis=0)
    q50, q90 = np.interp(50, pct, q), np.interp(90, pct, q)
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    ax.axvspan(90, 100, color=ORANGE, alpha=0.13, lw=0)
    ax.plot(pct, q, color=BLUE, lw=2.2)
    for p_, v, c in ((50, q50, INK), (90, q90, ORANGE)):
        ax.plot([p_, p_], [0, v], color=c, lw=1.0, ls=(0, (3, 2)))
        ax.plot([0, p_], [v, v], color=c, lw=1.0, ls=(0, (3, 2)))
        ax.plot([p_], [v], "o", ms=6, color=c, zorder=5)
    ax.text(53, q50 - 0.3, f"typical\n{q50:.1f} cm", ha="left", va="top", fontsize=10,
            color=INK, fontweight="bold", linespacing=1.15)
    ax.text(88, q90 + 0.45, f"hard\n{q90:.1f} cm", ha="right", va="bottom", fontsize=10,
            color=ORANGE, fontweight="bold", linespacing=1.15)
    ax.text(95, 6.15, "worst 10%", ha="center", va="bottom", fontsize=9, color=ORANGE,
            clip_on=False)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 6)
    ax.set_xticks([0, 50, 90, 100])
    ax.set_yticks([0, 2, 4, 6])
    ax.tick_params(labelsize=9)
    ax.set_xlabel("conditions sorted by error [%]", fontsize=9.5)
    ax.set_ylabel("RMSE [cm]", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.3)
    return save(fig, out, "fig_typical_hard")


def f3(A, out, S):
    keys = [k for k in KEYS if k in S]
    t = np.arange(S[KB][0].shape[0]) * DT
    late = t >= 5.0
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 2.55 * len(keys) + 1.1), sharex=True,
                             sharey=True)
    info = {}
    for ax, k in zip(axes, keys):
        y50, y90 = np.percentile(S[k][0], 50, axis=1), np.percentile(S[k][0], 90, axis=1)
        ax.plot(t, y50, color=BLUE, lw=2.2)
        ax.plot(t, y90, color=ORANGE, lw=2.2)
        ax.set_title(LAB[k], loc="left")
        ax.set_ylabel("error [cm]")
        info[k] = {"peak typical": float(y50.max()), "hard after 5 s": float(y90[late].mean())}
        corner(ax, f"typical: peak {y50.max():.1f} cm\nhard: {y90[late].mean():.1f} cm "
                   "after 5 s")
    headroom(axes, lo=0)
    axes[0].set_xlim(0, t[-1])
    axes[-1].set_xlabel("time [s]")
    headline(fig, "The residual cuts the error burst and the error that stays",
             ([line_handle(BLUE), line_handle(ORANGE)],
              ["typical condition (median)", "hard conditions (90th percentile)"]))
    return finish(fig, out, "fig3_error_over_time",
                  "At each instant, the median and the 90th percentile of the position error "
                  "over all surviving conditions (512 × 3 training seeds). The uncertainty is "
                  "on from t = 0.", legend=True), info


def f4(A, out, S, frac):
    if KB not in S or KF not in S:
        return None, None
    Gr, hit = T.integrator_groups(A, frac)
    t = np.arange(S[KB][0].shape[0]) * DT
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, sharey=True)
    for ax, k, who in ((axes[0], KB, "pid"), (axes[1], KF, "rrl")):
        E, ids, _ = S[k]
        for msk, c in ((~hit[ids], GREEN), (hit[ids], ORANGE)):
            ax.plot(t, np.median(E[:, msk], axis=1), color=c, lw=2.2)
        ax.set_title(LAB[k], loc="left")
        ax.set_ylabel("position error [cm]")
        corner(ax, f"within its limit: RMSE {Gr['within'][who]:.1f} cm\n"
                   f"at its limit: RMSE {Gr['limit'][who]:.1f} cm")
    headroom(axes, lo=0)
    axes[0].set_xlim(0, t[-1])
    axes[1].set_xlabel("time [s]")
    headline(fig, "The residual helps most where PID's integral action runs out",
             ([line_handle(GREEN), line_handle(ORANGE)],
              [f"PID's integral term stays within its limit ({Gr['within']['n']} conditions)",
               f"reaches its limit ({Gr['limit']['n']} conditions)"]))
    return finish(fig, out, "fig4_integral_limit",
                  "Median position error over time in each group. 'At its limit' = PID's "
                  "translational integral is clipped for more than a quarter of the flight "
                  "(the clip caps the integral force at 3.6 N per axis).", legend=True), Gr


def f5_cases(A, out, frac):
    s = A.seeds[0]
    if not all(A.has(k, s) for k in (KB, KF)):
        return [], {}
    mm = {k: A.m(k, s) for k in (KB, KF)}
    j = C.joint_survivors(mm)
    rb = mm[KB]["rmse_alive"]
    med, p90 = np.median(rb[j]), np.percentile(rb[j], 90)
    picked = {"typical": int(np.argmin(np.where(j, np.abs(rb - med), np.inf))),
              "hard": int(np.argmin(np.where(j & (frac > 0.25), np.abs(rb - p90), np.inf)))}
    for tag in picked:                                   # fixed picks keep the deck stable
        e_ = getattr(A.a, f"case_{tag}", -1)
        if e_ is not None and e_ >= 0:
            picked[tag] = int(e_)
    cond = bank_conditions(A.spec[KB]["envs"][s], A.a.eval_seed, list(picked.values()))
    names, info = [], {}
    comp = (("north", BLUE), ("east", ORANGE), ("up", GREEN))
    for ci, (tag, e) in enumerate(picked.items()):
        tr = {k: fly_positions(A, k, s, [e]) for k in (KB, KF)}
        t = np.arange(tr[KB]["x"].shape[0]) * DT
        r_b, r_r = 100 * mm[KB]["rmse_alive"][e], 100 * mm[KF]["rmse_alive"][e]
        ratio = r_b / r_r
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2) if T.SLIDE else (11, 6.2),
                                 sharex=True, sharey=True)
        for ax, k, rr in ((axes[0], KB, r_b), (axes[1], KF, r_r)):
            d = 100 * (tr[k]["x"][:, 0] - tr[k]["xd"][:, 0])
            d[:, 2] *= -1                                       # NED down -> up
            for i, (lab, c) in enumerate(comp):
                ax.plot(t, d[:, i], color=c, lw=1.8)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(LAB[k], loc="left")
            ax.set_ylabel("position error [cm]")
            corner(ax, f"RMSE {rr:.1f} cm")
        headroom(axes)
        axes[0].set_xlim(0, t[-1])
        axes[1].set_xlabel("time [s]")
        headline(fig, f"With the residual the error stays {ratio:.0f}× smaller" if ratio >= 1.5
                 else f"With the residual the error stays smaller ({r_b:.1f} → {r_r:.1f} cm)",
                 ([line_handle(c) for _, c in comp], [f"{lab} error" for lab, _ in comp]))
        names.append(finish(fig, out, f"fig5_case_{tag}",
                            f"Training seed {s}, condition {e} of the bank (Table C"
                            f"{ci + 1}). The uncertainty is on from t = 0.", legend=True))
        fig = plt.figure(figsize=(7.0, 4.4) if T.SLIDE else (12, 5.4))
        for i, (k, rr) in enumerate(((KB, r_b), (KF, r_r))):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            ax.set_xticks([-1, 0, 1]); ax.set_yticks([-0.5, 0.5]); ax.set_zticks([-1, 0, 1])
            xd, x = tr[k]["xd"][:, 0], tr[k]["x"][:, 0]
            ax.plot(xd[:, 0], xd[:, 1], -xd[:, 2], color=INK, lw=1.3, ls="--")
            ax.plot(x[:, 0], x[:, 1], -x[:, 2], color=COL[k] if k == KF else INK2, lw=2)
            ax.view_init(elev=27, azim=-38)
            ax.set_xlabel("north [m]", labelpad=6)
            ax.set_ylabel("east [m]", labelpad=6)
            ax.set_zlabel("up [m]", labelpad=6)
            ax.set_title(LAB[k], loc="left")
            ax.text2D(0.98, 0.95, f"RMSE {rr:.1f} cm", transform=ax.transAxes, ha="right",
                      va="top", fontsize=12, fontweight="bold", color=INK)
        headline(fig, "With the residual the flown path stays on the reference",
                 ([line_handle(INK, "--"), line_handle(INK2), line_handle(BLUE)],
                  ["reference", "flown, PID", "flown, PID + Residual + LPF"]))
        names.append(finish(fig, out, f"fig5_path_{tag}",
                            f"Training seed {s}, condition {e} (Table C{ci + 1}).",
                            legend=True))
        names.append(cond_table(out, f"tableC{ci + 1}_condition_{tag}", f"C{ci + 1}",
                                f"Condition {e}: the vehicle and the wind in this flight.",
                                cond, ci)[0])
        info[tag] = {"condition": e, "seed": s, "pid_cm": float(r_b), "rrl_cm": float(r_r),
                     **{q: (cond[q][ci].tolist() if q == "force" else float(cond[q][ci]))
                        for q in cond}}
    return names, info


# ------------------------------------------------------------------ 2. thrust filter
LPF_ROWS = [("Tracking error, median [cm]", "median", 100, "{:.2f}"),
            ("Input roughness", "roughness", 1, "{:.3f}"),
            ("Residual-thrust rate [N/s]", "res_f_slew", 1, "{:.2f}")]


def t2(A, out):
    rows, _ = table_stats(A, list(KEYS))
    R = {r["key"]: r for r in rows}
    if KN not in R or KF not in R:
        return None, None, None
    body = []
    for lab, f, sc, fm in LPF_ROWS:
        a_, b_ = sc * R[KN][f], sc * R[KF][f]
        body.append({"cells": [lab, fm.format(a_), fm.format(b_),
                               f"{100 * (b_ / a_ - 1):+.0f}%".replace("-", "−")]})
    fn, md = latex_table(
        out, "table2_thrust_filter", 2, "Effect of the thrust LPF.",
        [("Metric", "l"), ("PID + Residual", "c"), ("PID + Residual + LPF", "c"),
         ("Change", "c")], body,
        notes="3 training seeds each, 512 conditions.", fs=14, bold_cols=(2,))
    return R, fn, md


def f6_filter_case(A, out):
    best = T._helping_case(A, "PID")
    if best is None:
        return None, {}
    s, e, _ = best
    trs = {k: D.fly_instr(A.spec[k]["envs"][s], A.spec[k]["runs"][s], A.a.eval_seed, [e],
                          None, chunk=1) for k in (KN, KF)}
    mot = trs[KN]["motor_cmd"][:, 0]
    act = np.convolve(np.abs(np.diff(mot, 2, axis=0)).sum(axis=1), np.ones(40), "same")
    act[:2 * W0] = 0
    c = int(np.clip(np.argmax(act), 2 * W0 + 40, len(act) - 41))
    w = slice(c - 40, c + 40)                                     # 0.4 s window
    t = np.arange(mot.shape[0])[w] * DT
    lo = min(trs[k]["motor_cmd"][w, 0].min() for k in trs)
    hi = max(trs[k]["motor_cmd"][w, 0].max() for k in trs)
    pad = 0.15 * (hi - lo + 1e-6)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    mets = {}
    for ax, k in ((axes[0], KN), (axes[1], KF)):
        for r in range(4):
            ax.plot(t, trs[k]["motor_cmd"][w, 0, r], color=SLOT[r], lw=1.6)
        m = D.case_metrics(trs[k], A.spec[k]["envs"][s])
        mets[k] = m
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel("rotor commands [N]")
        ax.set_title(LAB[k] + (" (proposed)" if k == KF else ""), loc="left")
        corner(ax, f"rotor jitter {m['motor_hf']:.3f} N")
    axes[1].set_xlabel("time [s]")
    headline(fig, "With the thrust LPF, the rotor commands stay smooth",
             ([line_handle(SLOT[r]) for r in range(4)], [f"rotor {r + 1}" for r in range(4)]))
    cond = bank_conditions(A.spec[KF]["envs"][s], A.a.eval_seed, [e])
    fn = finish(fig, out, "fig6_filter_case",
                f"Training seed {s}, condition {e} (Table C3). A 0.4 s window in steady "
                "flight. The condition where the filter helps most; averaged over all "
                "conditions see Table 2. Tracking "
                f"{mets[KN]['rmse_cm']:.2f} vs {mets[KF]['rmse_cm']:.2f} cm.", legend=True)
    ct = cond_table(out, "tableC3_condition_filter", "C3",
                    f"Condition {e}: the vehicle and the wind in this flight.", cond, 0)[0]
    return [fn, ct], {"seed": s, "condition": e, "no filter": mets[KN], "filter": mets[KF]}


# ------------------------------------------------------------------ 3. ablations (with LPF)
def _abl_from_csv(path, mapping):
    R = {}
    if not Path(path).is_file():
        return R

    def pm(s_):
        a_, _, b_ = s_.partition("±")
        return float(a_) / 100, (float(b_) / 100 if b_ else 0.0)
    for r in csv.DictReader(open(path)):
        if r["base"] != "PID" or r["variant"] not in mapping:
            continue
        med, msd = pm(r["median [cm]"])
        p90, psd = pm(r["P90 [cm]"])
        mean, sd = pm(r["mean ± SD [cm]"])
        R[mapping[r["variant"]]] = dict(median=med, median_sd=msd, p90=p90, p90_sd=psd,
                                        mean=mean, sd=sd, sat_pct=float(r["saturated steps [%]"]))
    return R


def ablation_pid(out, name, number, title, caption, items, R, note, colors, figsize=(11, 4.9)):
    items = [it for it in items if it[0] in R]
    if len(items) < 2:
        return [], f"_{name}: not available_"
    body = [{"cells": [lab, pm_cm(R[k]["median"], R[k]["median_sd"]),
                       pm_cm(R[k]["p90"], R[k]["p90_sd"]), pm_cm(R[k]["mean"], R[k]["sd"])],
             "bold": prop} for k, lab, prop in items]
    tfn, md = latex_table(out, f"table{number}_{name}", number, caption,
                          [("PID + Residual + LPF", "l"), ("Median [cm]", "c"),
                           ("P90 [cm]", "c"), ("Mean ± SD [cm]", "c")], body, notes=note,
                          fs=13)
    fig, ax = plt.subplots(figsize=figsize)
    x, w = np.array([0.0, 1.0]), 0.8 / len(items)
    top = 100 * max(R[k]["p90"] for k, _, _ in items)
    for i, ((k, lab, prop), c) in enumerate(zip(items, colors)):
        v = [100 * R[k]["median"], 100 * R[k]["p90"]]
        sd = [100 * np.nan_to_num(R[k]["median_sd"]), 100 * np.nan_to_num(R[k]["p90_sd"])]
        xs = x + (i - (len(items) - 1) / 2) * w
        ax.bar(xs, v, w * 0.92, color=c, yerr=sd, label=lab,
               error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
        for xi, vi, si in zip(xs, v, sd):
            ax.text(xi, vi + si + 0.02 * top, f"{vi:.1f}", ha="center", va="bottom",
                    fontsize=11.5, fontweight="bold" if prop else None)
    ax.set_ylim(0, top * 1.55)
    ax.set_xticks(x, ["typical condition (median)", "hard conditions (P90)"])
    ax.set_ylabel("position RMSE [cm]")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False)
    corner(ax, "all variants: PID + Residual + LPF")
    headline(fig, title)
    ffn = finish(fig, out, f"fig_{name}", note)
    return [ffn, tfn], md


def ablations(A, out, csv_dir):
    hist_items = [("PID_h1", "history 1 (current error only)", False),
                  (KF, "history 10 (proposed)", True), ("PID_h20", "history 20", False)]
    chan_items = [(KF, "LPF on thrust only (proposed)", True),
                  ("PID_lpfall", "LPF on thrust + torques", False)]
    R = {}
    if A is not None:
        keys = [k for k, _, _ in hist_items + chan_items if A.has(k)]
        if len(keys) >= 3:
            R = {r["key"]: r for r in table_stats(A, keys)[0]}
    if csv_dir and not all(k in R for k, _, _ in hist_items):
        R.update(_abl_from_csv(Path(csv_dir) / "fig9_history.csv",
                               {"history 1 (current error only)": "PID_h1",
                                "history 10 (proposed)": KF, "history 20": "PID_h20"}))
    if csv_dir and "PID_lpfall" not in R:
        R.update(_abl_from_csv(Path(csv_dir) / "fig10_filter_channels.csv",
                               {"thrust + torques": "PID_lpfall"}))
    names, md = [], []
    n, m = ablation_pid(
        out, "history", 3, "History length: one step loses much of the gain, 20 adds nothing",
        "Observation history length (all with the thrust LPF).", hist_items, R,
        "History = how many past steps of the error against the nominal twin (and of the "
        "policy's own residual) the policy sees; 1 = the current error only. All three are "
        "PID + Residual + LPF; 3 training seeds each, 512 conditions.",
        (T.VARIANT_GRAYS[0], BLUE, T.VARIANT_GRAYS[1]),
        figsize=(7.7, 4.4) if T.SLIDE else (11, 4.9))     # slide: room for the explainer
    names += n
    md.append(m)
    n, m = ablation_pid(
        out, "filter_channels", 4, "Filtering the torques too costs tracking",
        "Which residual channels the LPF filters.", chan_items, R,
        "Same first-order filter (β 0.2) on the residual thrust only, or on the thrust and the "
        "three torques. 3 training seeds each, 512 conditions.", (BLUE, T.VARIANT_GRAYS[1]))
    names += n
    md.append(m)
    return names, md, R


# ------------------------------------------------------------------ Q&A: base vs residual
ROLE_F = ("residual", "u_base", "ctrl_ei", "pos_err_desired", "terminated")


def role_bank(A):
    """Per condition, after the start transient (t >= 2 s): mean residual thrust, mean
    integral force of the base, RMSE; PID alone and the proposed policy (all seeds)."""
    ids = list(range(A.a.episodes))
    env_b = A.spec[KB]["envs"][A.seeds[0]]
    ki = float(getattr(env_b, "pid_ki", None) or 3.6)       # Gains default
    out = {"cond": bank_conditions(env_b, A.a.eval_seed, ids)}
    W = slice(int(2.0 / DT), None)

    def stats(tr):
        dead = (np.cumsum(tr["terminated"], 0) > 0).any(0)
        return {"alive": ~dead, "rmse": np.sqrt((tr["pos_err_desired"] ** 2).mean(0)),
                "res": tr["residual"][W].mean(0) if "residual" in tr else None,
                "int": ki * tr["ctrl_ei"][W].mean(0), "base": tr["u_base"][W].mean(0)}
    out["pid"] = stats(D.fly_instr(env_b, None, A.a.eval_seed, ids, chunk=A.a.chunk,
                                   fields=ROLE_F))
    out["rrl"] = []
    for s in A.seeds:
        if A.has(KF, s):
            out["rrl"].append(stats(D.fly_instr(A.spec[KF]["envs"][s], A.spec[KF]["runs"][s],
                                                A.a.eval_seed, ids, chunk=A.a.chunk,
                                                fields=ROLE_F)))
    return out


def f_roles(rb, out):
    need = needed_thrust(rb["cond"])
    alive = rb["pid"]["alive"] & np.all([r["alive"] for r in rb["rrl"]], axis=0)
    res = np.mean([r["res"][:, 0] for r in rb["rrl"]], axis=0)
    i_r = np.mean([r["int"][:, 2] for r in rb["rrl"]], axis=0)
    i_p = rb["pid"]["int"][:, 2]
    e_p = 100 * rb["pid"]["rmse"]
    e_r = 100 * np.mean([r["rmse"] for r in rb["rrl"]], axis=0)
    a = alive
    slope, icpt = np.polyfit(need[a], res[a], 1)
    r_corr = np.corrcoef(need[a], res[a])[0, 1]
    nom = np.abs(need) < 1.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1]})
    ax = axes[0]
    lim = 3.6
    ax.axhspan(-lim, lim, color="#eeeeea", zorder=0, lw=0)
    ax.text(-9.6, lim - 0.25, "PID integral force limit ±3.6 N", fontsize=10, color=INK2,
            va="top")
    ax.scatter(need[a], i_p[a], s=11, color=BASE_GRAY, alpha=0.75, lw=0, zorder=3)
    ax.scatter(need[a], res[a], s=11, color=ORANGE, alpha=0.85, lw=0, zorder=4)
    ax.scatter(need[a], i_r[a], s=11, color=BLUE, alpha=0.6, lw=0, zorder=3)
    xx = np.linspace(-10, 12, 2)
    ax.plot(xx, xx, color=INK, lw=1.0, ls=":", zorder=2)
    ax.text(-4.6, -8.9, "dotted: residual = missing thrust", fontsize=10, color=INK2,
            ha="left")
    ax.text(11.8, 7.25, "residual limit (authority 0.2)", fontsize=10, color=INK2, ha="right",
            va="bottom")
    ax.set_ylabel("steady thrust offset [N]")
    ax.set_title("Who supplies the missing thrust", loc="left")
    corner(ax, f"residual ≈ {slope:.2f} × missing thrust\n(r = {r_corr:.2f})")
    ax = axes[1]
    order = np.argsort(need[a])
    xs = need[a][order]
    for e_, c in ((e_p, BASE_GRAY), (e_r, BLUE)):
        ys = e_[a][order]
        ax.scatter(xs, ys, s=9, color=c, alpha=0.35, lw=0)
        edges = np.linspace(-10, 12, 23)
        mid = 0.5 * (edges[1:] + edges[:-1])
        med = [np.median(ys[(xs >= lo_) & (xs < hi_)]) if ((xs >= lo_) & (xs < hi_)).sum() >= 5
               else np.nan for lo_, hi_ in zip(edges[:-1], edges[1:])]
        ax.plot(mid, med, color=c, lw=2.4)
    ax.set_yscale("log")
    plain_log(ax, "y")
    ax.set_ylabel("position RMSE [cm]")
    ax.set_xlabel("thrust the nominal model is missing at hover [N]  (heavier, weaker rotors "
                  "or downward wind → positive)")
    ax.set_title("Tracking error", loc="left")
    corner(ax, f"|missing| < 1 N: |residual| {np.mean(np.abs(res[a & nom])):.1f} N on average")
    axes[0].set_xlim(-10, 12)
    headline(fig, f"The residual supplies about {100 * slope:.0f} % of the thrust the model "
                  "is missing",
             ([Line2D([0], [0], ls="none", marker="o", color=ORANGE, ms=7),
               Line2D([0], [0], ls="none", marker="o", color=BLUE, ms=7),
               Line2D([0], [0], ls="none", marker="o", color=BASE_GRAY, ms=7)],
              ["residual thrust (PID + Residual + LPF)",
               "base integral force, with the residual", "base integral force, PID alone"]),
             ncol=3)
    fn = finish(fig, out, "figQ1_roles",
                "Each dot is one of the 512 conditions (mean after t = 2 s; residual: mean of "
                "3 training seeds). Missing thrust = (k·m·g + F_down)/k_f − m·g. Beyond "
                "≈ 6.8 N the residual is at its limit and the base integral takes the rest. "
                "Lines: binned medians.", legend=True)
    info = {"slope": float(slope), "intercept": float(icpt), "r": float(r_corr),
            "n": int(a.sum()), "near_nominal_abs_res": float(np.mean(np.abs(res[a & nom]))),
            "int_rrl_mean_abs": float(np.mean(np.abs(i_r[a]))),
            "int_pid_at_limit_pct": float(100 * np.mean(np.abs(i_p[a]) > 3.5))}
    return fn, info


# ------------------------------------------------------------------ Q&A: fixed conditions
def fly_fixed(env_cfg, run_dir, conds, fields):
    """Flights under chosen conditions: the drawn vehicle and wind are overwritten before
    the first step (nothing else in the state depends on them)."""
    from qrjax.core.dynamics import Inertia
    from qrjax.core.mixer import MixerParams
    agent, params = (None, None) if run_dir is None else C.load_policy(run_dir)[1:]
    E = len(conds)
    env = C.VecEnv(env_cfg, E)
    ranges = env.broadcast_ranges(C.flat_ranges(env_cfg, force_freq=0.0, force_dc_prob=0.0))
    g = lambda n_: jnp.asarray([float(c.get(n_, 1.0)) for c in conds], jnp.float32)
    k = g("k")
    J = jnp.diag(jnp.asarray(env_cfg.J_nom, jnp.float32))
    rot = lambda n_: jnp.broadcast_to(g(n_)[:, None], (E, 4))
    F = jnp.asarray([np.asarray(c.get("force", (0, 0, 0)), float) for c in conds], jnp.float32)

    def run(params):
        st = env._reset(jax.random.split(jax.random.PRNGKey(7), E), ranges,
                        jnp.zeros(E, jnp.int32))
        st = st._replace(k=k, inertia_true=Inertia(mass=k * env_cfg.mass_nom,
                                                   J=k[:, None, None] * J[None]),
                         mixer_true=MixerParams(kf_scale=rot("kf"), moment_scale=rot("km"),
                                                arm_scale=rot("arm")),
                         external_force=F, force_freq=jnp.zeros(E),
                         force_phase=jnp.zeros((E, 3)))
        ob = env._reset_obs(st)

        def body(c, _):
            st, ob = c
            act = (jnp.zeros((E, 4)) if agent is None
                   else jnp.tanh(agent.actor.apply(params, ob)[0]))
            st, ob, _, _, info = env.step(st, act, ranges)
            return (st, ob), {f: info[f] for f in fields}
        return jax.lax.scan(body, (st, ob), None, length=env_cfg.episode_steps)[1]
    return {f: np.asarray(v) for f, v in jax.jit(run)(params).items()}


def qa_fixed(A, out):
    """Nominal vs large uncertainty, and a downdraft: climbing vs descending."""
    fields = ("residual", "u_base", "pos_err_desired", "terminated", "ctrl_ei")
    conds = {"nominal": {}, "large": dict(k=1.069, kf=0.789, km=1.026, arm=0.950,
                                          force=(-2.535, 2.502, 0.817)),
             "down1": dict(force=(0, 0, 1.0)), "down2": dict(force=(0, 0, 2.0)),
             "down3": dict(force=(0, 0, 3.0))}
    L = list(conds.values())
    trb = fly_fixed(A.spec[KB]["envs"][A.seeds[0]], None, L, fields)
    trr = [fly_fixed(A.spec[KF]["envs"][s], A.spec[KF]["runs"][s], L, fields)
           for s in A.seeds if A.has(KF, s)]
    t = np.arange(1, trb["residual"].shape[0] + 1) * DT
    after = t >= 2.5                                  # one climb (2.5-7.5 s), one descent
    desc = after & (t >= 7.5)
    climb = after & (t < 7.5)
    i = {n_: j for j, n_ in enumerate(conds)}
    rm = lambda tr, n_, m=slice(None): 100 * np.sqrt(np.mean(tr["pos_err_desired"][m, i[n_]]
                                                              ** 2))
    # table: nominal vs large
    body = []
    for n_, lab in (("nominal", "No uncertainty"), ("large", "Large uncertainty (cond. 339)")):
        res = [np.mean(tr["residual"][int(2 / DT):, i[n_], 0]) for tr in trr]
        base = [np.mean(tr["u_base"][int(2 / DT):, i[n_], 0]) for tr in trr]
        body.append({"cells": [lab, f"{rm(trb, n_):.2f}",
                               f"{np.mean([rm(tr, n_) for tr in trr]):.2f}",
                               f"{np.mean(base):.1f}", f"{np.mean(res):+.2f}",
                               f"{100 * np.mean(res) / (np.mean(base) + np.mean(res)):.0f}%"]})
    fq2 = latex_table(out, "tableQ2_nominal_vs_large", "Q2",
                      "Base and residual: no uncertainty vs large uncertainty.",
                      [("Condition", "l"), ("PID [cm]", "c"), ("PID + Res. + LPF [cm]", "c"),
                       ("Base thrust [N]", "c"), ("Residual thrust [N]", "c"),
                       ("Residual share", "c")], body,
                      notes="RMSE over the flight; thrust commands averaged after t = 2 s, "
                            "mean of 3 training seeds. Large uncertainty: mass ×1.07, rotor "
                            "thrust ×0.79, torque ×1.03, arm ×0.95, wind (−2.5, 2.5, 0.8) N.",
                      fs=13)[0]
    # figure: downdraft, climbing vs descending
    forces = [1.0, 2.0, 3.0]
    rd = np.array([[np.mean(tr["residual"][desc, i[f"down{int(f)}"], 0]) for tr in trr]
                   for f in forces])
    rc = np.array([[np.mean(tr["residual"][climb, i[f"down{int(f)}"], 0]) for tr in trr]
                   for f in forces])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    x, w = np.arange(len(forces), dtype=float), 0.36
    for j, (v, c, lab) in enumerate(((rc, LIGHT_BLUE, "climbing (2.5–7.5 s)"),
                                     (rd, BLUE, "descending (7.5–10 s)"))):
        m_, s_ = v.mean(1), v.std(1, ddof=1)
        xs = x + (j - 0.5) * w
        ax.bar(xs, m_, w * 0.92, color=c, yerr=s_, label=lab,
               error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
        for xi, vi, si in zip(xs, m_, s_):
            ax.text(xi, vi + si + 0.08, f"{vi:.1f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x, [f"{f:.0f} N" for f in forces])
    ax.set_xlabel("downward wind")
    ax.set_ylabel("mean residual thrust [N]")
    ax.set_ylim(0, max(rd.max(), rc.max()) * 1.45)
    ax.set_title("Residual thrust", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False)
    ax = axes[1]
    eb_d = [rm(trb, f"down{int(f)}", desc) for f in forces]
    eb_c = [rm(trb, f"down{int(f)}", climb) for f in forces]
    er_d = [np.mean([rm(tr, f"down{int(f)}", desc) for tr in trr]) for f in forces]
    er_c = [np.mean([rm(tr, f"down{int(f)}", climb) for tr in trr]) for f in forces]
    for v, c, ls, lab in ((eb_c, BASE_GRAY, "--", "PID, climbing"),
                          (eb_d, BASE_GRAY, "-", "PID, descending"),
                          (er_c, BLUE, "--", "PID + Residual + LPF, climbing"),
                          (er_d, BLUE, "-", "PID + Residual + LPF, descending")):
        ax.plot(forces, v, color=c, ls=ls, lw=2.2, marker="o", label=lab)
    ax.set_xticks(forces, [f"{f:.0f} N" for f in forces])
    ax.set_xlabel("downward wind")
    ax.set_ylabel("position RMSE [cm]")
    ax.set_ylim(0, max(max(er_d), max(er_c), max(eb_d), max(eb_c)) * 1.6)
    ax.set_title("Tracking error after the start (t ≥ 2.5 s)", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    headline(fig, "A downward wind: the residual adds about the same thrust climbing and "
                  "descending")
    ffn = finish(fig, out, "figQ3_downdraft",
                 "Nominal vehicle, constant downward force. After the start transient (t ≥ 2.5 s"
                 "): one climb and one descent of the figure-8. 3 training seeds (whiskers: SD). "
                 "The simulated wind is a constant force: it pushes the same whether the vehicle "
                 "climbs or descends (no air-speed drag, no descent into its own downwash).")
    # the left panel alone, for a slide
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    for j, (v, c, lab) in enumerate(((rc, LIGHT_BLUE, "climbing (2.5–7.5 s)"),
                                     (rd, BLUE, "descending (7.5–10 s)"))):
        m_, s_ = v.mean(1), v.std(1, ddof=1)
        xs = x + (j - 0.5) * w
        ax.bar(xs, m_, w * 0.92, color=c, yerr=s_, label=lab,
               error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
        for xi, vi, si in zip(xs, m_, s_):
            ax.text(xi, vi + si + 0.08, f"{vi:.1f}", ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x, [f"{f:.0f} N" for f in forces])
    ax.set_xlabel("downward wind")
    ax.set_ylabel("mean residual thrust [N]")
    ax.set_ylim(0, max(rd.max(), rc.max()) * 1.5)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False)
    headline(fig, "Climbing or descending: the same residual")
    ffn2 = finish(fig, out, "figQ3a_downdraft_residual",
                  "Nominal vehicle, constant downward force; t ≥ 2.5 s; 3 training seeds "
                  "(whiskers: SD).")
    info = {"res_desc": rd.mean(1).tolist(), "res_climb": rc.mean(1).tolist(),
            "pid_desc": eb_d, "pid_climb": eb_c, "rrl_desc": er_d, "rrl_climb": er_c}
    return [fq2, ffn, ffn2], info


# ------------------------------------------------------------------ backup
def tB1(A, out):
    keys = [KB, "PID_fixed_noclip", KF]
    if not all(A.has(k) for k in keys):
        return None
    R = {r["key"]: r for r in table_stats(A, keys)[0]}
    lab = {KB: "PID", "PID_fixed_noclip": "PID, no integrator limit", KF: LAB[KF]}
    body = [{"cells": [lab[k], cm(R[k]["median"]), cm(R[k]["p90"]),
                       pm_cm(R[k]["mean"], R[k]["sd"])], "bold": k == KF} for k in keys]
    return latex_table(out, "tableB1_integrator_limit", "B1", "Is it only PID's integrator limit?",
                       [("Controller", "l"), ("Median [cm]", "c"), ("P90 [cm]", "c"),
                        ("Mean ± SD [cm]", "c")], body,
                       notes="The reference PID clips its translational integral (anti-windup), "
                             "capping the integral force at 3.6 N per axis. Removing the clip "
                             "helps PID's hard conditions (P90), not its typical ones.",
                       fs=13)[0]


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="study_results")
    p.add_argument("--paper_dir", default="auto")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--parts", default="S,1,2,3,B,Q")
    p.add_argument("--slide", action="store_true",
                   help="slide versions: figures without title / footnote, tables without "
                        "caption / notes; writes 07_deck_slide/")
    p.add_argument("--case_typical", type=int, default=-1, help="-1: the median condition")
    p.add_argument("--case_hard", type=int, default=339,
                   help="339: the P90 condition of the talk (PID 13.2 cm, P90 13.1 cm); -1: auto")
    p.add_argument("--font_dir", default="",
                   help="folder with KoPubDotum-Medium/Bold.ttf: figures in the deck font")
    p.add_argument("--ablation_csv", default="",
                   help="06_talk dir with fig9_history.csv / fig10_filter_channels.csv to use "
                        "when the H / M checkpoints are not here")
    a = p.parse_args()
    a.case = None
    T.SLIDE = a.slide
    if a.font_dir:
        T.use_deck_font(a.font_dir)
    out = Path(a.results) / ("07_deck_slide" if a.slide else "07_deck")
    out.mkdir(parents=True, exist_ok=True)
    parts = set(a.parts.split(","))
    A = T.extend(Arms(a))
    L, names, info = ["# KSAS deck v4 (PID only)", ""], [], {}
    if "S" in parts:
        names += t_setup(out)
        names.append(f_reference(out))
    if "1" in parts:
        R1, fn, md = t1(A, out)
        names += [fn, f1(R1, out), f2(A, out), f_typical_hard(A, out)]
        L += ["## 1. Tracking", "", md, ""]
        S = T._err_series(A, list(KEYS))
        fn, info["fig3"] = f3(A, out, S)
        names.append(fn)
        frac = T.pid_integrator_clip(A)
        fn, info["integrator_groups"] = f4(A, out, S, frac)
        names.append(fn)
        n, info["cases"] = f5_cases(A, out, frac)
        names += n
    if "C" in parts and "1" not in parts:
        n, info["cases"] = f5_cases(A, out, T.pid_integrator_clip(A))
        names += n
    if "2" in parts:
        R2, fn, md = t2(A, out)
        names.append(fn)
        L += ["## 2. Thrust LPF", "", md or "_missing_", ""]
        n, info["filter_case"] = f6_filter_case(A, out)
        names += n or []
        names.append(T.f8_reason_diagram(out))
    if "3" in parts:
        n, md, _ = ablations(A, out, a.ablation_csv)
        names += n
        L += ["## 3. Ablations (all with the thrust LPF)", ""] + md + [""]
    if "B" in parts:
        names.append(tB1(A, out))
    if "Q" in parts:
        rb = role_bank(A)
        fn, info["roles"] = f_roles(rb, out)
        names.append(fn)
        n, info["qa_fixed"] = qa_fixed(A, out)
        names += n
    import json
    jf = out / "DECK.json"
    old = json.loads(jf.read_text()) if jf.is_file() else {}
    old.update(json.loads(json.dumps(info, default=float)))
    jf.write_text(json.dumps(old, indent=2))
    L += ["## Files", ""] + [f"- {n}" for n in names if n]
    (out / f"DECK_{'_'.join(sorted(parts))}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
