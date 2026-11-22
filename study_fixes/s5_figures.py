"""Stage 5 -- presentation tables and figures from the trained policies.

No training. Every statistic comes from the per-arm evaluation cache written by
stages 1 and 3 (512-episode common bank); only the trajectory, error-vs-time
and rotor-command figures fly new rollouts, with the same bank keys.

Arms (PID base unless stated; "fixed" = every bug fix on):
  PID / PD                 baselines, original and fixed code
  Paper                    the paper's PID/PD + RRL + LPF checkpoints, original code
  Fixed                    retrained with every fix, paper reward (tau_pos 0.5) and authority 0.2
  D1                       fixed, tau_pos 0.1, authority 0.2   (sharper reward)
  D2                       fixed, tau_pos 0.2, authority 0.4   (sharper reward + more authority)
plus the unfiltered (no-LPF) counterparts used by the LPF ablation.

Statistics are per seed on the episodes that every arm of the table survives
(the paper's convention), then mean ± SD over the 3 seeds.

Writes <results>/04_presentation/.
"""

import argparse
from pathlib import Path

import numpy as np

from . import common as C
from .s1_ablation import paper_dir_from, variant_overrides
from .s2_train import finished_run

import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter   # noqa: E402

# ------------------------------------------------------------------ style
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
BASE_GRAY = "#7a7975"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # validated categorical order
AUTH_RAMP = {0.2: "#86b6ef", 0.4: "#2a78d6", 0.6: "#104281"}   # sequential blue (ordinal)
EXTREME_N = 0.3        # absolute chatter threshold [N]: ~3x the original baseline's motor HF

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2.0,
})


def plain_log(ax, axis="y"):
    """Log axis with plain-number ticks (1, 2, 5 per decade) instead of 10^x."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{name}.png"


# ------------------------------------------------------------------ arms
class Arms:
    """Registry of arms: label, colour, base, per-seed (run_dir, env)."""

    def __init__(self, a):
        self.a = a
        self.kw = dict(eval_seed=a.eval_seed, n_episodes=a.episodes, chunk=a.chunk)
        self.seeds = [int(s) for s in a.seeds.split(",")]
        self.paper_dir = paper_dir_from(a.results, a.paper_dir)
        self.pol = C.paper_policies(self.paper_dir) if self.paper_dir else {}
        self.best_seed = C.paper_best_seed(self.paper_dir) if self.paper_dir else None
        fx = C.all_fixes()
        self.spec = {}
        for base in ("pid", "pd"):
            B = base.upper()
            self._base(f"{B}_orig", f"{B} (original code)", base, C.paper_env(base_controller=base))
            self._base(f"{B}_fixed", f"{B} (fixed code)", base,
                       C.paper_env(base_controller=base, **fx))
            for tag, sfx, lpf in (("thrust", "lpf", True), ("none", "nolpf", False)):
                runs = {s: self.pol.get((s, tag, base)) for s in self.seeds}
                self._pol(f"{B}_paper_{sfx}", f"{B}+RRL{'' if lpf else ' no LPF'} (paper)",
                          base, runs, {}, "original")
                self._pol(f"{B}_paperfx_{sfx}", f"{B}+RRL{'' if lpf else ' no LPF'} "
                          "(paper policy, fixed physics)", base, runs, fx, "fixed")
                self._pol(f"{B}_paperan_{sfx}", f"{B}+RRL{'' if lpf else ' no LPF'} "
                          "(paper policy, analytic rates only)", base, runs,
                          {"rate_ff": "analytic"}, "original")
                runs = {s: finished_run(a.results, f"A_{base}_{sfx}_s{s}") for s in self.seeds}
                self._pol(f"{B}_fixed_{sfx}", f"{B}+RRL{'' if lpf else ' no LPF'} (fixed)",
                          base, runs, {}, "fixed")
        for key, job, lab in (
                ("D1", "B_pid_lpf_tau0.1_auth0.2", "PID+RRL D1 (τ 0.1, α 0.2)"),
                ("D2", "B_pid_lpf_tau0.2_auth0.4", "PID+RRL D2 (τ 0.2, α 0.4)"),
                ("sharp_lpf", "B_pid_lpf_tau0.05_auth0.2", "τ 0.05, α 0.2, LPF"),
                ("sharp_nolpf", "C_pid_nolpf_tau0.05_auth0.2", "τ 0.05, α 0.2, no LPF"),
                ("hi_lpf", "B_pid_lpf_tau0.05_auth0.6", "τ 0.05, α 0.6, LPF"),
                ("hi_nolpf", "C_pid_nolpf_tau0.05_auth0.6", "τ 0.05, α 0.6, no LPF")):
            runs = {s: finished_run(a.results, f"{job}_s{s}") for s in self.seeds}
            self._pol(key, lab, "pid", runs, {}, "fixed")
        self._m = {}

    def _base(self, key, label, base, env):
        self.spec[key] = {"label": label, "base": base, "code": "fixed" if "fixed" in key
                          else "original", "runs": {s: None for s in self.seeds},
                          "envs": {s: env for s in self.seeds}, "policy": False}

    def _pol(self, key, label, base, runs, overrides, code):
        runs = {s: r for s, r in runs.items() if r is not None}
        self.spec[key] = {"label": label, "base": base, "code": code, "runs": runs,
                          "envs": {s: C.env_from_run(r, **overrides) for s, r in runs.items()},
                          "policy": True}

    def has(self, key, seed=None):
        sp = self.spec.get(key)
        if not sp or not sp["envs"]:
            return False
        return seed is None or seed in sp["envs"]

    def m(self, key, seed):
        if (key, seed) not in self._m:
            sp = self.spec[key]
            self._m[(key, seed)] = C.cached_fly(
                self.a.results, f"{key} s{seed}", sp["envs"][seed],
                run_dir=sp["runs"].get(seed), **self.kw)
        return self._m[(key, seed)]

    def baseline_of(self, key):
        sp = self.spec[key]
        return f"{sp['base'].upper()}_{'fixed' if sp['code'] == 'fixed' else 'orig'}"

    def seeds_of(self, keys):
        return [s for s in self.seeds if all(self.has(k, s) for k in keys)]


# ------------------------------------------------------------------ statistics
def ms(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not v.size:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def table_stats(A, keys):
    """Seed-mean statistics for a set of arms on their per-seed joint survivors."""
    keys = [k for k in keys if A.has(k)]
    per = {k: [] for k in keys}
    for s in A.seeds:
        ks = [k for k in keys if A.has(k, s)]
        if not ks:
            continue
        mm = {k: A.m(k, s) for k in ks}
        bases = {A.baseline_of(k) for k in ks}
        for b in bases:
            mm.setdefault(b, A.m(b, s))
        C.check_same_draws(mm)
        joint = C.joint_survivors(mm)
        for k in ks:
            m = mm[k]
            r = m["rmse_alive"][joint]
            b = mm[A.baseline_of(k)]
            rb = b["rmse_alive"][joint]
            alive = m["terminated"] < 0.5
            hf = m["motor_hf"][alive]
            per[k].append({
                "seed": s, "n_cond": int(joint.sum()),
                "mean": float(r.mean()), "sd": float(r.std(ddof=1)),
                "median": float(np.median(r)), "p90": float(np.percentile(r, 90)),
                "red_mean": 100 * (1 - r.mean() / rb.mean()),
                "red_median": 100 * (1 - np.median(r) / np.median(rb)),
                "red_p90": 100 * (1 - np.percentile(r, 90) / np.percentile(rb, 90)),
                "term_pct": 100 * float(np.mean(~alive)),
                "own_fail": C.own_failures(m, b),
                "sat_pct": float(np.mean(m["sat_pct"][joint])),
                "roughness": float(np.mean(m["roughness"][joint])),
                "motor_hf": float(np.mean(m["motor_hf"][joint])),
                "motor_hf_max": float(np.max(hf)) if hf.size else np.nan,
                "n_extreme": int(np.sum(hf > EXTREME_N)),
                "res_f_slew": float(np.nanmean(m["res_f_slew"][joint])),
            })
    rows = []
    for k in keys:
        if not per[k]:
            continue
        r = {"key": k, "arm": A.spec[k]["label"], "seeds": len(per[k])}
        for f in per[k][0]:
            if f == "seed":
                continue
            r[f], r[f + "_sd"] = ms([p[f] for p in per[k]])
        rows.append(r)
    return rows, per


def pm(v, s, f="{:.4f}"):
    if not np.isfinite(v):
        return "–"
    return f.format(v) + (f" ± {f.format(s)}" if np.isfinite(s) and s > 0 else "")


# ------------------------------------------------------------------ tables
def write_tables(A, out):
    L, rows_all = [], {}
    # Table 1 in the paper's format
    for base in ("PID", "PD"):
        keys = [f"{base}_orig", f"{base}_paper_lpf", f"{base}_fixed", f"{base}_fixed_lpf"]
        if base == "PID":
            keys += ["D1", "D2"]
        rows, _ = table_stats(A, keys)
        rows_all[f"table1_{base}"] = rows
        C.write_csv(out / f"table1_{base.lower()}.csv", rows)
        show = [{"arm": r["arm"], "seeds": r["seeds"],
                 "msd": f"{r['mean']:.4f} ± {r['sd']:.4f}",
                 "mp": f"{r['median']:.4f} / {r['p90']:.4f}",
                 "sat": f"{r['sat_pct']:.3f}", "term": f"{r['term_pct']:.2f}",
                 "red": "–" if r["key"].endswith(("_orig", "_fixed")) else
                 f"{r['red_mean']:.1f} / {r['red_median']:.1f} / {r['red_p90']:.1f}",
                 "rough": f"{r['roughness']:.3f}"} for r in rows]
        L += [f"## Table 1 ({base}), paper format", "",
              "Episode statistics on the episodes every arm survives, averaged over seeds. "
              "`reduction` = mean / median / P90 RMSE reduction against the baseline of the "
              "same code, in %.", "",
              C.md_table(show, [("Controller", "arm"), ("seeds", "seeds"),
                                ("RMSE mean ± SD [m]", "msd"), ("Median / P90 [m]", "mp"),
                                ("Sat. [%]", "sat"), ("term. [%]", "term"),
                                ("reduction [%]", "red"), ("roughness", "rough")]), ""]

    # Table 2: LPF ablation
    pairs = [("original code, paper policies", "PID_paper_nolpf", "PID_paper_lpf"),
             ("paper policies flown with every fix", "PID_paperfx_nolpf", "PID_paperfx_lpf"),
             ("retrained with every fix (τ 0.5, α 0.2)", "PID_fixed_nolpf", "PID_fixed_lpf"),
             ("sharp reward (τ 0.05, α 0.2)", "sharp_nolpf", "sharp_lpf"),
             ("sharp reward + authority 0.6", "hi_nolpf", "hi_lpf"),
             ("PD, original code, paper policies", "PD_paper_nolpf", "PD_paper_lpf"),
             ("PD, retrained with every fix", "PD_fixed_nolpf", "PD_fixed_lpf")]
    lrows = []
    for name, k0, k1 in pairs:
        rows, _ = table_stats(A, [k0, k1])
        if len(rows) < 2:
            continue
        for r, lpf in zip(rows, ("no", "yes")):
            lrows.append({"setting": name, "lpf": lpf, **r})
    rows_all["table2_lpf"] = lrows
    C.write_csv(out / "table2_lpf_ablation.csv", lrows)
    show = [{"setting": r["setting"] if r["lpf"] == "no" else "", "lpf": r["lpf"],
             "med": pm(r["median"], r["median_sd"]), "p90": pm(r["p90"], r["p90_sd"]),
             "sat": pm(r["sat_pct"], r["sat_pct_sd"], "{:.2f}"),
             "rough": pm(r["roughness"], r["roughness_sd"], "{:.3f}"),
             "hf": pm(r["motor_hf"], r["motor_hf_sd"], "{:.3f}"),
             "max": f"{r['motor_hf_max']:.2f}", "ext": f"{r['n_extreme']:.1f}",
             "slew": f"{r['res_f_slew']:.1f}", "term": f"{r['term_pct']:.2f}",
             "own": f"{r['own_fail']:.1f}"} for r in lrows]
    L += ["## Table 2 — LPF ablation (thrust LPF on/off, same seed)", "",
          f"`worst HF` = largest per-episode motor-command HF RMS (>10 Hz) of 512 episodes; "
          f"`extreme eps` = episodes above {EXTREME_N} N (≈3× the original baseline's "
          "level), per seed. Seed mean ± SD.", "",
          C.md_table(show, [("setting", "setting"), ("LPF", "lpf"), ("median [m]", "med"),
                            ("P90 [m]", "p90"), ("sat [%]", "sat"), ("roughness", "rough"),
                            ("motor HF [N]", "hf"), ("worst HF [N]", "max"),
                            ("extreme eps", "ext"), ("thrust slew [N/s]", "slew"),
                            ("term [%]", "term"), ("own fails", "own")]), ""]

    # Table 3: bug ablation (no retraining)
    s = A.best_seed if A.best_seed in A.seeds else A.seeds[0]
    brows = []
    for v in ["original"] + list(C.SINGLE_FIXES) + ["all_fixes"]:
        ov = variant_overrides(v)
        r = {"fix": C.FIX_LABELS[v], "key": v}
        for base in ("pid", "pd"):
            mb = C.cached_fly(A.a.results, f"{base} | {v}", C.paper_env(base_controller=base, **ov),
                              **A.kw)
            r[f"{base}_median"] = float(np.median(mb["rmse_alive"][mb["terminated"] < 0.5]))
            r[f"{base}_rough"] = float(np.mean(mb["roughness"][mb["terminated"] < 0.5]))
            r[f"{base}_hf"] = float(np.mean(mb["motor_hf"][mb["terminated"] < 0.5]))
            r[f"{base}_term"] = 100 * float(np.mean(mb["terminated"] > 0.5))
            for tag, sfx in (("none", "nolpf"), ("thrust", "lpf")):
                run = A.pol.get((s, tag, base))
                if run is None:
                    continue
                mp = C.cached_fly(A.a.results, f"paper s{s} {base} {sfx} | {v}",
                                  C.env_from_run(run, **ov), run_dir=run, **A.kw)
                alive = mp["terminated"] < 0.5
                r[f"{base}_{sfx}_median"] = float(np.median(mp["rmse_alive"][alive]))
                r[f"{base}_{sfx}_ext"] = int(np.sum(mp["motor_hf"][alive] > EXTREME_N))
                r[f"{base}_{sfx}_max"] = float(np.max(mp["motor_hf"][alive]))
        brows.append(r)
    o = brows[0]
    for r in brows:
        for base in ("pid", "pd"):
            r[f"{base}_dmed"] = 100 * (r[f"{base}_median"] / o[f"{base}_median"] - 1)
            r[f"{base}_drough"] = 100 * (r[f"{base}_rough"] / o[f"{base}_rough"] - 1)
            r[f"{base}_dhf"] = 100 * (r[f"{base}_hf"] / o[f"{base}_hf"] - 1)
            if f"{base}_lpf_median" in r:
                r[f"{base}_lpf_dmed"] = 100 * (r[f"{base}_lpf_median"] / o[f"{base}_lpf_median"] - 1)
    rows_all["table3_bugs"] = brows
    C.write_csv(out / "table3_bug_ablation.csv", brows)
    f = {k: "{:+.1f}%" for k in ("pid_dmed", "pid_drough", "pid_dhf", "pid_lpf_dmed")}
    f.update({"pid_nolpf_max": "{:.2f}", "pd_nolpf_max": "{:.2f}"})
    L += [f"## Table 3 — each bug fix alone, no retraining (paper seed {s})", "",
          f"Baseline change against the original code, and the paper's unfiltered policies' "
          f"extreme-chatter episodes (> {EXTREME_N} N) and worst episode under each fix.", "",
          C.md_table(brows, [("fix", "fix"), ("PID Δmedian", "pid_dmed"),
                             ("PID Δroughness", "pid_drough"), ("PID Δmotor HF", "pid_dhf"),
                             ("paper PID+RRL Δmedian", "pid_lpf_dmed"),
                             ("PID no-LPF extreme eps", "pid_nolpf_ext"),
                             ("PID no-LPF worst [N]", "pid_nolpf_max"),
                             ("PD no-LPF extreme eps", "pd_nolpf_ext"),
                             ("PD no-LPF worst [N]", "pd_nolpf_max")], f), ""]
    return L, rows_all


# ------------------------------------------------------------------ rollouts with positions
_TR = {}


def _trace_fn(A, key, seed):
    sp = A.spec[key]
    env_cfg = sp["envs"][seed]
    run = sp["runs"].get(seed)
    chunk = int(min(A.a.chunk, A.a.episodes))
    agent, params = (None, None) if run is None else C.load_policy(run)[1:]
    sig = (repr(sorted(env_cfg.__dict__.items())), agent is not None, chunk)
    if sig not in _TR:
        env = C.VecEnv(env_cfg, chunk)
        ranges = env.broadcast_ranges(C.flat_ranges(env_cfg, force_freq=0.0, force_dc_prob=0.0))

        def run_fn(params, keys):
            state = env._reset(keys, ranges, jnp.zeros(chunk, jnp.int32))
            obs = env._reset_obs(state)

            def body(c, _):
                st, ob = c
                act = (jnp.zeros((chunk, 4)) if agent is None
                       else jnp.tanh(agent.actor.apply(params, ob)[0]))
                st, ob, _, _, info = env.step(st, act, ranges)
                return (st, ob), (info["pos_err_desired"], info["x_true"], info["x_des"],
                                  info["motor_cmd"], info["terminated"])
            _, tr = jax.lax.scan(body, (state, obs), None, length=env_cfg.episode_steps)
            return tr
        _TR[sig] = jax.jit(run_fn)
    return _TR[sig], params, chunk


def fly_positions(A, key, seed, ids):
    fn, params, chunk = _trace_fn(A, key, seed)
    ids = [int(i) for i in ids]
    out = {n: [] for n in ("err", "x", "xd", "motor", "term")}
    for c0 in range(0, len(ids), chunk):
        part = ids[c0:c0 + chunk]
        kk = C.bank_keys(A.a.eval_seed, part + [part[-1]] * (chunk - len(part)))
        tr = [np.asarray(t)[:, :len(part)] for t in fn(params, kk)]
        for n, t in zip(out, tr):
            out[n].append(t)
    return {n: np.concatenate(v, axis=1) for n, v in out.items()}


# ------------------------------------------------------------------ figures
PALETTE = {"PID_fixed": BASE_GRAY, "PID_orig": BASE_GRAY, "PD_fixed": BASE_GRAY,
           "PD_orig": BASE_GRAY, "PID_paper_lpf": SLOT[0], "PD_paper_lpf": SLOT[0],
           "PID_fixed_lpf": SLOT[1], "PD_fixed_lpf": SLOT[1], "D1": SLOT[2], "D2": SLOT[3]}
SHORT = {"PID_orig": "PID", "PID_fixed": "PID", "PD_orig": "PD", "PD_fixed": "PD",
         "PID_paper_lpf": "Paper", "PD_paper_lpf": "Paper", "PID_fixed_lpf": "Fixed",
         "PD_fixed_lpf": "Fixed", "D1": "D1", "D2": "D2"}


def fig_tracking(A, rows_all, out):
    panels = [("PID", ["PID_fixed", "PID_paper_lpf", "PID_fixed_lpf", "D1", "D2"]),
              ("PD", ["PD_fixed", "PD_paper_lpf", "PD_fixed_lpf"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [5, 3]})
    for ax, (base, keys) in zip(axes, panels):
        rows = {r["key"]: r for r in rows_all[f"table1_{base}"]}
        keys = [k for k in keys if k in rows]
        x = np.arange(len(keys))
        w = 0.36
        for i, k in enumerate(keys):
            r = rows[k]
            c = PALETTE[k]
            ax.bar(i - w / 2 - 0.01, r["median"] * 100, w, color=c, yerr=r["median_sd"] * 100,
                   error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
            ax.bar(i + w / 2 + 0.01, r["p90"] * 100, w, color=c, alpha=0.45, hatch="////",
                   edgecolor=c, linewidth=0, yerr=r["p90_sd"] * 100,
                   error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
            ax.text(i - w / 2, r["median"] * 100, f"{r['median'] * 100:.2f}", ha="center",
                    va="bottom", fontsize=8, color=INK)
            ax.text(i + w / 2, r["p90"] * 100, f"{r['p90'] * 100:.2f}", ha="center",
                    va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, [SHORT[k] for k in keys])
        ax.set_ylabel("position RMSE [cm]")
        ax.set_title(f"{base} base")
        ax.grid(axis="x", visible=False)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INK2),
                            plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.45, hatch="////")],
                   labels=["median", "P90"], loc="upper right")
    fig.text(0.01, -0.04, "Paper = paper checkpoints (original code); Fixed = retrained with "
             "every fix, paper reward; D1 = τ_pos 0.1, α 0.2; D2 = τ_pos 0.2, α 0.4. "
             "Bars: seed mean, whiskers: seed SD.", fontsize=8, color=INK2)
    return save(fig, out, "fig1_tracking")


def fig_distribution(A, out):
    keys = [k for k in ["PID_fixed", "PID_paper_lpf", "PID_fixed_lpf", "D1", "D2"] if A.has(k)]
    data = {k: [] for k in keys}
    for s in A.seeds_of(keys):
        mm = {k: A.m(k, s) for k in keys}
        joint = C.joint_survivors(mm)
        for k in keys:
            data[k].append(mm[k]["rmse_alive"][joint] * 100)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    vals = [np.concatenate(data[k]) for k in keys]
    bp = ax.boxplot(vals, widths=0.55, patch_artist=True, showfliers=True,
                    medianprops={"color": INK, "lw": 1.5},
                    flierprops={"marker": "o", "ms": 2.5, "alpha": 0.4, "mec": "none"})
    for patch, flier, k in zip(bp["boxes"], bp["fliers"], keys):
        patch.set_facecolor(PALETTE[k])
        patch.set_alpha(0.85)
        patch.set_edgecolor(SURFACE)
        flier.set_markerfacecolor(PALETTE[k])
    for i, v in enumerate(vals):
        ax.text(i + 1, np.median(v) * 1.08, f"{np.median(v):.2f}", ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.set_yscale("log")
    plain_log(ax)
    ax.set_xticks(range(1, len(keys) + 1), [SHORT[k] for k in keys])
    ax.set_ylabel("per-episode position RMSE [cm]")
    ax.set_title("Tracking error over the uncertainty box (PID base, 3 seeds pooled)")
    ax.grid(axis="x", visible=False)
    return save(fig, out, "fig2_rmse_distribution")


def fig_error_time(A, out):
    keys = [k for k in ["PID_fixed", "PID_paper_lpf", "PID_fixed_lpf", "D1", "D2"] if A.has(k)]
    seeds = A.seeds_of(keys)
    ids = list(range(A.a.episodes))
    series = {k: [] for k in keys}
    for s in seeds:
        tr = {k: fly_positions(A, k, s, ids) for k in keys}
        alive = np.ones(len(ids), bool)
        for k in keys:
            alive &= ~tr[k]["term"].any(axis=0)
        for k in keys:
            series[k].append(tr[k]["err"][:, alive])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    dt = 1.0 / C.PAPER_HZ
    for ax, (q, title) in zip(axes, ((50, "median episode"), (90, "90th-percentile episode"))):
        for k in keys:
            e = np.concatenate(series[k], axis=1) * 100
            t = np.arange(e.shape[0]) * dt
            ax.plot(t, np.percentile(e, q, axis=1), color=PALETTE[k], lw=2, label=SHORT[k])
        ax.set_yscale("log")
        plain_log(ax)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("time [s]")
        ax.set_title(f"Position error over the episode: {title}")
    axes[0].set_ylabel("position error [cm]")
    axes[1].legend(fontsize=9, loc="upper right", ncol=len(keys))
    fig.tight_layout()
    return save(fig, out, "fig3_error_vs_time")


def fig_case(A, out, case=None):
    keys = [k for k in ["PID_fixed", "PID_paper_lpf", "D1", "D2"] if A.has(k)]
    s = A.seeds_of(keys)[0]
    mm = {k: A.m(k, s) for k in keys}
    joint = C.joint_survivors(mm)
    force = mm[keys[0]]["force"]
    if case is None:
        case = int(np.argmax(np.where(joint, force, -1)))
    tr = {k: fly_positions(A, k, s, [case]) for k in keys}
    fig = plt.figure(figsize=(12, 4.4))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    xd = tr[keys[0]]["xd"][:, 0]
    ax3.plot(xd[:, 0], xd[:, 1], -xd[:, 2], color=INK, lw=1.2, ls="--", label="reference")
    for k in keys:
        x = tr[k]["x"][:, 0]
        ax3.plot(x[:, 0], x[:, 1], -x[:, 2], color=PALETTE[k], lw=1.6, label=SHORT[k])
    ax3.set_xlabel("N [m]")
    ax3.set_ylabel("E [m]")
    ax3.set_zlabel("up [m]")
    ax3.legend(loc="upper left", fontsize=8)
    d = mm[keys[0]]
    ax3.set_title(f"episode {case}: |F| = {force[case]:.2f} N, mass×{d['k'][case]:.2f}, "
                  f"kf×{d['kf'][case]:.2f}, arm×{d['arm'][case]:.2f}", fontsize=9)
    ax = fig.add_subplot(1, 2, 2)
    t = np.arange(xd.shape[0]) / C.PAPER_HZ
    for k in keys:
        e = tr[k]["err"][:, 0] * 100
        ax.plot(t, e, color=PALETTE[k], lw=1.6, label=SHORT[k])
    ax.legend(fontsize=9, loc="upper right", ncol=len(keys))
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [cm]")
    ax.set_title(f"tracking error (seed {s})")
    fig.tight_layout()
    return save(fig, out, f"fig4_trajectory_case_ep{case}")


def fig_lpf_story(A, out):
    rows = []
    # rows 1-2: worst unfiltered paper episode, original code vs analytic rates only
    best = None
    for s in A.seeds:
        if A.has("PID_paper_nolpf", s):
            m = A.m("PID_paper_nolpf", s)
            hf = np.where(m["terminated"] < 0.5, m["motor_hf"], -1)
            e = int(np.argmax(hf))
            if best is None or hf[e] > best[2]:
                best = (s, e, hf[e])
    if best:
        s, e, _ = best
        rows.append((f"Original code, paper policies: worst of {A.a.episodes} episodes "
                     f"(seed {s}, episode {e})",
                     [("PID_paper_nolpf", "no LPF"), ("PID_paper_lpf", "thrust LPF")], s, e))
        if A.has("PID_paperan_nolpf", s):
            rows.append(("Same policies and episode, analytic desired rates only "
                         "(no retraining)",
                         [("PID_paperan_nolpf", "no LPF"), ("PID_paperan_lpf", "thrust LPF")],
                         s, e))
    # row 3: high authority, the seed of middle severity and its median-chatter episode
    cand = []
    for s in A.seeds:
        if A.has("hi_nolpf", s) and A.has("hi_lpf", s):
            m = A.m("hi_nolpf", s)
            alive = m["terminated"] < 0.5
            cand.append((float(np.mean(m["motor_hf"][alive])), s))
    if cand:
        cand.sort()
        _, s = cand[len(cand) // 2]
        m = A.m("hi_nolpf", s)
        alive = m["terminated"] < 0.5
        order = np.argsort(np.where(alive, m["motor_hf"], np.inf))
        e = int(order[alive.sum() // 2])
        rows.append((f"Fixed code, τ_pos 0.05, authority 0.6: typical episode (median chatter, "
                     f"seed {s}, episode {e})",
                     [("hi_nolpf", "no LPF"), ("hi_lpf", "thrust LPF")], s, e))
    if not rows:
        return None
    fig = plt.figure(figsize=(12, 2.8 * len(rows)), layout="constrained")
    subs = np.atleast_1d(fig.subfigures(len(rows), 1))
    for i, (sub, (title, pair, s, e)) in enumerate(zip(subs, rows)):
        sub.set_facecolor(SURFACE)
        axes = sub.subplots(1, 2, sharey=True)
        sub.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
        for ax, (k, lab) in zip(axes, pair):
            if not A.has(k, s):
                ax.set_visible(False)
                continue
            mot = fly_positions(A, k, s, [e])["motor"][:, 0]
            t = np.arange(mot.shape[0]) / C.PAPER_HZ
            for r in range(4):
                ax.plot(t, mot[:, r], color=SLOT[r], lw=0.7)
            hf = A.m(k, s)["motor_hf"][e]
            ax.set_title(f"{lab}   (motor HF {hf:.2f} N)", fontsize=9, loc="left")
            ax.set_xlim(0, t[-1])
            if i == len(rows) - 1:
                ax.set_xlabel("time [s]")
        axes[0].set_ylabel("rotor cmd [N]")
        if i == 0:
            axes[1].legend(handles=[Line2D([], [], color=SLOT[r], lw=1.5) for r in range(4)],
                           labels=[f"rotor {r}" for r in range(4)], ncol=4, fontsize=8,
                           loc="upper right")
    return save(fig, out, "fig5_lpf_rotor_commands")


def fig_chatter_cdf(A, out):
    curves = [("PID_fixed", BASE_GRAY, "-", "PID baseline (fixed code)"),
              ("PID_paper_nolpf", SLOT[0], "--", "paper, original code, no LPF"),
              ("PID_paper_lpf", SLOT[0], "-", "paper, original code, LPF"),
              ("PID_paperfx_nolpf", SLOT[2], "--", "paper policy + fixed physics, no LPF"),
              ("PID_fixed_nolpf", SLOT[1], "--", "retrained fixed, no LPF"),
              ("PID_fixed_lpf", SLOT[1], "-", "retrained fixed, LPF")]
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, c, ls, lab in curves:
        if not A.has(k):
            continue
        v = np.concatenate([A.m(k, s)["motor_hf"][A.m(k, s)["terminated"] < 0.5]
                            for s in A.seeds if A.has(k, s)])
        v = np.sort(v[np.isfinite(v)])
        n_ext = int(np.sum(v > EXTREME_N))
        ax.plot(v, 1 - np.arange(v.size) / v.size, color=c, ls=ls, lw=1.8,
                label=f"{lab}  ({n_ext} eps > {EXTREME_N} N)")
    ax.axvline(EXTREME_N, color=MUTED, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log(ax, "x")
    plain_log(ax, "y")
    ax.set_xlabel("per-episode motor-command HF RMS > 10 Hz [N]")
    ax.set_ylabel("fraction of episodes above x")
    ax.set_title("Chatter tail (PID base, all seeds pooled)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    return save(fig, out, "fig6_chatter_tail")


def fig_bug_ablation(rows, out):
    names = [r["fix"] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    ax = axes[0]
    h = 0.26
    for j, (f, lab, c) in enumerate((("pid_dmed", "median RMSE", INK2),
                                     ("pid_drough", "input roughness", SLOT[0]),
                                     ("pid_dhf", "motor HF", SLOT[1]))):
        v = [r[f] for r in rows]
        ax.barh(y + (1 - j) * h, v, h * 0.92, color=c, label=lab)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlabel("change of the PID baseline vs original code [%]")
    ax.set_title("Baseline: tracking unchanged, commands smoother")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    allv = [r.get(f, np.nan) for r in rows for f in ("pd_nolpf_ext", "pid_nolpf_ext")]
    xmax = max([1.0] + [v for v in allv if np.isfinite(v)]) * 1.15
    for j, (f, lab, c) in enumerate((("pd_nolpf_ext", "PD+RRL, no LPF", SLOT[0]),
                                     ("pid_nolpf_ext", "PID+RRL, no LPF", SLOT[1]))):
        v = [r.get(f, np.nan) for r in rows]
        ax.barh(y + (0.5 - j) * 0.36, v, 0.33, color=c, label=lab)
        for yy, vv in zip(y + (0.5 - j) * 0.36, v):
            if np.isfinite(vv):
                ax.text(vv + 0.01 * xmax, yy, f"{vv:.0f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(f"paper policies: episodes with motor HF > {EXTREME_N} N (of 512)")
    ax.set_title("Unfiltered residual: extreme chatter episodes")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, out, "fig7_bug_ablation")


def fig_sweep(A, out):
    from .s2_train import AUTHS, TAUS
    pts = []
    for auth in AUTHS:
        for tau in TAUS:
            job = ("A_pid_lpf" if (tau, auth) == (0.5, 0.2)
                   else f"B_pid_lpf_tau{tau:g}_auth{auth:g}")
            pts.append((tau, auth, True, job))
    pts += [(0.05, 0.2, False, "C_pid_nolpf_tau0.05_auth0.2"),
            (0.05, 0.6, False, "C_pid_nolpf_tau0.05_auth0.6")]
    res = []
    for tau, auth, lpf, job in pts:
        st = []
        for s in A.seeds:
            run = finished_run(A.a.results, f"{job}_s{s}")
            if run is None:
                continue
            m = C.cached_fly(A.a.results, f"{job}_s{s}", C.env_from_run(run), run_dir=run, **A.kw)
            b = A.m("PID_fixed", s)
            alive = m["terminated"] < 0.5
            st.append((float(np.median(m["rmse_charged"])),
                       float(np.percentile(m["rmse_charged"], 90)),
                       float(np.mean(m["sat_pct"][alive])), float(np.mean(m["motor_hf"][alive])),
                       float(np.nanmean(m["res_f_slew"][alive])), C.own_failures(m, b)))
        if st:
            v = np.mean(st, axis=0)
            res.append({"tau_pos": tau, "L_cm": 25 * tau, "authority": auth, "lpf": lpf,
                        "seeds": len(st), "median": v[0], "p90": v[1], "sat_pct": v[2],
                        "motor_hf": v[3], "res_f_slew": v[4], "own_fail": v[5]})
    C.write_csv(out / "table4_sweep.csv", res)
    if not res:
        return None, res
    bs = [A.m("PID_fixed", s) for s in A.seeds]
    alive_b = [x["terminated"] < 0.5 for x in bs]
    base = {"median": np.mean([np.median(x["rmse_charged"]) for x in bs]),
            "p90": np.mean([np.percentile(x["rmse_charged"], 90) for x in bs]),
            "sat_pct": np.mean([np.mean(x["sat_pct"][al]) for x, al in zip(bs, alive_b)]),
            "motor_hf": np.mean([np.mean(x["motor_hf"][al]) for x, al in zip(bs, alive_b)]),
            "res_f_slew": None, "own_fail": None}
    panels = [("median", "median RMSE [cm]", 100, True), ("p90", "P90 RMSE [cm]", 100, True),
              ("own_fail", "episodes lost that PID flew (of 512)", 1, False),
              ("sat_pct", "saturated steps [%]", 1, True), ("motor_hf", "motor HF > 10 Hz [N]", 1, True),
              ("res_f_slew", "thrust-residual slew [N/s]", 1, True)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    xs = {t: i for i, t in enumerate(TAUS)}
    for ax, (f, lab, scale, log) in zip(axes.ravel(), panels):
        for auth in AUTHS:
            r = sorted([x for x in res if x["authority"] == auth and x["lpf"]],
                       key=lambda x: xs[x["tau_pos"]])
            if r:
                ax.plot([xs[x["tau_pos"]] for x in r], [x[f] * scale for x in r], "-o",
                        color=AUTH_RAMP[auth], ms=6, mec=SURFACE, mew=1.2, label=f"α {auth:g}, LPF")
            for x in res:
                if x["authority"] == auth and not x["lpf"]:
                    ax.plot(xs[x["tau_pos"]] + 0.14, x[f] * scale, "X", color=AUTH_RAMP[auth],
                            ms=9, mec=SURFACE, mew=0.8, ls="")
        for tag, tau, auth in (("D1", 0.1, 0.2), ("D2", 0.2, 0.4)):
            r = [x for x in res if x["tau_pos"] == tau and x["authority"] == auth and x["lpf"]]
            if r:
                ax.plot(xs[tau], r[0][f] * scale, "o", ms=15, mfc="none", mec=INK, mew=1.2)
                ax.annotate(tag, (xs[tau], r[0][f] * scale), xytext=(9, 7),
                            textcoords="offset points", fontsize=9, fontweight="bold", color=INK)
        if base.get(f) is not None:
            ax.axhline(base[f] * scale, color=BASE_GRAY, ls="--", lw=1.2)
            ax.annotate("PID", (0, base[f] * scale), xytext=(-4, 4), textcoords="offset points",
                        fontsize=8, color=INK2, ha="left")
        if log:
            ax.set_yscale("log")
            plain_log(ax)
        ax.set_xticks(range(len(TAUS)), [f"{t:g}\n({25 * t:g} cm)" for t in TAUS])
        ax.set_xlim(-0.3, len(TAUS) - 0.55)
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("τ_pos (position-reward length scale), sharper →")
    handles = [Line2D([], [], color=AUTH_RAMP[x], marker="o", ms=6, label=f"authority {x:g}, LPF")
               for x in AUTHS]
    handles += [Line2D([], [], color=INK2, marker="X", ms=8, ls="", label="no LPF (τ 0.05)"),
                Line2D([], [], color=BASE_GRAY, ls="--", label="PID baseline")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Reward sharpness × residual authority (PID + RRL, every fix on, 3-seed mean)",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, out, "fig8_sweep_tradeoff"), res


def fig_authority_lpf(rows_all, out):
    lrows = [r for r in rows_all["table2_lpf"] if r["setting"].startswith("sharp reward")]
    if len(lrows) < 4:
        return None
    groups = [("α 0.2", "sharp reward (τ 0.05, α 0.2)"),
              ("α 0.6", "sharp reward + authority 0.6")]
    metrics = [("sat_pct", "saturated steps [%]"), ("motor_hf", "motor HF [N]"),
               ("own_fail", "episodes lost that PID flew"), ("median", "median RMSE [cm]")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (f, lab) in zip(axes, metrics):
        for gi, (gname, setting) in enumerate(groups):
            for li, lpf in enumerate(("no", "yes")):
                r = [x for x in lrows if x["setting"] == setting and x["lpf"] == lpf]
                if not r:
                    continue
                v = r[0][f] * (100 if f == "median" else 1)
                c = AUTH_RAMP[0.2 if gi == 0 else 0.6]
                ax.bar(gi + (li - 0.5) * 0.38, v, 0.36, color=c,
                       hatch="" if lpf == "yes" else "////", edgecolor=SURFACE if lpf == "yes"
                       else INK2, linewidth=0.6 if lpf == "no" else 0)
                ax.text(gi + (li - 0.5) * 0.38, v, f"{v:.2f}" if v < 10 else f"{v:.0f}",
                        ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks([0, 1], [g for g, _ in groups])
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, hatch="////",
                                          edgecolor=INK2),
                            plt.Rectangle((0, 0), 1, 1, facecolor=MUTED)],
                   labels=["no LPF", "thrust LPF"], fontsize=8, loc="upper left")
    fig.suptitle("Sharp reward (τ_pos 0.05): the thrust LPF matters once authority is raised",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, out, "fig9_authority_lpf")


def fig_reward_basin(A, out):
    """Per-episode RRL error against the PID error on the same episode."""
    keys = [k for k in ("PID_fixed_lpf", "D1", "D2") if A.has(k)]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(4.3 * len(keys), 4.2), sharex=True,
                             sharey=True, squeeze=False)
    axes = axes[0]
    lim = (0.3, 150.0)
    for ax, k in zip(axes, keys):
        xs, ys = [], []
        for s in A.seeds:
            if not A.has(k, s):
                continue
            b, m = A.m("PID_fixed", s), A.m(k, s)
            ok = (b["terminated"] < 0.5) & (m["terminated"] < 0.5)
            xs.append(b["rmse_alive"][ok] * 100)
            ys.append(m["rmse_alive"][ok] * 100)
        x, y = np.concatenate(xs), np.concatenate(ys)
        ax.scatter(x, y, s=8, color=PALETTE[k], alpha=0.5, lw=0)
        bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 14)
        mid, med = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (x >= lo) & (x < hi)
            if sel.sum() >= 8:
                mid.append(np.sqrt(lo * hi))
                med.append(np.median(y[sel]))
        ax.plot(mid, med, color=INK, lw=1.8, label="binned median")
        ax.plot(lim, lim, color=MUTED, lw=1, ls="--", label="no better than PID")
        env = next(iter(A.spec[k]["envs"].values()))
        L_cm = 25.0 * env.tau_pos
        ax.axhline(L_cm, color=INK2, lw=1, ls=":")
        ax.text(lim[0] * 1.15, L_cm * 1.12, f"reward length scale {L_cm:g} cm", fontsize=8,
                color=INK2)
        n5 = int(np.sum(y > 5))
        ax.set_title(f"{SHORT[k]}: τ_pos {env.tau_pos:g}, authority {env.residual_authority:g}")
        ax.text(0.97, 0.04, f"{n5} of {y.size} episodes > 5 cm", transform=ax.transAxes,
                ha="right", fontsize=9, color=INK)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log(ax, "x")
        plain_log(ax, "y")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel("PID RMSE on the same episode [cm]  (episode difficulty)")
    axes[0].set_ylabel("PID + RRL + LPF RMSE [cm]")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return save(fig, out, "fig10_reward_basin")


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="study_results")
    p.add_argument("--paper_dir", default="auto")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--case", type=int, default=None,
                   help="episode for the trajectory figure (default: largest force)")
    a = p.parse_args()
    out = C.results_dir(Path(a.results) / "04_presentation")
    A = Arms(a)
    L = ["# 04 — Presentation tables and figures", "",
         "Arms: **Paper** = the paper's checkpoints on the original code; **Fixed** = "
         "retrained with every fix, paper reward (τ_pos 0.5) and authority 0.2; **D1** = "
         "fixed, τ_pos 0.1, authority 0.2; **D2** = fixed, τ_pos 0.2, authority 0.4. "
         f"Common bank of {a.episodes} episodes (eval seed {a.eval_seed}); statistics on "
         "the episodes every arm of a table survives, mean over seeds.", ""]
    T, rows_all = write_tables(A, out)
    figs = []
    steps = [("tracking bars", lambda: fig_tracking(A, rows_all, out)),
             ("RMSE distribution", lambda: fig_distribution(A, out)),
             ("error vs time", lambda: fig_error_time(A, out)),
             ("trajectory case", lambda: fig_case(A, out, a.case)),
             ("LPF rotor commands", lambda: fig_lpf_story(A, out)),
             ("chatter tail", lambda: fig_chatter_cdf(A, out)),
             ("bug ablation", lambda: fig_bug_ablation(rows_all["table3_bugs"], out)),
             ("sweep trade-off", lambda: fig_sweep(A, out)[0]),
             ("authority x LPF", lambda: fig_authority_lpf(rows_all, out)),
             ("reward basin", lambda: fig_reward_basin(A, out))]
    for name, fn in steps:
        try:
            f = fn()
            if f:
                figs.append(f)
                print(f"  figure: {f}", flush=True)
        except Exception as e:                               # noqa: BLE001
            print(f"  figure '{name}' FAILED: {type(e).__name__}: {e}", flush=True)
            L.append(f"- figure '{name}' failed: `{type(e).__name__}: {e}`")
    L += T + ["## Figures (PNG at 200 dpi, PDF alongside)", ""] + \
        [f"![]({f})" for f in figs] + [""]
    (out / "FIGURES.md").write_text("\n".join(L) + "\n")
    print((out / "FIGURES.md").read_text())


if __name__ == "__main__":
    main()
