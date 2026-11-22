"""Stage 3 -- evaluate every finished training run on the common bank.

02_retrain_fixed  (Q1 + Q2): policies retrained with every fix, against the
    paper's policies on the original code, seed by seed; LPF effect and the
    chatter tail ("does the unfiltered residual still oscillate badly?").
03_reward_sweep   (Q3): tau_pos x authority grid for PID + LPF, plus the
    no-LPF arms at the sharp reward.

Statistics are computed per seed and then averaged over seeds (mean ± SD of
the seed-level numbers); the paper's single-best-seed selection is not used.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import common as C
from .s1_ablation import paper_dir_from
from .s2_train import AUTHS, TAUS, finished_run, job_list

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import matplotlib.ticker                             # noqa: E402

ARM_NAMES = {("pd", None): "PD", ("pid", None): "PID",
             ("pd", 1.0): "PD+RRL", ("pid", 1.0): "PID+RRL",
             ("pd", 0.2): "PD+RRL+LPF", ("pid", 0.2): "PID+RRL+LPF"}


def mean_sd(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


def pm(m, s, n, f="{:.4f}"):
    if not np.isfinite(m):
        return "–"
    return (f.format(m) + (" ± " + f.format(s) if n > 1 else "") + f"  (n={n})")


def trio_stats(base_m, arms):
    """Seed-level statistics for {baseline, RRL, RRL+LPF} on joint survivors."""
    joint = C.joint_survivors({"base": base_m, **arms})
    out = {"base": {**C.summarize(base_m, joint), "own_fail": 0}}
    for k, m in arms.items():
        out[k] = {**C.summarize(m, joint), "own_fail": C.own_failures(m, base_m),
                  **C.chatter_tail(m, base_m, joint)}
    sb = out["base"]
    for k in arms:
        s = out[k]
        s["rrl_median_red_pct"] = 100 * (1 - s["cond_median"] / sb["cond_median"])
        s["rrl_mean_red_pct"] = 100 * (1 - s["cond_mean"] / sb["cond_mean"])
        s["rrl_p90_red_pct"] = 100 * (1 - s["cond_p90"] / sb["cond_p90"])
    if "nolpf" in arms and "lpf" in arms:
        s0, s1 = out["nolpf"], out["lpf"]
        out["lpf_effect"] = {
            "roughness_red_pct": 100 * (1 - s1["roughness"] / s0["roughness"]),
            "sat_red_pct": 100 * (1 - s1["sat_pct"] / s0["sat_pct"]) if s0["sat_pct"] > 0 else np.nan,
            "median_red_pct": 100 * (1 - s1["cond_median"] / s0["cond_median"]),
            "motor_hf_red_pct": 100 * (1 - s1["motor_hf"] / s0["motor_hf"]),
            "res_f_hf_red_pct": 100 * (1 - s1["res_f_hf"] / s0["res_f_hf"])
            if s0["res_f_hf"] > 0 else np.nan,
            "n_cond": s0["n_cond"]}
    return out


# ====================================================================== Q1 + Q2
def retrain_section(a, kw, out):
    paper_dir = paper_dir_from(a.results, a.paper_dir)
    pol = C.paper_policies(paper_dir) if paper_dir else {}
    fixed = C.all_fixes()
    seeds = [int(s) for s in a.seeds.split(",")]
    per_seed = defaultdict(dict)        # (code, base) -> seed -> trio stats
    pooled = defaultdict(lambda: defaultdict(list))   # (code, base, arm) -> metric lists
    worst = {}
    for base in ("pd", "pid"):
        bm = {"original": C.cached_fly(a.results, f"{base.upper()} | original",
                                       C.paper_env(base_controller=base), **kw),
              "fixed": C.cached_fly(a.results, f"{base.upper()} | all_fixes",
                                    C.paper_env(base_controller=base, **fixed), **kw)}
        for code in ("original", "fixed"):
            for s in seeds:
                arms = {}
                for tag, key, beta in (("none", "nolpf", 1.0), ("thrust", "lpf", 0.2)):
                    if code == "original":
                        run = pol.get((s, tag, base))
                    else:
                        run = finished_run(a.results, f"A_{base}_{key}_s{s}")
                    if run is None:
                        continue
                    arms[key] = C.cached_fly(
                        a.results, f"{code} s{s} {ARM_NAMES[(base, beta)]}",
                        C.env_from_run(run), run_dir=run, **kw)
                if not arms:
                    continue
                C.check_same_draws({"base": bm[code], **arms})
                st = trio_stats(bm[code], arms)
                per_seed[(code, base)][s] = st
                for k, m in [("base", bm[code])] + list(arms.items()):
                    pooled[(code, base, k)]["motor_hf"].append(
                        m["motor_hf"][m["terminated"] < 0.5])
                if code == "fixed" and "nolpf" in arms and s == seeds[0]:
                    ok = (arms["nolpf"]["terminated"] < 0.5) & (bm[code]["terminated"] < 0.5)
                    ratio = np.where(ok, arms["nolpf"]["motor_hf"]
                                     / np.maximum(bm[code]["motor_hf"], 1e-6), -1)
                    worst[base] = (int(np.argmax(ratio)), float(ratio.max()), s)

    rows, lpf_rows = [], []
    for (code, base), by_seed in sorted(per_seed.items()):
        for key in ("base", "nolpf", "lpf"):
            vals = [st[key] for st in by_seed.values() if key in st]
            if not vals:
                continue
            r = {"code": code, "arm": ARM_NAMES[(base, {"base": None, "nolpf": 1.0,
                                                         "lpf": 0.2}[key])],
                 "seeds": len(vals)}
            for f in ("term_pct", "cond_median", "cond_p90", "cond_mean", "sat_pct",
                      "roughness", "motor_hf", "motor_hf_max", "res_f_hf", "res_f_slew",
                      "own_fail", "rrl_median_red_pct", "rrl_p90_red_pct", "n_gt3x",
                      "n_gt10x", "hf_ratio_max"):
                m, sd, n = mean_sd([v.get(f) for v in vals])
                r[f] = m
                r[f + "_sd"] = sd
            rows.append(r)
        effs = [st["lpf_effect"] for st in by_seed.values() if "lpf_effect" in st]
        if effs:
            r = {"code": code, "base": base.upper(), "seeds": len(effs)}
            for f in ("roughness_red_pct", "sat_red_pct", "median_red_pct",
                      "motor_hf_red_pct", "res_f_hf_red_pct"):
                r[f], r[f + "_sd"], _ = mean_sd([e[f] for e in effs])
                r[f + "_per_seed"] = " / ".join(f"{e[f]:.1f}" for e in effs)
            lpf_rows.append(r)
    seed_rows = []
    for (code, base), by_seed in sorted(per_seed.items()):
        for s, st in sorted(by_seed.items()):
            for key in ("base", "nolpf", "lpf"):
                if key in st:
                    seed_rows.append({"code": code, "seed": s,
                                      "arm": ARM_NAMES[(base, {"base": None, "nolpf": 1.0,
                                                               "lpf": 0.2}[key])],
                                      **st[key]})
    C.write_csv(out / "summary_seed_mean.csv", rows)
    C.write_csv(out / "summary_per_seed.csv", seed_rows)
    C.write_csv(out / "lpf_effect.csv", lpf_rows)

    # ---------------------------------------------------------- figures
    figs = []
    for base in ("pd", "pid"):
        if not any((code, base, "nolpf") in pooled for code in ("original", "fixed")):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for ax, code in zip(axes, ("original", "fixed")):
            for key, col, lab in (("base", "0.4", base.upper()),
                                  ("nolpf", "#c0392b", "+RRL (no LPF)"),
                                  ("lpf", "#2471a3", "+RRL+LPF")):
                vals = pooled.get((code, base, key), {}).get("motor_hf")
                if not vals:
                    continue
                v = np.sort(np.concatenate(vals))
                v = v[np.isfinite(v)]
                if v.size:
                    ax.plot(v, np.linspace(0, 1, v.size), color=col, label=lab)
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(f"{code} code")
            ax.set_xlabel("per-episode motor command HF RMS > 10 Hz [N]")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("fraction of episodes")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"{base.upper()}: chatter distribution, all seeds pooled")
        fig.tight_layout()
        f = out / f"fig_chatter_cdf_{base}.png"
        fig.savefig(f, dpi=140)
        plt.close(fig)
        figs.append(f.name)
    for base, (ep, ratio, s) in worst.items():
        traces = {}
        traces["base"] = C.fly_traces(C.paper_env(base_controller=base, **fixed),
                                      episode_ids=[ep], **kw_tr(kw))[ep]
        for key in ("nolpf", "lpf"):
            run = finished_run(a.results, f"A_{base}_{key}_s{s}")
            if run is not None:
                traces[key] = C.fly_traces(C.env_from_run(run), run_dir=run,
                                           episode_ids=[ep], **kw_tr(kw))[ep]
        fig, axes = plt.subplots(len(traces), 1, figsize=(10, 2.2 * len(traces)),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        t = np.arange(traces["base"]["motor"].shape[0]) * 0.005
        for ax, (key, tr) in zip(axes, traces.items()):
            for j in range(4):
                ax.plot(t, tr["motor"][:, j], lw=0.6)
            ax.set_ylabel("rotor cmd [N]")
            ax.set_title({"base": base.upper(), "nolpf": f"{base.upper()}+RRL (no LPF)",
                          "lpf": f"{base.upper()}+RRL+LPF"}[key], fontsize=9)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(f"fixed code, seed {s}: episode {ep}, the worst unfiltered chatter "
                     f"({ratio:.1f}× the baseline's)", fontsize=10)
        fig.tight_layout()
        f = out / f"fig_worst_chatter_{base}_ep{ep}.png"
        fig.savefig(f, dpi=140)
        plt.close(fig)
        figs.append(f.name)

    # ---------------------------------------------------------- markdown
    L = ["# 02 — Retrained with every fix (Q1, Q2)", "",
         "Original = the paper's checkpoints on the original code. Fixed = policies "
         "retrained with every fix (same protocol, paper reward and authority). "
         "Each number is the mean ± SD over seeds of the seed-level statistic; "
         "`cond` statistics use the episodes that the baseline, RRL and RRL+LPF of "
         "that seed all survive.", ""]
    show = []
    for r in rows:
        show.append({"code": r["code"], "arm": r["arm"], "seeds": r["seeds"],
                     "term": pm(r["term_pct"], r["term_pct_sd"], r["seeds"], "{:.2f}"),
                     "median": pm(r["cond_median"], r["cond_median_sd"], r["seeds"]),
                     "p90": pm(r["cond_p90"], r["cond_p90_sd"], r["seeds"]),
                     "sat": pm(r["sat_pct"], r["sat_pct_sd"], r["seeds"], "{:.3f}"),
                     "rough": pm(r["roughness"], r["roughness_sd"], r["seeds"]),
                     "hf": pm(r["motor_hf"], r["motor_hf_sd"], r["seeds"]),
                     "gt3": pm(r["n_gt3x"], r["n_gt3x_sd"], r["seeds"], "{:.1f}"),
                     "own": pm(r["own_fail"], r["own_fail_sd"], r["seeds"], "{:.1f}")})
    L += ["## Tracking and command smoothness", "",
          C.md_table(show, [("code", "code"), ("arm", "arm"), ("term %", "term"),
                            ("cond median [m]", "median"), ("cond P90 [m]", "p90"),
                            ("sat %", "sat"), ("roughness", "rough"),
                            ("motor HF [N]", "hf"), ("eps >3× base HF", "gt3"),
                            ("own failures", "own")]), ""]
    show = [{"code": r["code"], "base": r["base"], "seeds": r["seeds"],
             "rough": f"{r['roughness_red_pct']:.1f} ± {r['roughness_red_pct_sd']:.1f}  "
                      f"[{r['roughness_red_pct_per_seed']}]",
             "sat": f"{r['sat_red_pct']:.1f} ± {r['sat_red_pct_sd']:.1f}",
             "med": f"{r['median_red_pct']:.1f} ± {r['median_red_pct_sd']:.1f}  "
                    f"[{r['median_red_pct_per_seed']}]",
             "hf": f"{r['motor_hf_red_pct']:.1f} ± {r['motor_hf_red_pct_sd']:.1f}"}
            for r in lpf_rows]
    L += ["## LPF effect (RRL+LPF vs RRL, same seed; positive = LPF better)", "",
          C.md_table(show, [("code", "code"), ("base", "base"), ("seeds", "seeds"),
                            ("roughness red. % [per seed]", "rough"),
                            ("saturation red. %", "sat"),
                            ("median RMSE red. % [per seed]", "med"),
                            ("motor HF red. %", "hf")]),
          "", "Paper (original code, best seed only, survivor bank): roughness 53.4 / 49.0 %, "
          "saturation 35.7 / 16.4 %, median 17.8 / 3.2 % (PD / PID).", ""]
    L += ["## Figures", ""] + [f"![]({f})" for f in figs] + [""]
    if not per_seed:
        L += ["No finished runs yet.", ""]
    (out / "RETRAIN.md").write_text("\n".join(L) + "\n")
    return rows, lpf_rows


def kw_tr(kw):
    return {"eval_seed": kw["eval_seed"], "chunk": kw["chunk"],
            "n_episodes": kw["n_episodes"]}


# ====================================================================== Q3
def sweep_section(a, kw, out):
    seeds = [int(s) for s in a.seeds.split(",")]
    base_m = C.cached_fly(a.results, "PID | all_fixes",
                          C.paper_env(base_controller="pid", **C.all_fixes()), **kw)
    cells = defaultdict(list)            # (tau, auth, lpf) -> per-seed stats
    for s in seeds:
        for j in job_list([s]):
            if j["base"] != "pid" or not (j["group"] in ("B", "C") or
                                          j["name"] == f"A_pid_lpf_s{s}"):
                continue
            run = finished_run(a.results, j["name"])
            if run is None:
                continue
            m = C.cached_fly(a.results, j["name"], C.env_from_run(run), run_dir=run, **kw)
            C.check_same_draws({"base": base_m, "arm": m})
            pair = C.joint_survivors({"b": base_m, "a": m})
            st = C.summarize(m, pair)
            sb = C.summarize(base_m, pair)
            st.update(C.chatter_tail(m, base_m, pair))
            st["own_fail"] = C.own_failures(m, base_m)
            st["median_vs_base"] = st["cond_median"] / sb["cond_median"]
            st["seed"] = s
            cells[(j["tau_pos"], j["auth"], j["beta"] < 1)].append(st)
    rows = []
    for (tau, auth, lpf), sts in sorted(cells.items()):
        r = {"tau_pos": tau, "L_cm": 25.0 * tau, "authority": auth,
             "lpf": "yes" if lpf else "no", "seeds": len(sts)}
        for f in ("charged_median", "charged_mean", "cond_median", "cond_p90",
                  "median_vs_base", "term_pct", "own_fail", "sat_pct", "sat_gt5_pct",
                  "roughness", "motor_hf", "motor_hf_max", "res_f_slew", "res_f_hf",
                  "n_gt3x", "hf_ratio_max"):
            r[f], r[f + "_sd"], _ = mean_sd([x[f] for x in sts])
        rows.append(r)
    C.write_csv(out / "sweep_seed_mean.csv", rows)
    C.write_csv(out / "sweep_per_seed.csv",
                [{"tau_pos": k[0], "authority": k[1], "lpf": k[2], **x}
                 for k, v in sorted(cells.items()) for x in v])
    sb = C.summarize(base_m)

    figs = []
    metrics = [("charged_median", "median RMSE, all episodes [m]", "{:.4f}"),
               ("cond_p90", "P90 RMSE (pair survivors) [m]", "{:.4f}"),
               ("term_pct", "terminated [%]", "{:.2f}"),
               ("own_fail", "own failures vs PID", "{:.1f}"),
               ("sat_pct", "saturated steps [%]", "{:.2f}"),
               ("roughness", "input roughness", "{:.3f}"),
               ("motor_hf", "motor HF > 10 Hz [N]", "{:.3f}"),
               ("res_f_slew", "thrust residual slew [N/s]", "{:.1f}")]
    lpf_rows = [r for r in rows if r["lpf"] == "yes"]
    if lpf_rows:
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
        for ax, (f, title, fmt) in zip(axes.ravel(), metrics):
            G = np.full((len(TAUS), len(AUTHS)), np.nan)
            for r in lpf_rows:
                if r["tau_pos"] in TAUS and r["authority"] in AUTHS:
                    G[TAUS.index(r["tau_pos"]), AUTHS.index(r["authority"])] = r[f]
            im = ax.imshow(G, cmap="viridis_r" if f != "dummy" else "viridis", aspect="auto")
            for i in range(len(TAUS)):
                for k in range(len(AUTHS)):
                    if np.isfinite(G[i, k]):
                        ax.text(k, i, fmt.format(G[i, k]), ha="center", va="center",
                                fontsize=8, color="w")
            ax.set_xticks(range(len(AUTHS)), [f"{x:g}" for x in AUTHS])
            ax.set_yticks(range(len(TAUS)), [f"{t:g}\n({25 * t:g} cm)" for t in TAUS])
            ax.set_xlabel("residual authority")
            ax.set_ylabel("tau_pos (length scale)")
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"PID + RRL + LPF, every fix on, seed mean. "
                     f"PID baseline: median {sb['charged_median']:.4f} m, "
                     f"term {sb['term_pct']:.2f} %, sat {sb['sat_pct']:.2f} %", fontsize=10)
        fig.tight_layout()
        f = out / "fig_sweep_heatmaps.png"
        fig.savefig(f, dpi=140)
        plt.close(fig)
        figs.append(f.name)

    L = ["# 03 — Reward × authority sweep (Q3)", "",
         "PID base, every fix on. `tau_pos` is the width of the position reward on "
         "the normalised error; its physical length scale is 0.25·tau_pos m "
         "(0.5 → 12.5 cm, the paper; 0.05 → 1.25 cm). Seed mean ± SD; "
         "`median, all` charges failures at 2 m and is comparable across cells; "
         "`own fails` = episodes the arm lost that the PID baseline flew.", "",
         f"PID baseline (fixed code): median {sb['charged_median']:.4f} m, "
         f"P90 {sb['cond_p90']:.4f} m, term {sb['term_pct']:.2f} %, "
         f"sat {sb['sat_pct']:.3f} %, motor HF {sb['motor_hf']:.4f} N.", ""]
    show = [{"tau": f"{r['tau_pos']:g} ({r['L_cm']:g} cm)", "auth": f"{r['authority']:g}",
             "lpf": r["lpf"], "seeds": r["seeds"],
             "med": pm(r["charged_median"], r["charged_median_sd"], r["seeds"]),
             "ratio": f"{r['median_vs_base']:.3f}",
             "p90": pm(r["cond_p90"], r["cond_p90_sd"], r["seeds"]),
             "term": f"{r['term_pct']:.2f}", "own": f"{r['own_fail']:.1f}",
             "sat": f"{r['sat_pct']:.3f}", "sat5": f"{r['sat_gt5_pct']:.1f}",
             "rough": f"{r['roughness']:.3f}", "hf": f"{r['motor_hf']:.3f}",
             "slew": f"{r['res_f_slew']:.1f}", "gt3": f"{r['n_gt3x']:.1f}"}
            for r in rows]
    L += [C.md_table(show, [("tau_pos", "tau"), ("authority", "auth"), ("LPF", "lpf"),
                            ("seeds", "seeds"), ("median, all [m]", "med"),
                            ("median / PID", "ratio"), ("P90 [m]", "p90"),
                            ("term %", "term"), ("own fails", "own"), ("sat %", "sat"),
                            ("eps sat>5 %", "sat5"), ("roughness", "rough"),
                            ("motor HF", "hf"), ("slew N/s", "slew"),
                            ("eps >3× PID HF", "gt3")]), ""]
    L += ["## Figures", ""] + [f"![]({f})" for f in figs] + [""]
    if not rows:
        L += ["No finished sweep runs yet.", ""]
    (out / "SWEEP.md").write_text("\n".join(L) + "\n")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="study_results")
    p.add_argument("--paper_dir", default="auto")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--sections", default="retrain,sweep")
    a = p.parse_args()
    kw = dict(eval_seed=a.eval_seed, n_episodes=a.episodes, chunk=a.chunk)
    secs = set(a.sections.split(","))
    if "retrain" in secs:
        out = C.results_dir(Path(a.results) / "02_retrain_fixed")
        retrain_section(a, kw, out)
        print((out / "RETRAIN.md").read_text())
    if "sweep" in secs:
        out = C.results_dir(Path(a.results) / "03_reward_sweep")
        sweep_section(a, kw, out)
        print((out / "SWEEP.md").read_text())


if __name__ == "__main__":
    main()
