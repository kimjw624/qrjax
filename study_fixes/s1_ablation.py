"""Stage 1 -- did the bugs matter? No training needed.

A. Baselines (PD, PID) on one fixed bank under: the original code, each fix
   alone, all fixes, and all fixes minus one.
B. The paper's trained policies (PD/PID + RRL, with and without the thrust
   LPF) flown under the same variants. They were trained on the original
   code, so this asks: does the result -- including the LPF's effect --
   survive when the physics bugs are fixed, before anything is retrained?

Writes <results>/01_fix_ablation/.
"""

import argparse
from pathlib import Path

import numpy as np

from . import common as C

VARIANTS = (["original"] + list(C.SINGLE_FIXES) + ["all_fixes"]
            + [f"loo_{k}" for k in C.SINGLE_FIXES if k != "fd_counter"])


def variant_overrides(name):
    if name == "original":
        return {}
    if name == "all_fixes":
        return C.all_fixes()
    if name.startswith("loo_"):
        return C.leave_one_out(name[4:])
    return dict(C.SINGLE_FIXES[name])


def variant_label(name):
    if name.startswith("loo_"):
        return "all fixes except " + C.FIX_LABELS[name[4:]]
    return C.FIX_LABELS[name]


def paper_dir_from(results, arg):
    if arg and arg != "auto":
        return Path(arg)
    f = Path(results) / "00_provenance" / "paper_dir.txt"
    if f.is_file() and f.read_text().strip():
        return Path(f.read_text().strip())
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="study_results")
    p.add_argument("--paper_dir", default="auto")
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--chunk", type=int, default=256)
    a = p.parse_args()
    out = C.results_dir(Path(a.results) / "01_fix_ablation")
    kw = dict(eval_seed=a.eval_seed, n_episodes=a.episodes, chunk=a.chunk)
    L = ["# 01 — Fix ablation (no retraining)", "",
         f"Bank: {a.episodes} episodes, eval seed {a.eval_seed}, 200 Hz, 10 s, the "
         "paper's full disturbance box. Every arm flies the same episodes. "
         "`charged` = all episodes, failures charged at 2 m; `cond` = episodes that "
         "every arm in the table survives (the paper's convention).", ""]

    # ================================================================ A
    base_arms = {}
    for base in ("pd", "pid"):
        arms = {}
        for v in VARIANTS:
            env = C.paper_env(base_controller=base, **variant_overrides(v))
            arms[v] = C.cached_fly(a.results, f"{base.upper()} | {v}", env, **kw)
        C.check_same_draws(arms)
        base_arms[base] = arms
        C.write_per_episode(out / f"per_episode_baseline_{base}.csv", arms)
        joint = C.joint_survivors(arms)
        ref = arms["original"]
        s_ref = C.summarize(ref, joint)
        rows = []
        for v, m in arms.items():
            s = C.summarize(m, joint)
            rows.append({"variant": variant_label(v), "key": v, **s,
                         "own_fail_vs_orig": C.own_failures(m, ref),
                         "d_cond_median_pct": 100 * (s["cond_median"] / s_ref["cond_median"] - 1),
                         "d_term_pp": s["term_pct"] - s_ref["term_pct"],
                         "wdd_max_mean": float(np.nanmean(m["wdd_max"]))})
        C.write_csv(out / f"summary_baseline_{base}.csv", rows)
        L += [f"## A. {base.upper()} baseline under each fix", "",
              C.md_table(rows, [("variant", "variant"), ("term %", "term_pct"),
                                ("Δterm pp", "d_term_pp"),
                                ("charged mean [m]", "charged_mean"),
                                ("cond median [m]", "cond_median"),
                                ("Δ cond median", "d_cond_median_pct"),
                                ("cond P90 [m]", "cond_p90"), ("sat %", "sat_pct"),
                                ("roughness", "roughness"), ("motor HF [N]", "motor_hf"),
                                ("max |ω̇_d| mean", "wdd_max_mean")],
                         {"term_pct": "{:.2f}", "d_term_pp": "{:+.2f}",
                          "charged_mean": "{:.4f}", "cond_median": "{:.4f}",
                          "d_cond_median_pct": "{:+.1f}%", "cond_p90": "{:.4f}",
                          "sat_pct": "{:.3f}", "roughness": "{:.4f}",
                          "motor_hf": "{:.4f}", "wdd_max_mean": "{:.1f}"}),
              "", f"cond = {int(joint.sum())} episodes survived by every variant.", ""]

    # ================================================================ B
    paper_dir = paper_dir_from(a.results, a.paper_dir)
    pol = C.paper_policies(paper_dir) if paper_dir else {}
    if not pol:
        L += ["## B. Paper policies", "", "Skipped: no paper experiment found "
              "(see 00_provenance).", ""]
        (out / "ABLATION.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    best = C.paper_best_seed(paper_dir)
    seeds = sorted({s for s, _, _ in pol})
    best = best if best in seeds else seeds[0]
    pol_variants = ["original"] + list(C.SINGLE_FIXES) + ["all_fixes"]
    L += ["## B. The paper's trained policies under each fix", "",
          f"Policies from `{paper_dir}`; every variant for the paper's selected seed "
          f"{best}, original and all fixes for every seed {seeds}. Paired against the "
          "baseline flown under the same variant.", ""]
    rows, lpf_rows, per_ep = [], [], {}
    for v in pol_variants:
        for seed in seeds:
            if seed != best and v not in ("original", "all_fixes"):
                continue
            for base in ("pd", "pid"):
                trio = {"base": base_arms[base][v]}
                for tag in ("none", "thrust"):
                    run = pol.get((seed, tag, base))
                    if run is None:
                        continue
                    env = C.env_from_run(run, **variant_overrides(v))
                    name = f"{base.upper()}+RRL{'+LPF' if tag == 'thrust' else ''}"
                    m = C.cached_fly(a.results, f"paper s{seed} {name} | {v}", env,
                                     run_dir=run, **kw)
                    trio[tag] = m
                    per_ep[f"{v}|s{seed}|{name}"] = m
                C.check_same_draws(trio)
                joint = C.joint_survivors(trio)
                sb = C.summarize(trio["base"], joint)
                for tag in ("none", "thrust"):
                    if tag not in trio:
                        continue
                    s = C.summarize(trio[tag], joint)
                    rows.append({"variant": variant_label(v), "key": v, "seed": seed,
                                 "arm": f"{base.upper()}+RRL{'+LPF' if tag == 'thrust' else ''}",
                                 **s, "own_fail": C.own_failures(trio[tag], trio["base"]),
                                 "rrl_median_red_pct": 100 * (1 - s["cond_median"] / sb["cond_median"]),
                                 **C.chatter_tail(trio[tag], trio["base"], joint)})
                if "none" in trio and "thrust" in trio:
                    s0, s1 = C.summarize(trio["none"], joint), C.summarize(trio["thrust"], joint)
                    lpf_rows.append({
                        "variant": variant_label(v), "key": v, "seed": seed,
                        "base": base.upper(), "n_cond": s0["n_cond"],
                        "lpf_roughness_pct": 100 * (1 - s1["roughness"] / s0["roughness"]),
                        "lpf_sat_pct": 100 * (1 - s1["sat_pct"] / s0["sat_pct"])
                        if s0["sat_pct"] > 0 else np.nan,
                        "lpf_median_pct": 100 * (1 - s1["cond_median"] / s0["cond_median"]),
                        "lpf_motor_hf_pct": 100 * (1 - s1["motor_hf"] / s0["motor_hf"]),
                        "nolpf_gt3x": C.chatter_tail(trio["none"], trio["base"], joint)["n_gt3x"],
                        "lpf_gt3x": C.chatter_tail(trio["thrust"], trio["base"], joint)["n_gt3x"]})
    C.write_csv(out / "summary_paper_policies.csv", rows)
    C.write_csv(out / "lpf_effect_paper_policies.csv", lpf_rows)
    C.write_per_episode(out / "per_episode_paper_policies.csv", per_ep)
    best_rows = [r for r in rows if r["seed"] == best]
    L += [C.md_table(best_rows, [("variant", "variant"), ("arm", "arm"),
                                 ("term %", "term_pct"), ("own fails", "own_fail"),
                                 ("cond median", "cond_median"), ("cond P90", "cond_p90"),
                                 ("RRL median red.", "rrl_median_red_pct"),
                                 ("sat %", "sat_pct"), ("roughness", "roughness"),
                                 ("motor HF", "motor_hf"), ("HF/base max", "hf_ratio_max"),
                                 (">3× base HF", "n_gt3x")],
                     {"term_pct": "{:.2f}", "cond_median": "{:.4f}", "cond_p90": "{:.4f}",
                      "rrl_median_red_pct": "{:.1f}%", "sat_pct": "{:.3f}",
                      "roughness": "{:.4f}", "motor_hf": "{:.4f}", "hf_ratio_max": "{:.1f}"}),
          "", "### LPF effect (paper policies; thrust-LPF arm vs unfiltered arm, same seed)", "",
          C.md_table(lpf_rows, [("variant", "variant"), ("seed", "seed"), ("base", "base"),
                                ("roughness red.", "lpf_roughness_pct"),
                                ("sat red.", "lpf_sat_pct"), ("median red.", "lpf_median_pct"),
                                ("motor HF red.", "lpf_motor_hf_pct"),
                                ("no-LPF eps >3× base HF", "nolpf_gt3x"),
                                ("LPF eps >3× base HF", "lpf_gt3x")],
                     {"lpf_roughness_pct": "{:.1f}%", "lpf_sat_pct": "{:.1f}%",
                      "lpf_median_pct": "{:.1f}%", "lpf_motor_hf_pct": "{:.1f}%"}),
          "", "Paper (original code, survivor bank): roughness −53.4 % / −49.0 %, "
          "saturation −35.7 % / −16.4 %, median −17.8 % / −3.2 % (PD / PID).", ""]
    (out / "ABLATION.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
