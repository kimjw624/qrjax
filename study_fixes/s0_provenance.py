"""Stage 0 -- provenance: which code and which experiment produced the paper?

1. Records the git commit, the uncommitted changes (this overlay) and the JAX
   device.
2. Finds every run_final_study output under experiments/ and scores it
   against the paper's Table 1 and effect sizes; the best match is the
   paper experiment.
3. Re-evaluates that experiment's checkpoints with the CURRENT code (defaults
   = original behaviour) through run_final_study's own survivor-bank code and
   compares the regenerated Table 1 with the stored table1.csv. An exact
   match shows both that the checkpoints on disk are the paper's and that the
   patched code with default settings is the code that produced them.

Writes <results>/00_provenance/.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from . import common as C

import numpy as np


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=C.REPO, stderr=subprocess.STDOUT,
                                       text=True).strip()
    except Exception as e:                                   # noqa: BLE001
        return f"(failed: {e})"


def read_table1(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["arm"]] = {k: float(v) for k, v in r.items() if k != "arm"}
    return rows


def read_effects(path):
    out = {}
    if not Path(path).is_file():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["base"]] = {k: float(v) for k, v in r.items() if k != "base"}
    return out


def score_against_paper(t1, eff):
    """Max abs deviation from the paper's printed numbers (4-decimal table)."""
    devs = []
    for arm, (mean, sd, med, p90, sat) in C.PAPER_TABLE1.items():
        if arm not in t1:
            return np.inf, ["missing arm " + arm]
        r = t1[arm]
        for name, paper, ours in (("mean", mean, r["rmse_mean"]), ("sd", sd, r["rmse_sd"]),
                                  ("median", med, r["rmse_median"]), ("p90", p90, r["rmse_p90"])):
            devs.append((abs(paper - ours), f"{arm} {name}: paper {paper} vs {ours:.4f}"))
        devs.append((abs(sat - r["sat_pct"]) / 10.0, f"{arm} sat: paper {sat} vs {r['sat_pct']:.3f}"))
    for base, d in C.PAPER_EFFECTS.items():
        for k, paper in d.items():
            if base in eff and k in eff[base]:
                devs.append((abs(paper - eff[base][k]) / 1000.0,
                             f"{base} {k}: paper {paper} vs {eff[base][k]:.1f}"))
    worst = max(devs, key=lambda x: x[0])
    return worst[0], [d for _, d in sorted(devs, reverse=True)[:6]]


def reproduce(paper_dir, out_dir, episodes, chunk):
    """Re-run run_final_study's evaluation for the selected seed, no training."""
    sys.path.insert(0, str(C.REPO))
    import scripts.run_final_study as F                       # noqa: E402

    manifest = json.loads((paper_dir / "manifest.json").read_text())
    best = int(manifest.get("best_seed", 0))
    eval_seed = int(manifest.get("eval_seed", 20260829))
    hz = float(manifest.get("control_hz", 200.0))
    ep = int(manifest.get("episodes", episodes))
    cfg_env = C.EnvConfig()
    cfg_env.dt = 1.0 / hz
    cfg_env.episode_steps = int(round(hz * 10.0))
    cfg_env.history = 10
    args = SimpleNamespace(episodes=ep, chunk=chunk, eval_seed=eval_seed,
                           checkpoint="best")
    keep, draw, stats, M = F.evaluate_seed(paper_dir / f"seed_{best:02d}", cfg_env,
                                           args, f"seed_{best:02d}")
    rows = {}
    for nm, *_ in F.ARM_SPEC:
        m = M[nm]
        rows[nm] = {"rmse_mean": float(m["rmse"].mean()),
                    "rmse_sd": float(m["rmse"].std(ddof=1)),
                    "rmse_median": float(np.median(m["rmse"])),
                    "rmse_p90": float(np.percentile(m["rmse"], 90)),
                    "sat_pct": float(m["sat_pct"].mean()),
                    "roughness": float(m["roughness"].mean()),
                    "motor_hf_N": float(m["motor_hf"].mean())}
    C.write_csv(out_dir / "table1_regenerated.csv",
                [{"arm": k, **v} for k, v in rows.items()])
    C.write_csv(out_dir / "bank_stats.csv", [{
        "candidates_drawn": stats["candidates_drawn"],
        "candidates_survived": stats["candidates_survived"],
        "rejection_rate_pct": stats["rejection_rate_pct"],
        **{f"term_{k}": v for k, v in stats["termination_rate_by_arm_pct"].items()}}])
    return rows, best, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="study_results")
    p.add_argument("--paper_dir", default="auto",
                   help="run_final_study output used for the paper; 'auto' searches experiments/")
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--skip_reproduce", action="store_true")
    a = p.parse_args()

    out = C.results_dir(Path(a.results) / "00_provenance")
    L = ["# 00 — Provenance", ""]

    # ---------------------------------------------------------------- code
    info = {"commit": sh(["git", "rev-parse", "HEAD"]),
            "commit_date": sh(["git", "log", "-1", "--format=%ci %s"]),
            "branch": sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "status": sh(["git", "status", "--porcelain"]),
            "remote": sh(["git", "remote", "-v"]),
            **C.device_info()}
    (out / "git_diff.patch").write_text(sh(["git", "diff"]) + "\n")
    (out / "code_info.json").write_text(json.dumps(info, indent=2))
    L += ["## Code", "",
          f"- commit `{info['commit'][:10]}` — {info['commit_date']}",
          f"- branch `{info['branch']}`",
          f"- JAX {info['jax']} on {', '.join(info['devices'])}",
          "- uncommitted changes (this study's overlay), see `git_diff.patch`:", "",
          "```", info["status"] or "(none)", "```", ""]

    # ---------------------------------------------------------------- experiments
    if a.paper_dir == "auto":
        cands = sorted({p.parent for p in (C.REPO / "experiments").glob("**/table1.csv")})
    else:
        cands = [Path(a.paper_dir)]
    rows = []
    for d in cands:
        t1 = d / "table1.csv"
        if not t1.is_file():
            continue
        try:
            dev, worst = score_against_paper(read_table1(t1), read_effects(d / "effects.csv"))
        except Exception as e:                               # noqa: BLE001
            dev, worst = np.inf, [f"unreadable: {e}"]
        rows.append({"dir": str(d.relative_to(C.REPO) if d.is_relative_to(C.REPO) else d),
                     "max_dev": dev, "worst": "; ".join(worst[:2]),
                     "modified": d.stat().st_mtime, "path": d})
    rows.sort(key=lambda r: r["max_dev"])
    C.write_csv(out / "experiment_candidates.csv",
                [{k: v for k, v in r.items() if k != "path"} for r in rows])
    L += ["## Which experiment produced the paper", ""]
    if not rows:
        L += ["No `table1.csv` found under `experiments/`. Set `PAPER_DIR=...` to the "
              "run_final_study output used for the paper. Stages that need the paper's "
              "checkpoints will skip those arms.", ""]
        (out / "paper_dir.txt").write_text("")
        (out / "PROVENANCE.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    L += [C.md_table(rows[:8], [("experiment", "dir"), ("max deviation from paper", "max_dev"),
                                ("largest deviations", "worst")], {"max_dev": "{:.4f}"}), ""]
    best = rows[0]
    paper_dir = best["path"]
    if best["max_dev"] <= 6e-5:
        verdict = ("matches every number of Table 1 and the effect sizes to the paper's "
                   "printed precision.")
    elif best["max_dev"] <= 5e-4:
        verdict = (f"matches to within {best['max_dev']:.4f} (last printed digit); "
                   "almost certainly the paper run.")
    else:
        verdict = (f"closest candidate, but deviates by {best['max_dev']:.4f}; the paper's "
                   "numbers may come from a run that is no longer on disk.")
    L += [f"**Selected:** `{best['dir']}` — {verdict}", ""]
    (out / "paper_dir.txt").write_text(str(paper_dir) + "\n")
    dst = out / "paper_experiment"
    dst.mkdir(exist_ok=True)
    for f in ("REPORT.md", "table1.csv", "effects.csv", "paired_tests.csv",
              "manifest.json", "all_seeds.json"):
        if (paper_dir / f).is_file():
            shutil.copy2(paper_dir / f, dst / f)
    pol = C.paper_policies(paper_dir)
    seeds = sorted({s for s, _, _ in pol})
    L += [f"- trained policies found: {len(pol)} (seeds {seeds}); "
          f"selected seed in the paper: {C.paper_best_seed(paper_dir)}", ""]

    # ---------------------------------------------------------------- reproduce
    if not a.skip_reproduce and (paper_dir / "manifest.json").is_file() and pol:
        print("re-evaluating the paper's checkpoints with the current code ...", flush=True)
        try:
            # The survivor bank's candidate episodes depend on the chunk size it
            # was drawn with, which the manifest does not record. Its
            # candidates_drawn count must be a multiple of it: try the default
            # (128) first, then the other sizes consistent with that count.
            man = json.loads((paper_dir / "manifest.json").read_text())
            drawn = int(man.get("bank_stats", {}).get("candidates_drawn", 0))
            chunks = [a.chunk] + [c for c in (128, 256, 64, 32, 512, 16)
                                  if c != a.chunk and drawn and drawn % c == 0]
            stored = read_table1(paper_dir / "table1.csv")
            best_try = None
            for chunk in chunks[:5]:
                regen, seed, stats = reproduce(paper_dir, out, None, chunk)
                cmp_rows, worst = [], 0.0
                for arm, r in regen.items():
                    s = stored.get(arm, {})
                    d = max(abs(r[k] - s[k]) for k in ("rmse_mean", "rmse_sd", "rmse_median",
                                                       "rmse_p90", "sat_pct") if k in s) \
                        if s else np.inf
                    worst = max(worst, d)
                    cmp_rows.append({"arm": arm, "stored_median": s.get("rmse_median", np.nan),
                                     "regenerated_median": r["rmse_median"],
                                     "stored_mean": s.get("rmse_mean", np.nan),
                                     "regenerated_mean": r["rmse_mean"], "max_abs_diff": d})
                print(f"  bank chunk {chunk}: drew {stats['candidates_drawn']} "
                      f"(stored {drawn}), max |diff| {worst:.2e}", flush=True)
                if best_try is None or worst < best_try[0]:
                    best_try = (worst, chunk, cmp_rows, seed, stats)
                if worst <= 5e-5:
                    break
            worst, chunk, cmp_rows, seed, stats = best_try
            C.write_csv(out / "reproduction.csv", cmp_rows)
            verdict = ("**exact** (≤1e-6): the stored checkpoints and the current code with "
                       "default settings reproduce the stored Table 1"
                       if worst <= 1e-6 else
                       ("**agrees to 4 decimals** (GPU kernels can differ in the last digits)"
                        if worst <= 5e-5 else
                        f"**differs** (max {worst:.2e}): the code or checkpoints changed after "
                        "the paper run"))
            L += ["## Re-evaluation with the current code", "",
                  f"Seed {seed}, survivor bank exactly as `run_final_study` builds it "
                  f"(chunk {chunk}).", "",
                  C.md_table(cmp_rows, [("arm", "arm"), ("stored median", "stored_median"),
                                        ("regenerated median", "regenerated_median"),
                                        ("stored mean", "stored_mean"),
                                        ("regenerated mean", "regenerated_mean"),
                                        ("max |diff|", "max_abs_diff")],
                             {"max_abs_diff": "{:.1e}"}), "",
                  f"Verdict: {verdict}.", "",
                  f"Survivor bank: {stats['candidates_drawn']} candidates drawn, "
                  f"{stats['rejection_rate_pct']:.1f}% rejected.", ""]
        except SystemExit as e:
            L += [f"Re-evaluation skipped: {e}", ""]
        except Exception as e:                               # noqa: BLE001
            L += [f"Re-evaluation FAILED: {type(e).__name__}: {e}", ""]
    (out / "PROVENANCE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
