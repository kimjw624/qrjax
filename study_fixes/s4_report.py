"""Stage 4 -- SUMMARY.md over everything that finished, plus the upload zip.

The zip holds every result file but no checkpoints, caches or raw arrays, so
it stays small enough to send back for analysis.
"""

import argparse
import csv
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from . import common as C

NEGLIGIBLE_MEDIAN_PCT = 2.0     # |Δ median| below this ...
NEGLIGIBLE_TERM_PP = 0.5        # ... and |Δ termination| below this = tracking unaffected
NEGLIGIBLE_SMOOTH_PCT = 5.0     # |Δ roughness|, |Δ motor HF| below this = smoothness unaffected


def read_csv(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def section(path, title):
    path = Path(path)
    if not path.is_file():
        return [f"## {title}", "", f"_not produced (`{path.name}` missing)_", ""]
    txt = path.read_text().splitlines()
    rel = path.parent.name
    out = []
    for line in txt:
        if line.startswith("# "):
            out.append("## " + line[2:])
        elif line.startswith("## "):
            out.append("### " + line[3:])
        elif line.startswith("![]("):
            out.append(line.replace("![](", f"![]({rel}/"))
        else:
            out.append(line)
    return out + [""]


def verdict(d_med, d_term, d_smooth):
    if not (np.isfinite(d_med) and np.isfinite(d_term)):
        return "–"
    tracking = abs(d_med) >= NEGLIGIBLE_MEDIAN_PCT or abs(d_term) >= NEGLIGIBLE_TERM_PP
    smooth = np.isfinite(d_smooth) and abs(d_smooth) >= NEGLIGIBLE_SMOOTH_PCT
    if tracking:
        return "**changes tracking**"
    if smooth:
        return "changes smoothness metrics only"
    return "negligible"


def bug_table(results):
    a = Path(results) / "01_fix_ablation"
    base = {b: {r["key"]: r for r in read_csv(a / f"summary_baseline_{b}.csv")}
            for b in ("pd", "pid")}
    pol = read_csv(a / "summary_paper_policies.csv")
    if not base["pid"]:
        return ["_01_fix_ablation not produced_", ""]
    best_seed = None
    if pol:
        seeds = [r["seed"] for r in pol if r["key"] != "original"]
        best_seed = seeds[0] if seeds else pol[0]["seed"]
    rows = []
    for key in list(C.SINGLE_FIXES) + ["all_fixes"]:
        r = {"fix": C.FIX_LABELS[key]}
        worst_med, worst_term, worst_smooth = 0.0, 0.0, 0.0
        for b in ("pid", "pd"):
            x = base[b].get(key)
            o = base[b].get("original")
            if x:
                r[f"{b}_dmed"] = x["d_cond_median_pct"]
                r[f"{b}_dterm"] = x["d_term_pp"]
                worst_med = max(worst_med, abs(x["d_cond_median_pct"]))
                worst_term = max(worst_term, abs(x["d_term_pp"]))
            if x and o:
                dr = 100 * (x["roughness"] / o["roughness"] - 1)
                dh = 100 * (x["motor_hf"] / o["motor_hf"] - 1)
                if b == "pid":
                    r["pid_drough"], r["pid_dhf"] = dr, dh
                worst_smooth = max(worst_smooth, abs(dr), abs(dh))
        for arm in ("PID+RRL+LPF", "PID+RRL"):
            o = [p for p in pol if p["key"] == "original" and p["arm"] == arm
                 and p["seed"] == best_seed]
            v = [p for p in pol if p["key"] == key and p["arm"] == arm
                 and p["seed"] == best_seed]
            if o and v:
                dm = 100 * (v[0]["cond_median"] / o[0]["cond_median"] - 1)
                dt = v[0]["term_pct"] - o[0]["term_pct"]
                r[f"{arm}_dmed"] = dm
                r[f"{arm}_dterm"] = dt
                worst_med = max(worst_med, abs(dm))
                worst_term = max(worst_term, abs(dt))
        r["verdict"] = verdict(worst_med, worst_term, worst_smooth)
        rows.append(r)
    f = {k: "{:+.1f}%" for k in ("pid_dmed", "pd_dmed", "PID+RRL+LPF_dmed", "PID+RRL_dmed",
                                 "pid_drough", "pid_dhf")}
    f.update({k: "{:+.2f}" for k in ("pid_dterm", "pd_dterm", "PID+RRL+LPF_dterm",
                                     "PID+RRL_dterm")})
    return [C.md_table(rows, [("fix (alone, vs original)", "fix"),
                              ("PID Δmedian", "pid_dmed"), ("PID Δterm pp", "pid_dterm"),
                              ("PID Δroughness", "pid_drough"), ("PID Δmotor HF", "pid_dhf"),
                              ("PD Δmedian", "pd_dmed"), ("PD Δterm pp", "pd_dterm"),
                              ("paper PID+RRL+LPF Δmedian", "PID+RRL+LPF_dmed"),
                              ("Δterm pp", "PID+RRL+LPF_dterm"),
                              ("paper PID+RRL Δmedian", "PID+RRL_dmed"),
                              ("Δterm pp", "PID+RRL_dterm"), ("verdict", "verdict")], f),
            "", f"Δmedian = change of the median RMSE on episodes every variant survives; "
            f"Δterm = change of the termination rate (percentage points); Δroughness / "
            f"Δmotor HF = change of the input-roughness and >10 Hz motor-command metrics of "
            f"the baseline. Tracking is 'unchanged' when |Δmedian| < {NEGLIGIBLE_MEDIAN_PCT:g} % "
            f"and |Δterm| < {NEGLIGIBLE_TERM_PP:g} pp for every arm shown; smoothness when "
            f"|Δ| < {NEGLIGIBLE_SMOOTH_PCT:g} % (PD and PID baselines). Paper policies: seed "
            f"{'–' if best_seed is None else int(best_seed)}, flown (not retrained) under the "
            "fix.", ""]


def build_zip(results):
    results = Path(results)
    up = results / "upload"
    up.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    zpath = up / f"study_results_{stamp}.zip"
    n, size = 0, 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(results.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(results)
            parts = rel.parts
            if parts[0] in ("upload", "cache") or "checkpoints" in parts \
                    or f.suffix in (".npz", ".pt"):
                continue
            if f.stat().st_size > 20 * 1024 * 1024:
                continue
            z.write(f, Path(results.name) / rel)
            n += 1
            size += f.stat().st_size
    return zpath, n, size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="study_results")
    p.add_argument("--no_zip", action="store_true")
    a = p.parse_args()
    R = Path(a.results)
    L = ["# Fixes study — summary", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Code: "
         + _code_line(R), ""]

    st = R / "logs" / "stages.tsv"
    if st.is_file():
        L += ["## Stages", "", "| stage | exit | minutes |", "|---|---|---|"]
        for line in st.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                L.append(f"| {parts[0]} | {parts[1]} | {parts[2]} |")
        L.append("")

    q = read_csv(R / "queue_status.csv")
    if q:
        done = [r for r in q if str(r["state"]).startswith("done")]
        defer = [r for r in q if r["state"] == "deferred"]
        fail = [r for r in q if str(r["state"]).startswith("FAILED")]
        L += ["## Training queue", "",
              f"{len(done)} of {len(q)} runs finished, {len(defer)} deferred to a later "
              f"night, {len(fail)} failed."
              + (" Failed: " + ", ".join(r["name"] for r in fail) if fail else ""),
              "", "Re-running `bash study_fixes/run_overnight.sh` continues with the "
              "deferred runs; finished runs are never repeated.", ""]

    L += section(R / "00_provenance" / "PROVENANCE.md", "Provenance")
    L += ["## Q1 — which bugs mattered (each fix alone, no retraining)", ""]
    L += bug_table(R)
    L += section(R / "01_fix_ablation" / "ABLATION.md", "Fix ablation, full tables")
    L += section(R / "02_retrain_fixed" / "RETRAIN.md", "Q1/Q2 — retrained with every fix")
    L += section(R / "03_reward_sweep" / "SWEEP.md", "Q3 — reward × authority sweep")
    L += section(R / "04_presentation" / "FIGURES.md", "Presentation tables and figures")
    L += section(R / "05_diagnosis" / "DIAGNOSIS.md", "Diagnosis: violent chatter and spikes")
    L += section(R / "06_talk" / "TALK.md", "Talk figures and tables")
    L += ["## Notes", "",
          "- Every fix is a switch in `EnvConfig` whose default is the original code "
          "(bit-identical, `tests/test_fixes.py`); `--fixed` turns all of them on.",
          "- The mixer sign fix is a rotor relabelling; with the paper's symmetric rotor "
          "scales it cannot change any result (see its row above).",
          "- `cond` statistics are conditional on survival like the paper's Table 1; "
          "always read them with the termination column.", ""]
    (R / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print(f"wrote {R / 'SUMMARY.md'}")
    if not a.no_zip:
        z, n, size = build_zip(R)
        print(f"wrote {z}  ({n} files, {size / 1e6:.1f} MB before compression, "
              f"{z.stat().st_size / 1e6:.1f} MB zipped)")


def _code_line(R):
    f = R / "00_provenance" / "code_info.json"
    if not f.is_file():
        return "unknown"
    d = json.loads(f.read_text())
    return f"commit `{d.get('commit', '')[:10]}` + study overlay; JAX {d.get('jax')} on " \
           f"{', '.join(d.get('devices', []))}"


if __name__ == "__main__":
    main()
