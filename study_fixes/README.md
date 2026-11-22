# Fixes study (November 2026)

Three questions about the KSAS 2026 paper's code, answered overnight:

1. **Did the bugs found later (in qrjax_adapt / qrjax_ood) matter?** Each fix is
   switched on alone, all together, and all-but-one, first on the baselines
   and the paper's own trained policies (no retraining), then by retraining.
2. **Does the thrust LPF still help once the finite-difference bug is fixed?**
   LPF on/off retrained with every fix, three seeds, with the chatter *tail*
   (per-episode motor high-frequency content against the baseline's) and not
   only means.
3. **Reward sharpness × residual authority.** `tau_pos` {0.5, 0.2, 0.1, 0.05}
   × authority {0.2, 0.4, 0.6} for PID + RRL + LPF, plus no-LPF arms at the
   sharp reward, watching saturation and chatter next to tracking.

## The fixes (all are `EnvConfig` switches; defaults = original code)

| switch | original | fixed | what it fixes |
|---|---|---|---|
| `rate_ff` | `finite_difference` | `analytic` | ω_d, ω̇_d by differentiating the commanded attitude twice numerically → exact derivatives of R_d (Lee et al.) with an ideal accelerometer, ported from qrjax_ood |
| `fd_counter_fix` | `False` | `True` | one flag gated ω_d and ω̇_d: on the 2nd call ω̇_d = ω_d/dt (≈21 rad/s², 0.46 N·m after every reset) |
| `allocator` | `clip` | `prioritized` | per-rotor clipping rotates the moment on infeasible requests → roll/pitch > collective > yaw |
| `mixer_geometry` | `legacy` | `frd` | roll/pitch signs vs the documented rotor positions (qrjax_adapt audit); a pure relabelling with the paper's symmetric rotor scales |
| `start_on_reference` | `False` | `True` | episodes start at R = I, ω = 0 with a cold controller → on the reference attitude and rate, warm controller |
| `term_ref_error`, `term_heading_deg` | off | 3 m, 90° | terminations on true-vs-desired error and heading loss, not only the twin |
| `TrainConfig.eval_fixed_key`, `best_window` | off, 1 | on, 3 | checkpoint chosen on a fresh draw each time by a single minimum → fixed draw, mean of the last 3 |

`--fixed` on `scripts.train` switches all of them on. With every switch at its
default the code is bit-identical to commit `c9d957f` (rollouts with resets,
terminations and saturation, and a short training run, compared array by
array); stage 0 checks the same on your machine by regenerating the paper's
Table 1 from its checkpoints.

## Run

```bash
cd ~/qrjax && source .venv/bin/activate
SMOKE=1 bash study_fixes/run_overnight.sh        # ~10-20 min, tiny budgets, checks every stage
bash study_fixes/run_overnight.sh                # the real run (HOURS=9 by default)
```

Re-running continues where it stopped: finished training runs are never
repeated and evaluations are cached. `STAGES="eval report"` only re-analyses.

Talk ablations (stage 2 groups H, M; 18 runs of ~16 min, ~5 h):

```bash
STAGES="train talk report" TRAIN_GROUPS=H,M bash study_fixes/run_overnight.sh
```

H = history 1 / 20 (10 is the method), M = filter on thrust and torques. `STAGES="talk report"` alone makes the talk
figures from whatever has finished.

## Output (`study_results/`)

```
SUMMARY.md                 everything below, stitched together
00_provenance/             code version, which experiments/ folder is the paper, exact re-evaluation
01_fix_ablation/           baselines and paper policies under each fix (no retraining)
02_retrain_fixed/          Q1 + Q2: retrained with every fix vs the paper's policies, LPF effect, chatter tails
03_reward_sweep/           Q3: tau_pos x authority heatmaps and tables
04_presentation/           slide tables (paper format) and figures, PNG + PDF (STAGES="figures report")
05_diagnosis/              why the unfiltered residual chatters, one-step spikes (STAGES="diagnose report")
06_talk/                   KSAS talk figures + tables, English, fixed code at authority 0.2 (STAGES="talk report")
runs/<job>/train/trial_*/  training runs (configs, metrics, checkpoints)
logs/                      one log per stage and per training run
queue_status.csv           which runs finished, which were deferred
upload/study_results_*.zip results without checkpoints -- send this one back
cache/                     per-arm evaluation cache (safe to delete)
```
