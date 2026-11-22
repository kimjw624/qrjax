# Maintaining qrjax, qrjax_ood and qrjax_adapt

State of 22 Nov 2026. One page on what each repository is for, which fixes each
one has, what they share and how changes move between them.

## The three repositories

| repo | package | GitHub | role | status |
|---|---|---|---|---|
| `qrjax` | `qrjax` | `kimjw624/qrjax`, public | the residual-RL method of the submitted paper; the base for new method work | **active**. Tag `paper-submission` = the code the paper ran; v2 defaults = the fixed code |
| `qrjax_ood` | `qrood` | `kimjw624/qrjax_ood`, private | out-of-distribution study: sensor noise, actuator lag/delay, adaptation, oracles, predictors, runtime monitors; report F1–F13 in `docs/report.pdf` | **active** research |
| `qrjax_adapt` | `qrjax` (distribution `qrjax-core`) | `kimjw624/qrjax_adapt`, private | learned one-step predictor for OOD detection | **archived**, superseded by `qrjax_ood` |

`qrjax_adapt` imports as `qrjax`: never install it next to the real qrjax;
run it with `PYTHONPATH=$PWD/src`.

## Fixes: which repository has what

**default** = on by default · **option** = available, off by default · **—** = not there

| # | issue | qrjax v2 | qrjax_ood | qrjax_adapt |
|---|---|---|---|---|
| 1 | desired body rates by double finite difference of R_d (noise ÷ dt², the extreme-oscillation loop of the fixes study) | default (`rate_ff="analytic"`) | default | — |
| 2 | one flag gates ω_d and ω̇_d → impulse after every reset | default (`fd_counter_fix`) | default | — |
| 3 | per-rotor clipping rotates the moment when saturated | default (`allocator="prioritized"`) | default | — |
| 4 | mixer roll/pitch signs vs documented rotor positions (a relabelling) | default (`mixer_geometry="frd"`) | default | default |
| 5 | episode starts at rest, R = I, while the reference moves | default (`start_on_reference`) | default (+ hover/ramp-in start) | — |
| 6 | crashes shared with the twin go unseen | default (`term_ref_error`, `term_heading_deg`) | default | n/a |
| 7 | fresh evaluation draw each time, single-best checkpoint | default (`eval_fixed_key`, `best_window=3`) | default | n/a |
| 8 | thrust filter defined per step (β), so it changes with the rate | option (`thrust_filter_tau`) | default (τ) | n/a |
| 9 | float32 hard-coded | — | default (`--x64`) | default |
| 10 | unflyable episodes counted as controller failures | label: `qrjax.envs.feasibility` | default (`qrood.eval.feasibility`) | n/a |
| 11 | integrator clip makes Ȧ jump → one-step moment kick | option (`pid_ei_mode="smooth"`) | option (same) | — |
| 12 | disturbances switch on as a step at t = 0 → rotors saturate for the first steps | option (`disturbance_ramp_s`) | option (same) | — |

Not needed in qrjax: qrjax_ood's kΩ 2.54 → 1.27 (the sampled-data gain margin
is exceeded at 100 Hz, not at qrjax's 200 Hz). The reward width `tau_pos` is a
setting, not a fix; qrjax keeps the paper's 0.5.

Measured effect of 11 and 12 (qrjax v2 defaults, PID baseline without residual,
256 episodes of the flat distribution, eval seed 20260829, CPU):

| | rotor-saturated steps in the first 0.1 s (per episode) | largest moment step in the first 0.1 s | largest moment step at an integrator-bound onset | median RMSE |
|---|---|---|---|---|
| v2 defaults | 1.66 | 7.78 N·m | 0.175 N·m | 4.04 cm |
| + `pid_ei_mode="smooth"` | 1.66 | 7.78 N·m | 0.00008 N·m | 4.06 cm |
| + `disturbance_ramp_s=0.5` | 0 | 0.004 N·m | 0.171 N·m | 4.02 cm |
| + both | 0 | 0.004 N·m | 0.00008 N·m | 4.04 cm |

Both are off by default so that existing results stay what they are; turning
them on is a separate decision (it changes every number slightly).

## Deliberate differences (not bugs)

- **Uncertainty model.** qrjax: one factor k for mass and inertia together,
  per-rotor thrust / yaw-torque / arm scales, force ±3 N per axis (optionally
  oscillating). qrjax_ood: mass and per-axis inertia drawn separately, rotor
  coefficients ±10 %, horizontal force ≤ 3 N, plus optional CoM offset,
  vertical force, drag, oscillation and delayed onset.
- **Timing and task.** qrjax: policy at 200 Hz, 10 s 3D figure-8. qrjax_ood:
  200 Hz physics, policy at 100 Hz, 5 s flat figure-8, sensor noise on,
  kΩ = 1.27.
- **Only in qrjax_ood:** sensor noise, motor lag/delay, online adaptation,
  oracles, gain-tuning action mode, predictors and monitors.
  **Only in qrjax:** PD baseline, curriculum, the 512-condition evaluation
  bank, the paper's figures, `study_fixes/`.

## Shared code

The two active repositories keep their own copies of a common core (decision
of Nov 2026: copies plus a check, no shared package). Meant to be identical:
`core/so3.py`, `rl/sac.py`, `rl/networks.py`, and in `core/controller.py` the
functions `_unit_derivs`, `attitude_from_thrust`, `integrator_projection`.
`core/dynamics.py` has the same equations under different helper names;
`core/mixer.py` differs by design (uncertainty model).

```bash
python tools/check_shared.py          # compares ~/qrjax and ~/qrjax_ood
```

The same script is in both repositories. When you change a shared unit in
one, port it to the other in the same session and run the check.

## How changes move (no GitHub connection to Claude)

1. Before a session: `git -C ~/qrjax_ood bundle create ~/qrjax_ood.bundle --all`
   (same for any private repo) and upload the bundle. qrjax is public and is
   read from GitHub directly.
2. Claude commits on top of your exact history and sends back
   `<repo>_update_<date>.bundle`.
3. You apply and publish it:

```bash
cd ~/<repo>
B=$(find ~ \( -name .venv -o -name .cache -o -name .git \) -prune -o -name '<repo>_update_*.bundle' -print | sort | tail -1)
git bundle verify "$B" && git pull --ff-only "$B" main && git push
```

`--ff-only` refuses if your local branch moved in the meantime, so nothing is
silently merged.
