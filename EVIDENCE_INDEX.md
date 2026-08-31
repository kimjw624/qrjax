# Evidence index — actuator oscillation

Two distinct phenomena, with different causes, different fixes, and different
evidence. Keeping them apart is the whole point; a single "oscillation" story
cannot explain both, because in one the residual is the culprit and in the
other it is the remedy.

| | Phenomenon A | Phenomenon B |
|---|---|---|
| Where | g < g_crit (~97% of the box) | g > g_crit (~3%) |
| What | residual self-oscillation, 2.6x baseline | baseline limit cycle, 16x baseline |
| Cause | policy loop gain through the plant | discrete gain margin of the attitude loop |
| Residual's role | **causes it** | **suppresses it** (7x) |
| LPF | **fixes it completely** | makes it worse |
| Fix | low-pass the residual command | adaptive gain, or faster loop |

---

## Phenomenon B — baseline gain-margin failure

### B1. Mechanism

The controller allocates with nominal rotor constants and arm lengths, and its
attitude gains assume nominal inertia. The true loop gain is therefore

```
g = kf · arm / k
```

**Figure:** `summary_figure.png` panel A
**Command:** `python -m scripts.verify_gain_margin`
**Table:** `gain_margin_*/REPORT.md` → "T1/T2 dose response"

| g | 50 Hz | 100 Hz | 200 Hz |
|---|---|---|---|
| 1.0 | **3.71** | 0.195 | 0.136 |
| 1.8 | 3.77 | 0.585 | 0.139 |
| 2.2 | 3.82 | **3.70** | 0.147 |
| 3.6 | 3.86 | 3.82 | **3.48** |

### B2. It is a discrete-time effect, not high gain per se

Continuous time: `θ̈ + (g·kΩ/J)θ̇ + (g·kR/J)θ = 0`. Both natural frequency and
damping ratio grow as √g, so the loop is **stable for every g** — and gets more
damped as g rises.

| g | natural freq | damping ratio |
|---|---|---|
| 1 | 3.18 Hz | 2.88 |
| 3 | 5.52 Hz | 5.00 |
| 5 | 7.12 Hz | 6.45 |

The instability only appears once the command is held for a full timestep. The
closed-form threshold:

```
g_crit = 2J / (dt·(kΩ − dt·kR/2))  ≈  2J / (dt·kΩ)
```

| rate | closed form | eigenvalue solve |
|---|---|---|
| 50 Hz | 0.90 | 0.87 |
| 100 Hz | 1.76 | 1.73 |
| 200 Hz | 3.49 | 3.46 |
| 400 Hz | 6.96 | 6.93 |

`g_crit ∝ 1/dt`, so doubling the control rate doubles the tolerable mismatch.
This is why raising the frequency is a valid fix, and why continuous-time
analysis (dt → 0, g_crit → ∞) sees no problem at all.

### B3. Causality

Allocating with the true rotor constants and arm lengths drops g from `kf·arm/k`
to `1/k`. Note this does **not** correct the inertia error, so g becomes 1/k
(= 1.41 at k = 0.71), not 1 — below the 1.76 limit, which is why it stabilises.

**Figure:** `summary_figure.png` panel B
**Table:** `gain_margin_*/REPORT.md` → "T4"

| g before | nominal allocation | true allocation | reduction |
|---|---|---|---|
| 2.2 | 2.612 N | 0.179 N | 14.6x |
| 2.6 | 3.068 N | 0.204 N | 15.0x |
| 3.0 | 2.750 N | 0.225 N | 12.2x |

So the parameters are not the problem. The **mismatch between assumed and true
parameters** is.

### B4. It is not the integral term, and not the residual

**Table:** `gain_margin_*/REPORT.md` → "T5"

| g | PD | PID |
|---|---|---|
| 1.0 | 0.1523 | 0.1521 |
| 2.2 | 3.8045 | 3.8045 |

Identical. Write "geometric baseline controller", not "PID controller".

Separately, on affected episodes the bare baseline oscillates as hard as
baseline-plus-residual, and the residual **reduces** it — see B6.

### B5. Exposure

**Figure:** `summary_figure.png` panel C

With kf, arm and k each uniform on [0.7, 1.3]:

- g ranges **0.38 to 2.40**, median 0.99
- **3.1%** of draws exceed g_crit = 1.76 at 100 Hz
- the worst corner, 1.3·1.3/0.7 = **2.41**, sits well past it
- observed severe-episode rate is ~4–5%, higher than 3.1% because marginal
  cases near the threshold also misbehave

Narrowing every factor to about [0.85, 1.18] keeps g under the limit
everywhere.

### B6. The residual helps here, and the LPF hurts

**Figure:** `beyond_gain_margin_ep109.png`
**Command:** `python -m scripts.plot_actuator_cases`

Episode 109, g = 2.15:

| | rotor HF |
|---|---|
| baseline alone | **2.582 N** |
| + residual | 0.360 N |
| + residual + LPF | 1.154 N |

The residual suppresses the baseline's limit cycle 7x. Adding the filter
removes authority it was using to do that, and the oscillation partly returns.
The LPF is **not** a fix for phenomenon B.

---

## Phenomenon A — residual self-oscillation in the stable region

### A1. It exists, and it is the residual's doing

**Figure:** `worst_added_chatter_ep043.png` (rows 1 → 2)
**Table:** `stable_chatter_*/REPORT.md` → "Arms"

Stable region only (123 of 128 episodes), PD base:

| arm | motor HF | **base wrench HF** | **residual wrench HF** | sat % | RMSE |
|---|---|---|---|---|---|
| baseline, no residual | 0.1262 | 0.1171 | 0.0000 | 0.47 | 0.1181 |
| residual, LPF off (matched) | 0.3638 | 0.0964 | **0.7762** | 1.37 | 0.0575 |
| residual, LPF on | 0.1631 | 0.0619 | 0.1995 | 0.51 | 0.0580 |

**2.6x** the baseline's chatter. The attribution is the key column pair: the
base wrench HF **falls** (0.1171 → 0.0964) while the residual wrench carries
0.7762 N. The baseline is not reacting to a perturbed state — it is calmer,
because the residual is absorbing the disturbance. All the excess is the
residual's own command.

So this is the policy closing a high-gain loop through the plant
(policy → thrust → velocity → observation → policy), not an interaction with
the controller.

### A2. It has an authority threshold

**Figure:** `authority_dose_response.png`
**Table:** `stable_chatter_*/authority_sweep.csv`

| authority | residual HF | motor HF |
|---|---|---|
| 1.00 | 1.4310 | 0.8949 |
| 0.50 | 0.3055 | 0.1924 |
| 0.25 | 0.1217 | 0.1334 |
| 0.10 | 0.0435 | 0.1252 |

Halving authority cuts chatter **4.7x**, not 2x. Below 0.5 the response is
roughly proportional; above it, something switches on. That is a loop-gain
threshold, the same kind of mechanism as B but with the policy as controller.
At authority 0.10, motor HF equals the no-residual floor.

### A3. The LPF removes it

**Figure:** `worst_added_chatter_ep043.png` (rows 2 → 3)

Episode 43, g = 1.46, inside the margin:

| | rotor HF | saturation | RMSE |
|---|---|---|---|
| baseline | 0.162 N | 0.3% | 0.1810 m |
| + residual, no LPF | 2.390 N | 20.6% | 0.0883 m |
| + residual + LPF | 0.251 N | 0.7% | **0.0667 m** |

14.7x more chatter and 69x more saturation from the residual; the filter takes
back 9.5x and 29x of it **and improves tracking**. The chatter was not buying
performance — at 20% saturation the commanded wrench is not what gets
delivered.

### A4. Filter thrust first; moments are a small second-order gain

**Command:** `python -m scripts.analyze_stable_chatter`

| thrust β | moment β | motor HF | thrust residual HF | moment residual HF | sat % |
|---|---|---|---|---|---|
| 1.0 | 1.0 | 0.3638 | 0.7762 | 0.0344 | 1.37 |
| 1.0 | 0.2 | 0.3307 | 0.7205 | 0.0111 | 2.29 |
| 0.2 | 1.0 | 0.1631 | 0.1995 | 0.0212 | 0.51 |
| 0.2 | 0.2 | **0.1286** | 0.1983 | 0.0102 | 0.47 |
| baseline | — | 0.1262 | — | — | 0.47 |

Moment filtering **alone** buys 9%; thrust filtering alone buys 2.2x. The moment
residual carries 22x less high-frequency energy, so there is little to remove.
Filtering both reaches the baseline floor (0.1286 vs 0.1262) at no tracking
cost — worth doing, but as an addition to thrust filtering, not a replacement.

### A5. Filter response

**Figure:** `lpf_*/plots/A_filter_response.png`
**Command:** `python -m scripts.analyze_lpf`

`u_t = 0.8·u_(t−1) + 0.2·a_t`, −3 dB at **3.57 Hz**. Analytic prediction matches
the environment exactly at every frequency including Nyquist:

| freq | predicted | measured |
|---|---|---|
| 1 Hz | 0.9627 | 0.9627 |
| 10 Hz | 0.3402 | 0.3402 |
| 50 Hz | 0.1111 | 0.1111 |

---

## Still open

- Why the policy learns a loop gain past its own stability threshold at all.
  Nothing in the reward penalises it heavily (`w_action_smooth = 0.01`), and
  the simulator has no motor dynamics, so high-frequency commands are free.
  Adding actuator lag or a stronger smoothness penalty would test this.
- Whether A2's threshold has a closed form like B2's. The analogous
  calculation for the thrust loop (policy gain, 2 integrators, one-step delay)
  is tractable and would predict the authority at which chatter starts.
- Whether adaptive gain recovers phenomenon B in practice. Panel B is the
  upper bound: perfect knowledge of kf and arm gives ~19x.

## Reproduce everything

```bash
python -m scripts.verify_gain_margin --run_dir <pid trial>
python -m scripts.analyze_lpf --run_dir <trial> --run_dir_nofilter <trial@1.0>
python -m scripts.analyze_stable_chatter --run_dir <trial> --run_dir_nofilter <trial@1.0>
python -m scripts.plot_actuator_cases --run_dir <trial> --run_dir_nofilter <trial@1.0>
python -m scripts.make_summary_figure --gain_margin <dir> --lpf <dir>
```
