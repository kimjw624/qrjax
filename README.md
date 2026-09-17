# qrjax

Residual reinforcement learning for quadrotor control on SE(3), implemented
end-to-end in JAX.

A geometric SE(3) controller (PD or PID) flies the vehicle; a SAC policy learns
a residual wrench correction on top of it. Physics, controller, environment and
learner are all pure JAX, so an entire training iteration — collect, insert,
learn — compiles to a single XLA graph and runs across hundreds of parallel
environments.

Roughly 90x faster than an equivalent NumPy/PyTorch implementation: 1M
environment steps in about 6 minutes on a laptop GPU, against 8.4 hours.

## What it does

The policy never sees absolute state. It sees the **discrepancy** between the
true (disturbed) vehicle and a disturbance-free nominal twin flying the same
controller. Its job is to push the disturbed vehicle back onto the twin's
behaviour, so it learns disturbance rejection rather than trajectory tracking.

Uncertainty is sampled per episode across five types simultaneously: mass and
inertia scale, external force, thrust-coefficient scale, moment-coefficient
scale, and arm-length scale.

The two baselines differ **only** by their integral gains — `Gains.pd()` is
literally `Gains.pid()` with `ki = kI = 0` — so a PD-vs-PID comparison isolates
integral action rather than comparing two separately written controllers. A
test asserts they are bit-for-bit identical when the integral gains are zero.

## Install

```bash
git clone https://github.com/kimjw624/qrjax.git
cd qrjax

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .
```

That installs the CPU build of JAX. For an NVIDIA GPU, install the CUDA build
**after** `pip install -e .`, since `-e .` pulls plain CPU JAX and would
overwrite the CUDA plugin if run second:

```bash
pip install -U "jax[cuda13]"       # driver with CUDA 13.x
# pip install -U "jax[cuda12]"     # driver with CUDA 12.x
```

Verify:

```bash
python -c "import jax; print(jax.devices())"
```

You want `[CudaDevice(id=0)]`. If it prints `CpuDevice`, JAX did not find CUDA
and will train silently on CPU at roughly a tenth the speed.

### GPU memory

JAX preallocates a fixed arena at startup, sized as a fraction of **total**
VRAM. The trap is that the arena is not the whole story: XLA's autotuner
profiles candidate kernels during compilation and allocates its scratch
buffers **outside** that arena. Set the fraction too high and the arena itself
succeeds while the autotuner starves, which surfaces as a wall of
`CUDA_ERROR_OUT_OF_MEMORY` lines ending in `All configs failed during
profiling` — before a single gradient step runs.

On a 6 GB laptop card, leave the autotuner room:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=.80    # ~4.9 GB arena, ~850 MB spare
```

If that still fails, stop preallocating and let JAX allocate on demand. A
little slower from fragmentation, but the arena and the scratch buffers stop
competing:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Still failing? Turn the autotuner off so it needs no profiling buffers at all,
at the cost of slightly less optimal kernel selection:

```bash
export XLA_FLAGS=--xla_gpu_autotune_level=0
```

As a last resort, shrink the work itself. Both of these change run time, not
results:

```bash
python -m scripts.train --num_envs 128 --buffer_size 150000 ...
```

Do **not** lower `--utd` to save memory. It reduces gradient steps per
collected transition, which changes the learning dynamics and makes runs
incomparable to everything else here.

Convenient shell alias:

```bash
alias qr='cd ~/qrjax && source .venv/bin/activate && export XLA_PYTHON_CLIENT_MEM_FRACTION=.80'
```

## Quick start — reproduce the paper

One command trains every configuration, evaluates all six of them on a shared
episode bank, picks the best seed, and writes the tables and figures:

```bash
python -m scripts.run_final_study --seeds 0,1,2 --total_steps 2000000
```

Twelve training runs — three seeds x two bases x filter on/off — roughly four
hours on an RTX 4050. `--resume` skips any run whose checkpoint already exists,
so an interruption costs only the run that was in flight.

Output lands in `experiments/final_<timestamp>/`:

```
REPORT.md                     tables, effect sizes, paired tests, all seeds
table1.csv  effects.csv  per_episode.csv  paired_tests.csv
01_summary/                   arm comparison, distribution, error-vs-time,
                              control cost, uncertainty coverage
02_cases/                     trajectory, error and actuator figures per case
03_actuators/                 actuator summary and command spectra
figures/                      fig2 presets and fig3 summary
seed_XX/{none,thrust}/        training runs
```

## Training one policy

To train a single configuration instead — PID base, 200 Hz, thrust filter:

```bash
python -m scripts.train \
    --no_curriculum \
    --base_controller pid \
    --control_hz 200 \
    --total_steps 2000000 \
    --num_envs 256 --utd 1.0 \
    --buffer_size 300000 \
    --thrust_filter_beta 0.2 \
    --eval_episodes 128 --eval_every_iters 10 \
    --live_plot
```

Why these values:

| flag | reason |
|---|---|
| `--control_hz 200` | the attitude loop's tolerable gain mismatch is `g_crit = 2J/(dt·kΩ)` — 1.76 at 100 Hz but 3.49 at 200 Hz. The uncertainty box reaches 2.41, so 200 Hz keeps every episode inside the stability margin |
| `--total_steps 2000000` | an episode is 2000 steps at 200 Hz, so this matches the episode count of 1M steps at 100 Hz |
| `--utd 1.0` | one gradient step per collected transition. Lower is faster; see `scripts/benchmark.py` |
| `--thrust_filter_beta 0.2` | low-pass on the thrust residual. Cuts command roughness ~2x at no tracking cost |

Progress is written to `runs/<name>/trial_NNN/`: `metrics.jsonl`,
`progress.png` (redrawn every log interval), and `checkpoints/best.pt`.
`--live_plot` also opens an interactive window; drop it if you are on a
headless machine.

Watch from another terminal:

```bash
python -m scripts.plot_progress --run_dir runs/residual_sac_flat_pid_200Hz/trial_001 --watch
```

## Evaluation

Every comparison is paired: all configurations see the identical disturbance
draw on every episode, and the evaluation aborts if the draws diverge.

`run_final_study` evaluates on a **survivor bank** — candidate disturbance
realisations are drawn in chunks until exactly `--episodes` of them survive in
*every* configuration. That keeps the episode count fixed and the pairing
intact, at a cost worth stating plainly: the reported RMSE is conditional on
survival, so robustness differences vanish from the table. The per-arm
termination rate over the candidate pool is written to `REPORT.md` for exactly
that reason, and belongs next to any table quoted from it.

To evaluate policies you trained separately:

```bash
python -m scripts.evaluate \
    --run_dir runs/residual_sac_flat_pid_200Hz/trial_001 \
    --checkpoint best \
    --arms pd,pid,pid_res \
    --episodes 256

python -m scripts.make_paper_figures \
    --run_dir_pd  <pd-base trial> \
    --run_dir_pid <pid-base trial> \
    --episodes 256
```

Every case figure is labelled with its uncertainty draw — `k`, `|F|`, `kf`,
`arm`, and the effective attitude gain `g = kf·arm/k` with whether it sits
inside the stability margin. Without `g` on the figure, a residual-induced
problem and a baseline gain-margin failure look identical in a raw actuator
trace.

## Multi-seed comparison

`scripts/train.py` trains one policy. To train both bases across seeds and
produce an aggregate report:

```bash
python -m scripts.run_stage1 --seeds 0,1,2 \
    --control_hz 200 --total_steps 2000000 \
    --thrust_filter_beta 0.2 --live_plot
```

Six runs, roughly two hours. One RL run is not evidence — differences of a few
percent sit inside seed-to-seed spread.

## Repository layout

```
src/qrjax/
  core/        so3, dynamics (NED, RK4), controller (PD+PID), trajectory, mixer
  envs/        vectorized twin-plant residual environment, observations, config,
               survivor-bank sampling, paper figure set
  rl/          SAC (Flax/Optax), device-resident replay buffer, curriculum
  train/       jitted training loop, live progress monitor
  utils/       run directories, serialization, manifests

scripts/
  run_final_study.py        everything: train, evaluate, tabulate, plot
  train.py                  train one policy
  run_stage1.py             both bases across seeds, with aggregate report
  run_lpf_study.py          filter ablation: none / thrust / moment / both
  run_history_study.py      observation-history ablation, down to memoryless
  run_stage2_sweep.py       disturbance-frequency sweep
  evaluate.py               paired four-arm comparison
  make_paper_figures.py     complete figure and table set
  plot_trajectories.py      3-D trajectory, actuator and error plots per case
  plot_actuators.py         rotor-command traces with spectra
  plot_actuator_cases.py    per-case baseline / residual / residual+LPF
  plot_progress.py          training curves, live or replotted
  benchmark.py              throughput and UTD sweep for your hardware
  verify_gain_margin.py     stability-margin verification suite
  analyze_lpf.py            filter response, effect and cost
  analyze_stable_chatter.py chatter attribution
  loop_margins.py           gain, phase and vector margins
```

## Configuration notes

**Error convention.** The observation is `true − nominal`, actual relative to
reference, matching the geometric controller's own `e_x = x − x_d`. Attitude is
the exception and not by choice: SO(3) is not a vector space, so `R − R_nom` is
not a rotation and carries no meaning. It uses the geometric error

```
e_R = ½ ( R_nomᵀ R − Rᵀ R_nom )^∨
```

the same map the controller applies, with the nominal attitude as the
reference. A test asserts all four blocks keep this sign.

**Low-pass filter.** `u_t = (1−β)·u_{t−1} + β·a_t`, a first-order IIR filter
with `τ = −dt/ln(1−β)` and `f_c = 1/(2πτ)`. **β is not rate-independent**:
β=0.2 gives 3.57 Hz at 100 Hz but 7.13 Hz at 200 Hz. To hold a cutoff fixed
across rates use `β = 1 − exp(−2π·f_c·dt)`.

**Frozen evaluation bank.** `--eval_seed` defines the disturbance draws. Keep
it fixed across every experiment you intend to compare. The bank is also stable
against code changes: disturbance quantities are drawn from named `fold_in`
sub-keys with permanent ids, so adding a new sampled quantity does not
regenerate the existing draws.

**Metrics.** Terminated episodes are charged at the 2.0 m termination threshold
rather than masked out, so the mean folds tracking quality and failure rate
into one number suitable for checkpoint selection. Read the **median** for
typical tracking and the termination rate for robustness.

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

40 tests covering SO(3) NaN-safety, PD/PID equivalence, mixer round-trips,
environment auto-reset and pairing, replay-buffer correctness, filter response,
and disturbance-bank stability.

## References

The implementation follows these directly.

**Geometric control on SE(3)** — the baseline controller. `Gains.pd()`
implements [1]; `Gains.pid()` adds the integral terms of [2], including the
saturated translational integral and the `c1`, `c2` error-mixing constants.

1. T. Lee, M. Leok, and N. H. McClamroch, "Geometric tracking control of a
   quadrotor UAV on SE(3)," *49th IEEE Conference on Decision and Control
   (CDC)*, 2010, pp. 5420–5425. [arXiv:1003.2005](https://arxiv.org/abs/1003.2005)

2. F. Goodarzi, D. Lee, and T. Lee, "Geometric nonlinear PID control of a
   quadrotor UAV on SE(3)," *European Control Conference (ECC)*, 2013,
   pp. 3845–3850. [arXiv:1304.6765](https://arxiv.org/abs/1304.6765)

**Residual reinforcement learning** — the learning formulation. A policy learns
a corrective term on top of a conventional feedback controller rather than
replacing it. Introduced independently and concurrently by [3] and [4].

3. T. Johannink, S. Bahl, A. Nair, J. Luo, A. Kumar, M. Loskyll, J. A. Ojea,
   E. Solowjow, and S. Levine, "Residual reinforcement learning for robot
   control," *IEEE International Conference on Robotics and Automation (ICRA)*,
   2019, pp. 6023–6029. [arXiv:1812.03201](https://arxiv.org/abs/1812.03201)

4. T. Silver, K. Allen, J. Tenenbaum, and L. Kaelbling, "Residual policy
   learning," 2018. [arXiv:1812.06298](https://arxiv.org/abs/1812.06298)

**Model-reference reinforcement learning** — the twin-plant structure used
here. A nominal, disturbance-free system defines the desired behaviour, and the
learned term compensates the deviation of the true system from it. The
observation and reward in this repository are both built from that discrepancy,
so the formulation is model-reference RL applied to a quadrotor on SE(3).

5. Q. Zhang, W. Pan, and V. Reppa, "Model-reference reinforcement learning
   control of autonomous surface vehicles with uncertainties," 2020.
   [arXiv:2003.13839](https://arxiv.org/abs/2003.13839)

6. Q. Zhang, W. Pan, and V. Reppa, "Model-reference reinforcement learning for
   collision-free tracking control of autonomous surface vehicles," 2020.
   [arXiv:2008.07240](https://arxiv.org/abs/2008.07240)

### Related work

Not implemented here, but directly relevant to the residual-plus-filter
structure. [7] is the closest classical counterpart: the same geometric
baseline with an L1 adaptive augmentation in place of a learned residual, and
the same use of a low-pass filter to decouple adaptation speed from robustness.

7. Z. Wu, S. Cheng, P. Zhao, A. Gahlawat, K. A. Ackerman, A. Lakshmanan,
   C. Yang, J. Yu, and N. Hovakimyan, "L1Quad: L1 adaptive augmentation of
   geometric control for agile quadrotors with performance guarantees," 2023.
   [arXiv:2302.07208](https://arxiv.org/abs/2302.07208)

8. Y. Cheng, P. Zhao, F. Wang, D. J. Block, and N. Hovakimyan, "Improving the
   robustness of reinforcement learning policies with L1 adaptive control,"
   *IEEE Robotics and Automation Letters*, 2022.
   [arXiv:2112.01953](https://arxiv.org/abs/2112.01953)

9. K. Huang, R. Rana, A. Spitzer, G. Shi, and B. Boots, "DATT: Deep adaptive
   trajectory tracking for quadrotor control," *Conference on Robot Learning
   (CoRL)*, 2023. [arXiv:2310.09053](https://arxiv.org/abs/2310.09053)

10. M. O'Connell, G. Shi, X. Shi, K. Azizzadenesheli, A. Anandkumar, Y. Yue,
    and S.-J. Chung, "Neural-Fly enables rapid learning for agile flight in
    strong winds," *Science Robotics*, vol. 7, no. 66, 2022.


## Requirements

Python 3.10+, JAX, Flax, Optax, NumPy, Matplotlib. See `pyproject.toml`.
