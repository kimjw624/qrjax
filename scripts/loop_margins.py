"""Measure gain and phase margin of the RESIDUAL loop by signal injection.

The residual is a nonlinear state-feedback controller: Delta_f = pi(obs), and
obs is the twin discrepancy, which is plant state. It is NOT feedforward -- a
feedforward term is a function of time or of the reference, cannot be
influenced by the plant, and therefore cannot destabilise anything. Evidence:
discarding the residual's action while still computing it from the real
observation halves the measured chatter, which an exogenous signal could not
do.

Because pi is nonlinear, L(z) cannot be written down, but it can be measured.
Break the loop at the residual output, inject a probe, and compare the signal
entering the plant with the signal leaving the policy:

        p (probe)
           |
           v
    pi --> + --> plant --> obs --> pi
    a      u

    L(jw) = -a_hat(jw) / u_hat(jw)

evaluated at the probe frequency by DFT. Gain margin is 1/|L| where the phase
crosses -180 degrees; phase margin is 180 + angle(L) where |L| = 1.

Two things make this a DESCRIBING FUNCTION rather than a transfer function:

  * L depends on probe amplitude, because pi is nonlinear. The sweep therefore
    covers amplitude as well as frequency, and a margin quoted without an
    amplitude is meaningless.
  * L depends on the operating point. The vehicle is flying a periodic
    trajectory with a per-episode disturbance, so L is measured per episode and
    compared across episodes rather than treated as a single system property.

The probe must be small enough to stay in the local linear regime and large
enough to rise above the policy's own activity; --probe_amp sweeps this so the
sensitivity is visible instead of assumed.

Example
-------
python -m scripts.loop_margins \\
    --run_dir experiments/stage3_nofilter/seed_00/pd/train/trial_001
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
import numpy as np                                   # noqa: E402

from qrjax.envs import EnvConfig, VecEnv             # noqa: E402
from qrjax.rl.curriculum import flat_ranges          # noqa: E402
from qrjax.utils import write_manifest               # noqa: E402

from scripts.analyze_stable_chatter import (         # noqa: E402
    critical_gain, hp_component, load_policy,
)

J_NOM, KR, KOMEGA = 0.022, 8.81, 2.54


def baseline_margins(dt):
    """Classical margins of the linearised attitude loop, for reference."""
    Ad = np.array([[1.0, dt], [0.0, 1.0]])
    Bd = np.array([[0.5 * dt ** 2 / J_NOM], [dt / J_NOM]])
    K = np.array([[KR, KOMEGA]])

    def L(w):
        z = np.exp(1j * w * dt)
        return complex((K @ np.linalg.solve(z * np.eye(2) - Ad, Bd))[0, 0])

    w = np.logspace(-1, np.log10(np.pi / dt), 100000)
    Lv = np.array([L(x) for x in w])
    mag, ph = np.abs(Lv), np.unwrap(np.angle(Lv)) * 180 / np.pi
    i = int(np.argmin(np.abs(mag - 1.0)))
    j = int(np.argmin(np.abs(ph + 180)))
    return {"gain_crossover_hz": w[i] / (2 * np.pi),
            "phase_margin_deg": 180 + ph[i],
            "phase_crossover_hz": w[j] / (2 * np.pi),
            "gain_margin_x": 1.0 / mag[j]}


def measure_loop(cfg, agent, params, key, episodes, freqs, amp, beta,
                 settle=300, capture=700):
    """Inject a sinusoid at the residual output; return L(jw) per episode.

    `settle` steps are discarded so the probe response is stationary before the
    DFT window opens.
    """
    e = cfg.env
    dt = e.dt
    ec = EnvConfig(**{**e.__dict__})
    ec.thrust_filter_beta = beta
    ec.episode_steps = settle + capture
    env = VecEnv(ec, episodes)
    ranges = env.broadcast_ranges(
        flat_ranges(ec, force_freq=0.0, force_dc_prob=0.0))

    out = {}
    for f in freqs:
        def run(key, f=f):
            k_reset, _ = jax.random.split(key)
            state, obs = env.reset(k_reset, ranges, stagger=False)
            g = (state.mixer_true.kf_scale[:, 0]
                 * state.mixer_true.arm_scale[:, 0] / state.k)

            def body(carry, t):
                state, obs = carry
                a = jnp.tanh(agent.actor.apply(params, obs)[0])
                probe = amp * jnp.sin(2 * jnp.pi * f * t * dt)
                u = a.at[:, 0].add(probe)
                state, obs, _, _, info = env.step(state, u, ranges)
                return (state, obs), (a[:, 0], u[:, 0])

            (_, _), (a_seq, u_seq) = jax.lax.scan(
                body, (state, obs), jnp.arange(ec.episode_steps))
            return a_seq, u_seq, g

        a_seq, u_seq, g = jax.jit(run)(key)
        a_seq, u_seq = np.asarray(a_seq)[settle:], np.asarray(u_seq)[settle:]
        n = a_seq.shape[0]
        t = np.arange(n) * dt
        # Single-bin DFT at the probe frequency: robust to the policy's own
        # broadband activity, which is incoherent with the probe and averages
        # out over the window.
        ref = np.exp(-2j * np.pi * f * t)[:, None]
        a_hat = (a_seq * ref).sum(axis=0)
        u_hat = (u_seq * ref).sum(axis=0)
        out[f] = (-a_hat / np.where(np.abs(u_hat) < 1e-12, 1e-12, u_hat),
                  np.asarray(g))
    return out


def margins_from_L(freqs, Lv):
    """Gain and phase margin from a sampled loop transfer."""
    mag, ph = np.abs(Lv), np.unwrap(np.angle(Lv)) * 180 / np.pi
    gm = pm = np.nan
    for i in range(len(freqs) - 1):
        if (mag[i] - 1) * (mag[i + 1] - 1) < 0:
            w = (1 - mag[i]) / (mag[i + 1] - mag[i])
            pm = 180 + ph[i] + w * (ph[i + 1] - ph[i])
            break
    for i in range(len(freqs) - 1):
        if (ph[i] + 180) * (ph[i + 1] + 180) < 0:
            w = (-180 - ph[i]) / (ph[i + 1] - ph[i])
            gm = 1.0 / (mag[i] + w * (mag[i + 1] - mag[i]))
            break
    return gm, pm


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--freqs", default="1,2,3,5,8,12,18,25,33,42,50",
                   type=lambda s: [float(x) for x in s.split(",") if x.strip()])
    p.add_argument("--probe_amp", default="0.02,0.05,0.10",
                   type=lambda s: [float(x) for x in s.split(",") if x.strip()])
    p.add_argument("--beta", type=float, default=1.0,
                   help="thrust filter during the measurement. 1.0 measures "
                        "the unfiltered loop; 0.2 shows what the filter buys "
                        "in margin")
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg, agent, params, _ = load_policy(args.run_dir, args.checkpoint)
    dt = cfg.env.dt
    gc = critical_gain(dt)
    out = Path(args.out) if args.out else Path("experiments") / (
        f"loop_margins_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    (out / "plots").mkdir(parents=True, exist_ok=True)

    bm = baseline_margins(dt)
    print(f"BASELINE attitude loop (linear, LTI):")
    print(f"  gain crossover  {bm['gain_crossover_hz']:.2f} Hz   "
          f"phase margin {bm['phase_margin_deg']:.1f} deg")
    print(f"  phase crossover {bm['phase_crossover_hz']:.2f} Hz   "
          f"gain margin  {20*np.log10(bm['gain_margin_x']):.2f} dB "
          f"({bm['gain_margin_x']:.2f}x)")
    print(f"  that gain margin IS g_crit = {gc:.2f}: kf*arm/k is a "
          "multiplicative\n  gain perturbation at the plant input.\n")

    key = jax.random.PRNGKey(args.eval_seed)
    rows, curves = [], {}
    for amp in args.probe_amp:
        data = measure_loop(cfg, agent, params, key, args.episodes,
                            args.freqs, amp, args.beta)
        g = data[args.freqs[0]][1]
        stable = g < gc
        Lmat = np.stack([data[f][0] for f in args.freqs])       # (F, E)
        curves[amp] = (Lmat, g, stable)

        gms, pms = [], []
        for e in range(Lmat.shape[1]):
            gm, pm = margins_from_L(args.freqs, Lmat[:, e])
            gms.append(gm)
            pms.append(pm)
        gms, pms = np.array(gms), np.array(pms)
        rows.append({"probe_amp": amp,
                     "median_gain_margin_x": float(np.nanmedian(gms[stable])),
                     "median_phase_margin_deg": float(np.nanmedian(pms[stable])),
                     "frac_gain_margin_below_2": float(
                         np.nanmean(gms[stable] < 2.0)),
                     "peak_abs_L": float(np.abs(Lmat[:, stable]).max())})
        print(f"  probe amp {amp:<5} median gain margin "
              f"{np.nanmedian(gms[stable]):6.2f}x   median phase margin "
              f"{np.nanmedian(pms[stable]):6.1f} deg   peak |L| "
              f"{np.abs(Lmat[:, stable]).max():.2f}")

    with (out / "margins.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    amp0 = args.probe_amp[len(args.probe_amp) // 2]
    Lmat, g, stable = curves[amp0]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    med = np.median(np.abs(Lmat[:, stable]), axis=1)
    axes[0].loglog(args.freqs, med, marker="o", lw=2, label="residual loop")
    axes[0].axhline(1.0, color="k", ls="--", lw=1)
    axes[0].set_ylabel("|L|")
    axes[0].set_title(f"Residual loop transfer, probe amplitude {amp0:g}\n"
                      "measured by injection; describing function, not an "
                      "LTI transfer")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")
    phm = np.median(np.unwrap(np.angle(Lmat[:, stable]), axis=0), axis=1)
    axes[1].semilogx(args.freqs, phm * 180 / np.pi, marker="o", lw=2)
    axes[1].axhline(-180, color="k", ls="--", lw=1)
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("phase [deg]")
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "plots" / "residual_loop_bode.png", dpi=150)
    plt.close(fig)

    L = ["# Loop margins\n",
         f"Generated {datetime.now().isoformat(timespec='seconds')}",
         "",
         "## Baseline attitude loop (LTI, classical)", "",
         f"- gain crossover {bm['gain_crossover_hz']:.2f} Hz, "
         f"**phase margin {bm['phase_margin_deg']:.1f} deg**",
         f"- phase crossover {bm['phase_crossover_hz']:.2f} Hz, "
         f"**gain margin {20*np.log10(bm['gain_margin_x']):.2f} dB "
         f"({bm['gain_margin_x']:.2f}x)**",
         "",
         f"The gain margin equals g_crit = {gc:.2f}, and that is not a "
         "coincidence: allocating with nominal kf and arm while the true "
         "vehicle differs is exactly a multiplicative gain error at the plant "
         "input, which is the perturbation gain margin measures. So "
         "'the parameter mismatch exceeded the gain margin' and "
         "'g exceeded g_crit' are the same statement.",
         "",
         "## Residual loop (nonlinear, measured)", "",
         "The residual is state feedback, so a margin is well defined, but pi "
         "is nonlinear so L must be measured rather than derived, and it "
         "depends on probe amplitude. A margin quoted without an amplitude is "
         "meaningless.",
         "",
         "| probe amplitude | median gain margin | median phase margin | peak |L| |",
         "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['probe_amp']:g} | {r['median_gain_margin_x']:.2f}x "
                 f"| {r['median_phase_margin_deg']:.1f} deg "
                 f"| {r['peak_abs_L']:.2f} |")
    L += ["",
          "## Caveats", "",
          "- This is a describing-function measurement at one operating point "
          "per episode, on a vehicle flying a periodic trajectory with a "
          "per-episode disturbance. It is not a single system property.",
          "- The probe must sit above the policy's own broadband activity and "
          "below the amplitude at which the network leaves its local linear "
          "regime. Sweeping amplitude is how you check both.",
          "- A fully rigorous treatment of the periodic trajectory would use "
          "Floquet analysis: linearise the closed loop about the nominal "
          "periodic solution, propagate the state-transition matrix over one "
          "10 s period, and read the multipliers. That is exact but requires "
          "the augmented state (plant, controller memory, and the 10-frame "
          "observation history), roughly a 200x200 monodromy matrix.",
          "",
          "## Files", "", "- `margins.csv`",
          "- `plots/residual_loop_bode.png`"]
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    write_manifest(out, {"run_dir": args.run_dir, "beta": args.beta,
                         "freqs": args.freqs, "probe_amps": args.probe_amp,
                         "baseline_margins": bm, "g_crit": gc})
    print(f"\nwrote {out}/REPORT.md")


if __name__ == "__main__":
    main()
