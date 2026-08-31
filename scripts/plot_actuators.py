"""Plot actuator traces for a trained policy, to inspect oscillation by eye.

Plots the PER-ROTOR THRUST COMMANDS, not the normalized policy action. That is
the physical actuator signal an ESC would have to track, and it is where
oscillation actually matters. The policy action is one linear map away, but the
motor command also carries the baseline controller's own activity and the
mixer's saturation, both of which change the picture.

Episodes are chosen by measured high-frequency content rather than at random,
so the figures show the range rather than a lucky sample: the worst episode,
the 75th percentile, the median, and the best. If the worst episode looks
clean, the policy is clean.

Zoom matters. A 10 s episode at 100 Hz is 1000 points, and chatter at 10-50 Hz
renders as a solid band at that scale. Every figure therefore pairs the full
episode with a 1 s window, and adds a power spectrum with the control rate and
Nyquist marked.

Example
-------
python -m scripts.plot_actuators \\
    --run_dir experiments/stage2_mixed_net512/seed_00/pid/train/trial_001 \\
    --thrust_filter_beta 0.2 --episodes 128
"""

import argparse
import json
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

from qrjax.core import mixer                         # noqa: E402
from qrjax.envs import Config, EnvConfig, VecEnv     # noqa: E402
from qrjax.rl import SAC                             # noqa: E402
from qrjax.rl.curriculum import flat_ranges          # noqa: E402
from qrjax.utils import load_params, write_manifest  # noqa: E402

ROTOR_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]


def load_policy(run_dir, checkpoint="best"):
    run_dir = Path(run_dir)
    cfg = Config.from_dict(json.loads((run_dir / "config.json").read_text()))
    ckpt = run_dir / "checkpoints" / (
        checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt")
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    probe = VecEnv(cfg.env, 1)
    agent = SAC(probe.obs_dim, probe.action_dim, cfg.sac)
    params = load_params(ckpt, agent.init(jax.random.PRNGKey(0)).actor_params)
    return cfg, agent, params


def rollout(env_cfg, controller, agent, params, key, episodes, ranges,
            use_policy=True):
    cfg = EnvConfig(**{**env_cfg.__dict__})
    cfg.base_controller = controller
    env = VecEnv(cfg, episodes)
    batched = env.broadcast_ranges(ranges)

    def run(key):
        k_reset, _ = jax.random.split(key)
        state, obs = env.reset(k_reset, batched, stagger=False)
        draw = {"k": state.k,
                "force_amp": jnp.linalg.norm(state.external_force, axis=-1),
                "motor": state.mixer_true.kf_scale.mean(axis=-1)}

        def body(carry, _):
            state, obs, alive = carry
            if use_policy:
                action = jnp.tanh(agent.actor.apply(params, obs)[0])
            else:
                action = jnp.zeros((episodes, env.action_dim))
            state, obs, reward, done, info = env.step(state, action, batched)
            m = alive.astype(jnp.float32)
            return (state, obs,
                    jnp.logical_and(alive, jnp.logical_not(info["terminated"]))), (
                info["motor_cmd"], action, info["u_total"], m,
                jnp.where(alive, info["pos_err_desired"], cfg.term_pos_error))

        alive0 = jnp.ones(episodes, dtype=bool)
        (_, _, alive), out = jax.lax.scan(
            body, (state, obs, alive0), None, length=cfg.episode_steps)
        return out, draw, alive

    (motor, action, wrench, mask, pos_err), draw, alive = jax.jit(run)(key)
    return (np.asarray(motor), np.asarray(action), np.asarray(wrench),
            np.asarray(mask), np.asarray(pos_err),
            {k: np.asarray(v) for k, v in draw.items()}, np.asarray(alive))


def hf_fraction(sig, mask, dt, cutoff):
    """Share of variance above `cutoff` Hz, per episode, over live steps only."""
    T, E = sig.shape
    freqs = np.fft.rfftfreq(T, d=dt)
    band = freqs >= cutoff
    out = np.zeros(E)
    for e in range(E):
        n = max(int(mask[:, e].sum()), 8)
        x = sig[:n, e] - sig[:n, e].mean()
        if np.sum(x ** 2) < 1e-12:
            continue
        spec = np.abs(np.fft.rfft(x, n=T)) ** 2
        out[e] = spec[band].sum() / max(spec[1:].sum(), 1e-12)
    return out


def hf_amplitude(sig, mask, dt, cutoff):
    """RMS of the >cutoff Hz content, in the signal's own units.

    Use this, not hf_fraction, to judge severity. The fraction is scale-free,
    so a perfectly smooth episode whose total variance is tiny scores high
    simply because the little variance it has is numerical noise at high
    frequency. Measured episodes made this concrete: one with an HF fraction of
    0.64 varied by 0.01 N on a 4.7 N rotor command (invisible), while one at
    0.99 swung 0.5-5.5 N (violent). The fraction ranked them a factor of 1.5
    apart; the amplitude ranks them a factor of 10 apart.

    Fraction is still worth reporting -- it says WHERE the energy sits, which
    identifies a limit cycle -- but severity is an amplitude question.
    """
    T, E = sig.shape
    freqs = np.fft.rfftfreq(T, d=dt)
    band = freqs >= cutoff
    out = np.zeros(E)
    for e in range(E):
        n = max(int(mask[:, e].sum()), 8)
        x = sig[:n, e] - sig[:n, e].mean()
        X = np.fft.rfft(x, n=T)
        hp = np.fft.irfft(np.where(band, X, 0.0), n=T)[:n]
        out[e] = np.sqrt(np.mean(hp ** 2))
    return out


def plot_episode(e, motor, action, wrench, mask, pos_err, draw, alive,
                 hf, dt, out_path, zoom_s=1.0, base_motor=None, frac_e=0.0):
    n = max(int(mask[:, e].sum()), 8)
    t = np.arange(n) * dt
    nz = min(int(zoom_s / dt), n)
    z0 = max(0, n // 2 - nz // 2)
    tz = t[z0:z0 + nz]

    fig, axes = plt.subplots(2, 2, figsize=(15, 7.5))

    ax = axes[0, 0]
    for r in range(4):
        ax.plot(t, motor[:n, e, r], lw=0.7, color=ROTOR_COLORS[r],
                label=f"rotor {r}")
    ax.axhline(float(mixer.MAX_MOTOR_THRUST), color="k", ls="--", lw=1,
               label="motor limit")
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.axvspan(tz[0], tz[-1], color="grey", alpha=0.15)
    ax.set_ylabel("rotor thrust command [N]")
    ax.set_title("full episode (grey band = zoom window)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for r in range(4):
        ax.plot(tz, motor[z0:z0 + nz, e, r], lw=1.0, marker=".", ms=2.5,
                color=ROTOR_COLORS[r])
    ax.set_title(f"{zoom_s:g} s zoom — chatter is only visible at this scale")
    ax.set_ylabel("rotor thrust command [N]")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    labels = ["thrust", "Mx", "My", "Mz"]
    for c in range(4):
        ax.plot(tz, action[z0:z0 + nz, e, c], lw=1.0, marker=".", ms=2.5,
                label=labels[c])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("normalized residual action")
    ax.set_title("policy output (same zoom window)")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    x = motor[:n, e, 0] - motor[:n, e, 0].mean()
    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.abs(np.fft.rfft(x)) ** 2
    ax.semilogy(freqs[1:], spec[1:] + 1e-16, lw=0.8, color=ROTOR_COLORS[0],
                label="rotor 0, with residual")
    if base_motor is not None:
        xb = base_motor[:n, e, 0] - base_motor[:n, e, 0].mean()
        sb = np.abs(np.fft.rfft(xb)) ** 2
        ax.semilogy(freqs[1:], sb[1:] + 1e-16, lw=0.8, color="grey",
                    alpha=0.8, label="baseline, no residual")
    ax.axvline(0.5 / dt, color="k", ls="--", lw=1, label="Nyquist")
    ax.axvline(20.0, color="tab:purple", ls=":", lw=1,
               label="~motor bandwidth")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("power")
    ax.set_title("spectrum of rotor 0 command")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"episode {e}   HF {hf[e]:.3f} N ({frac_e:.2f} of variance)   "
        f"pos RMSE {np.sqrt(np.mean(pos_err[:n, e] ** 2)):.4f} m   "
        f"k={draw['k'][e]:.2f}  |F|={draw['force_amp'][e]:.2f} N  "
        f"motor scale {draw['motor'][e]:.2f}"
        f"{'   TERMINATED' if not alive[e] else ''}",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--base_controller", choices=("pd", "pid"), default=None)
    p.add_argument("--thrust_filter_beta", type=float, default=None,
                   help="defaults to the value the policy was trained with")
    p.add_argument("--moment_filter_beta", type=float, default=None)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--force_freq", type=float, default=0.0)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--hf_cutoff", type=float, default=10.0)
    p.add_argument("--zoom_s", type=float, default=1.0)
    p.add_argument("--which", default="worst,p75,median,best",
                   help="comma list of worst,p75,median,best or explicit "
                        "episode indices")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg, agent, params = load_policy(args.run_dir, args.checkpoint)
    controller = args.base_controller or cfg.env.base_controller
    env_cfg = EnvConfig(**{**cfg.env.__dict__})
    if args.thrust_filter_beta is not None:
        env_cfg.thrust_filter_beta = args.thrust_filter_beta
    if args.moment_filter_beta is not None:
        env_cfg.moment_filter_beta = args.moment_filter_beta

    out = Path(args.out) if args.out else (
        Path(args.run_dir) / "evaluation" /
        f"actuators_b{env_cfg.thrust_filter_beta:g}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    ranges = flat_ranges(env_cfg, force_freq=args.force_freq, force_dc_prob=0.0)
    key = jax.random.PRNGKey(args.eval_seed)

    motor, action, wrench, mask, pos_err, draw, alive = rollout(
        env_cfg, controller, agent, params, key, args.episodes, ranges)
    base_motor = rollout(env_cfg, controller, agent, params, key,
                         args.episodes, ranges, use_policy=False)[0]

    # Severity is measured in newtons; the fraction is kept for diagnosis.
    hf = hf_amplitude(motor[:, :, 0], mask, env_cfg.dt, args.hf_cutoff)
    hf_base = hf_amplitude(base_motor[:, :, 0], mask, env_cfg.dt, args.hf_cutoff)
    frac = hf_fraction(motor[:, :, 0], mask, env_cfg.dt, args.hf_cutoff)

    print(f"policy      : {args.run_dir}")
    print(f"base        : {controller}   thrust beta "
          f"{env_cfg.thrust_filter_beta}   force freq {args.force_freq} Hz")
    print(f"episodes    : {args.episodes}")
    print()
    print(f"rotor-0 command, RMS content above {args.hf_cutoff:g} Hz [N]:")
    print(f"  with residual : median {np.median(hf):.4f}  "
          f"p95 {np.percentile(hf, 95):.4f}  max {hf.max():.4f}")
    print(f"  baseline only : median {np.median(hf_base):.4f}  "
          f"p95 {np.percentile(hf_base, 95):.4f}  max {hf_base.max():.4f}")
    print(f"  mean rotor command is ~{motor[:, :, 0][mask > 0].mean():.2f} N, "
          "so judge these against that")
    print()
    for thr in (0.10, 0.25, 0.50):
        print(f"  episodes above {thr:.2f} N : "
              f"with residual {int((hf > thr).sum()):>3}   "
              f"baseline only {int((hf_base > thr).sum()):>3}")
    print()

    order = np.argsort(-hf)
    named = {"worst": int(order[0]),
             "p75": int(order[len(order) // 4]),
             "median": int(order[len(order) // 2]),
             "best": int(order[-1])}
    picks = []
    for tok in args.which.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in named:
            picks.append((tok, named[tok]))
        else:
            picks.append((f"ep{int(tok)}", int(tok)))

    for label, e in picks:
        path = out / f"{label}_episode_{e:03d}.png"
        plot_episode(e, motor, action, wrench, mask, pos_err, draw, alive,
                     hf, env_cfg.dt, path, zoom_s=args.zoom_s,
                     base_motor=base_motor, frac_e=frac[e])
        print(f"  {label:<7} episode {e:>3}  HF {hf[e]:.4f} N "
              f"(baseline {hf_base[e]:.4f} N)  -> {path.name}")

    write_manifest(out, {"run_dir": str(args.run_dir),
                         "base_controller": controller,
                         "thrust_filter_beta": env_cfg.thrust_filter_beta,
                         "episodes": args.episodes,
                         "force_freq": args.force_freq,
                         "hf_amp_median_with_residual_N": float(np.median(hf)),
                         "hf_amp_median_baseline_only_N": float(np.median(hf_base)),
                         "hf_amp_max_with_residual_N": float(hf.max()),
                         "n_episodes_above_0.5N": int((hf > 0.5).sum())})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
