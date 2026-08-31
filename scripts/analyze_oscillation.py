"""Stage 3, steps 1-2: detect residual oscillation and characterise when it happens.

This does NOT try to explain the oscillation. It answers the two questions that
have to come first, because the design of any causal test depends on them:

  1. Is there a distinct oscillating MODE, or a severity gradient?
     If episodes cluster into "oscillating" and "clean", a classifier is
     meaningful and switch-on/switch-off experiments make sense. If the metric
     is unimodal, "oscillating episodes" is the wrong frame and step 3 should
     be designed around a continuous response instead.

  2. What distinguishes the affected episodes?
     Each episode has a known disturbance draw (mass ratio, force amplitude and
     frequency, per-rotor actuator and geometry scales). If oscillation
     concentrates in a region of that space, the mechanism is probably specific;
     if it is spread uniformly, it is probably a property of the policy itself.

Metrics, per episode and per action channel:

  hf_fraction   share of action variance above --hf_cutoff Hz. Primary metric:
                a limit cycle at the control rate puts nearly all its energy at
                high frequency, while legitimate control activity is slow.
  lag1_autocorr lag-1 autocorrelation. Strongly negative means the command
                flips sign every step -- the signature of a Nyquist-frequency
                limit cycle specifically, as opposed to merely fast activity.
  tv_per_step   mean |a_t - a_{t-1}|, an amplitude-sensitive roughness measure.

Run the same policy with the filter on and off to see the effect directly:

    python -m scripts.analyze_oscillation \\
        --run_dir experiments/stage1_.../seed_00/pid/train/trial_001 \\
        --base_controller pid --thrust_filter_beta 1.0,0.2 --episodes 256
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
import jax                                              # noqa: E402
import jax.numpy as jnp                                 # noqa: E402
import numpy as np                                      # noqa: E402

from qrjax.envs import Config, EnvConfig, VecEnv        # noqa: E402
from qrjax.rl import SAC                                # noqa: E402
from qrjax.rl.curriculum import flat_ranges             # noqa: E402
from qrjax.utils import load_params, write_json, write_manifest   # noqa: E402

CHANNELS = ["thrust", "Mx", "My", "Mz"]


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


def rollout_actions(env_cfg, controller, agent, params, key, episodes, ranges):
    """Full action traces plus the per-episode disturbance draw.

    Unlike the evaluation rollout this keeps the raw (T, E, 4) action array,
    because every oscillation metric is spectral and cannot be computed from
    summary statistics.
    """
    cfg = EnvConfig(**{**env_cfg.__dict__})
    cfg.base_controller = controller
    env = VecEnv(cfg, episodes)
    batched = env.broadcast_ranges(ranges)

    def run(key):
        k_reset, _ = jax.random.split(key)
        state, obs = env.reset(k_reset, batched, stagger=False)
        draw = {
            "k": state.k,
            "force_amp": jnp.linalg.norm(state.external_force, axis=-1),
            "force_freq": state.force_freq,
            "motor": state.mixer_true.kf_scale.mean(axis=-1),
            "moment": state.mixer_true.moment_scale.mean(axis=-1),
            "arm": state.mixer_true.arm_scale.mean(axis=-1),
        }

        def body(carry, _):
            state, obs, alive = carry
            mean, _ = agent.actor.apply(params, obs)
            action = jnp.tanh(mean)
            state, obs, reward, done, info = env.step(state, action, batched)
            m = alive.astype(jnp.float32)
            alive_next = jnp.logical_and(alive,
                                         jnp.logical_not(info["terminated"]))
            return (state, obs, alive_next), (
                action, m,
                jnp.where(alive, info["pos_err_desired"], cfg.term_pos_error),
                info["saturated"] * m,
            )

        alive0 = jnp.ones(episodes, dtype=bool)
        (_, _, alive), out = jax.lax.scan(
            body, (state, obs, alive0), None, length=cfg.episode_steps)
        return out, draw, alive

    (actions, alive_mask, pos_err, sat), draw, alive = jax.jit(run)(key)
    return (np.asarray(actions), np.asarray(alive_mask), np.asarray(pos_err),
            np.asarray(sat), {k: np.asarray(v) for k, v in draw.items()},
            np.asarray(alive))


def oscillation_metrics(actions, alive_mask, dt, hf_cutoff):
    """Per-episode, per-channel oscillation metrics.

    Only the live portion of each episode is used: a terminated episode's
    trailing frozen steps would otherwise read as perfectly smooth and dilute
    the metric of exactly the episodes most likely to have oscillated.
    """
    T, E, C = actions.shape
    hf = np.zeros((E, C))
    ac1 = np.zeros((E, C))
    tv = np.zeros((E, C))

    freqs = np.fft.rfftfreq(T, d=dt)
    hf_band = freqs >= hf_cutoff

    for e in range(E):
        n = max(int(alive_mask[:, e].sum()), 8)
        for c in range(C):
            x = actions[:n, e, c]
            x = x - x.mean()
            var = np.sum(x ** 2)
            if var < 1e-12:
                continue
            spec = np.abs(np.fft.rfft(x, n=T)) ** 2
            hf[e, c] = spec[hf_band].sum() / max(spec[1:].sum(), 1e-12)
            d = np.diff(x)
            ac1[e, c] = np.sum(x[:-1] * x[1:]) / var
            tv[e, c] = np.mean(np.abs(d))
    return {"hf_fraction": hf, "lag1_autocorr": ac1, "tv_per_step": tv}


def hf_amplitude(sig, mask, dt, cutoff):
    """RMS of the >cutoff Hz content in the signal's own units.

    Severity must be measured in physical units. hf_fraction is scale-free, so
    a smooth episode with tiny total variance scores high simply because what
    little variance it has is numerical noise at high frequency. Both are
    reported: amplitude answers "how bad", fraction answers "is it a limit
    cycle".
    """
    T, E = sig.shape
    freqs = np.fft.rfftfreq(T, d=dt)
    band = freqs >= cutoff
    out = np.zeros(E)
    for e in range(E):
        n = max(int(mask[:, e].sum()), 8)
        x = sig[:n, e] - sig[:n, e].mean()
        X = np.fft.rfft(x, n=T)
        out[e] = np.sqrt(np.mean(
            np.fft.irfft(np.where(band, X, 0.0), n=T)[:n] ** 2))
    return out


def bimodality_coefficient(x):
    """Sarle's bimodality coefficient. > 5/9 suggests a bimodal distribution.

    Reported as a hint, not a verdict: it is only a moment-based heuristic and
    a long right tail can push it over the threshold without any real cluster
    structure. Read it next to the histogram.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4 or x.std() < 1e-12:
        return float("nan")
    m3 = ((x - x.mean()) ** 3).mean() / x.std() ** 3
    m4 = ((x - x.mean()) ** 4).mean() / x.std() ** 4 - 3.0
    return float((m3 ** 2 + 1.0) /
                 (m4 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--base_controller", choices=("pd", "pid"), default=None,
                   help="defaults to the value the policy was trained with")
    p.add_argument("--thrust_filter_beta", default="1.0,0.2",
                   type=lambda s: [float(x) for x in s.split(",") if x.strip()],
                   help="filter settings to compare. 1.0 disables filtering")
    p.add_argument("--moment_filter_beta", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--force_freq", type=float, default=0.0)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--hf_cutoff", type=float, default=10.0,
                   help="Hz; action variance above this counts as oscillation")
    p.add_argument("--n_worst", type=int, default=3,
                   help="how many worst episodes to plot as time series")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg, agent, params = load_policy(args.run_dir, args.checkpoint)
    controller = args.base_controller or cfg.env.base_controller
    out = Path(args.out) if args.out else (
        Path(args.run_dir) / "evaluation" /
        f"oscillation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    (out / "plots").mkdir(parents=True, exist_ok=True)

    print(f"policy       : {args.run_dir}")
    print(f"base         : {controller}")
    print(f"episodes     : {args.episodes}, force freq {args.force_freq} Hz")
    print(f"hf cutoff    : {args.hf_cutoff} Hz  (control rate "
          f"{1/cfg.env.dt:.0f} Hz, Nyquist {0.5/cfg.env.dt:.0f} Hz)")
    print()

    key = jax.random.PRNGKey(args.eval_seed)
    rows, per_beta = [], {}

    for beta in args.thrust_filter_beta:
        env_cfg = EnvConfig(**{**cfg.env.__dict__})
        env_cfg.thrust_filter_beta = beta
        env_cfg.moment_filter_beta = args.moment_filter_beta
        ranges = flat_ranges(env_cfg, force_freq=args.force_freq,
                             force_dc_prob=0.0)

        actions, alive_mask, pos_err, sat, draw, alive = rollout_actions(
            env_cfg, controller, agent, params, key, args.episodes, ranges)
        m = oscillation_metrics(actions, alive_mask, env_cfg.dt, args.hf_cutoff)
        per_beta[beta] = (m, draw, actions, alive_mask, pos_err, sat, alive)

        hf_thrust = m["hf_fraction"][:, 0]
        print(f"  thrust_filter_beta = {beta:<5}  "
              f"HF fraction (thrust): mean {hf_thrust.mean():.3f}  "
              f"median {np.median(hf_thrust):.3f}  max {hf_thrust.max():.3f}   "
              f"lag1 AC {m['lag1_autocorr'][:, 0].mean():+.3f}   "
              f"term {100*(1-alive.mean()):.1f}%")

        for e in range(args.episodes):
            row = {"beta": beta, "episode": e,
                   "terminated": float(not alive[e]),
                   "pos_rmse": float(np.sqrt(np.mean(pos_err[:, e] ** 2))),
                   "saturation_frac": float(sat[:, e].mean()),
                   **{k: float(v[e]) for k, v in draw.items()}}
            for ci, ch in enumerate(CHANNELS):
                for name, arr in m.items():
                    row[f"{name}_{ch}"] = float(arr[e, ci])
            rows.append(row)

    with (out / "per_episode.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------------- Q1: is there a distinct oscillating mode? -------------
    fig, axes = plt.subplots(1, len(args.thrust_filter_beta),
                             figsize=(6 * len(args.thrust_filter_beta), 4.2),
                             squeeze=False)
    bimodal = {}
    for ax, beta in zip(axes[0], args.thrust_filter_beta):
        m = per_beta[beta][0]
        hf = m["hf_fraction"][:, 0]
        bc = bimodality_coefficient(hf)
        bimodal[beta] = bc
        ax.hist(hf, bins=40, color="tab:red", alpha=0.75, label="thrust")
        ax.hist(m["hf_fraction"][:, 1:].mean(axis=1), bins=40,
                color="tab:blue", alpha=0.5, label="moments (mean)")
        ax.set_title(f"beta = {beta}   Sarle BC = {bc:.3f}"
                     f"{'  (bimodal hint)' if bc > 5/9 else ''}")
        ax.set_xlabel(f"fraction of action variance above {args.hf_cutoff:g} Hz")
        ax.set_ylabel("episodes")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Q1: distinct oscillating mode, or a severity gradient?")
    fig.tight_layout()
    fig.savefig(out / "plots" / "hf_distribution.png", dpi=150)
    plt.close(fig)

    # ---------------- Q2: what distinguishes affected episodes? ------------
    drivers = ["k", "force_amp", "motor", "moment", "arm",
               "saturation_frac", "pos_rmse"]
    corr_rows = []
    beta_worst = max(args.thrust_filter_beta)      # least filtered
    m, draw, actions, alive_mask, pos_err, sat, alive = per_beta[beta_worst]
    hf = m["hf_fraction"][:, 0]
    sel = [r for r in rows if r["beta"] == beta_worst]

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for ax, d in zip(axes.flat, drivers):
        x = np.array([r[d] for r in sel])
        if x.std() < 1e-12:
            ax.set_title(f"{d} (constant)")
            continue
        c = float(np.corrcoef(x, hf)[0, 1])
        corr_rows.append({"driver": d, "pearson_r": c})
        ax.scatter(x, hf, s=10, alpha=0.55)
        ax.set_xlabel(d)
        ax.set_ylabel("thrust HF fraction")
        ax.set_title(f"r = {c:+.3f}")
        ax.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    fig.suptitle(f"Q2: what predicts thrust oscillation?  "
                 f"(beta = {beta_worst}, {args.episodes} episodes)")
    fig.tight_layout()
    fig.savefig(out / "plots" / "drivers_scatter.png", dpi=150)
    plt.close(fig)

    if corr_rows:
        with (out / "correlations.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["driver", "pearson_r"])
            w.writeheader()
            w.writerows(sorted(corr_rows, key=lambda r: -abs(r["pearson_r"])))

    # ---------------- worst-episode traces ---------------------------------
    worst = np.argsort(-hf)[:args.n_worst]
    fig, axes = plt.subplots(len(worst), 1, figsize=(11, 3.0 * len(worst)),
                             squeeze=False)
    for ax, e in zip(axes[:, 0], worst):
        n = int(alive_mask[:, e].sum())
        t = np.arange(n) * cfg.env.dt
        ax.plot(t, actions[:n, e, 0], lw=0.8, color="tab:red", label="thrust")
        ax.plot(t, actions[:n, e, 1], lw=0.8, color="tab:blue", alpha=0.7,
                label="Mx")
        ax.set_title(f"episode {e}: HF {hf[e]:.3f}, lag1 AC "
                     f"{m['lag1_autocorr'][e, 0]:+.3f}, k={draw['k'][e]:.2f}, "
                     f"|F|={draw['force_amp'][e]:.2f} N, "
                     f"sat {100*sat[:, e].mean():.1f}%"
                     f"{', TERMINATED' if not alive[e] else ''}")
        ax.set_ylabel("normalized action")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time [s]")
    fig.suptitle(f"Worst {len(worst)} episodes by thrust HF fraction "
                 f"(beta = {beta_worst})")
    fig.tight_layout()
    fig.savefig(out / "plots" / "worst_episodes.png", dpi=150)
    plt.close(fig)

    # ---------------- report ------------------------------------------------
    L = ["# Oscillation: detection and characterisation\n",
         f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
         f"- Policy: `{args.run_dir}` on a **{controller}** base",
         f"- {args.episodes} episodes, force frequency {args.force_freq} Hz, "
         f"eval seed {args.eval_seed}",
         f"- HF cutoff {args.hf_cutoff:g} Hz (control rate "
         f"{1/cfg.env.dt:.0f} Hz, Nyquist {0.5/cfg.env.dt:.0f} Hz)",
         "",
         "## Q1 — is there a distinct oscillating mode?", "",
         "| beta | HF mean | HF median | HF p95 | HF max | lag1 AC (thrust) | "
         "lag1 AC (moments) | Sarle BC | term % |",
         "|---|---|---|---|---|---|---|---|---|"]
    for beta in args.thrust_filter_beta:
        mm, _, _, _, _, _, al = per_beta[beta]
        h = mm["hf_fraction"][:, 0]
        L.append(
            f"| {beta} | {h.mean():.3f} | {np.median(h):.3f} "
            f"| {np.percentile(h, 95):.3f} | {h.max():.3f} "
            f"| {mm['lag1_autocorr'][:, 0].mean():+.3f} "
            f"| {mm['lag1_autocorr'][:, 1:].mean():+.3f} "
            f"| {bimodal[beta]:.3f} | {100*(1-al.mean()):.1f} |")
    L += ["",
          "Sarle's bimodality coefficient exceeds 5/9 ≈ 0.556 for a bimodal "
          "distribution, but it is a moment-based heuristic and a long right "
          "tail alone can push it over. Read it beside "
          "`plots/hf_distribution.png` rather than on its own.",
          "",
          "A strongly **negative** lag-1 autocorrelation means the command "
          "flips sign every step. That is specifically a limit cycle at the "
          "control rate, as opposed to merely fast activity.",
          "",
          "## Q2 — what predicts oscillation?", "",
          f"Pearson correlation with thrust HF fraction at beta = {beta_worst}:",
          "", "| driver | r |", "|---|---|"]
    for r in sorted(corr_rows, key=lambda r: -abs(r["pearson_r"])):
        L.append(f"| {r['driver']} | {r['pearson_r']:+.3f} |")
    L += ["",
          "Correlation here is not causation in the usual way: `pos_rmse` and "
          "`saturation_frac` are OUTCOMES of the episode, so a correlation "
          "with them says oscillation co-occurs with poor tracking, not that "
          "either caused the other. Only `k`, `force_amp`, `motor`, `moment` "
          "and `arm` are fixed before the episode starts and can be treated as "
          "candidate causes.",
          "",
          "## Files", "",
          "- `per_episode.csv` — every episode, every metric, every channel",
          "- `correlations.csv`",
          "- `plots/hf_distribution.png` — Q1",
          "- `plots/drivers_scatter.png` — Q2",
          "- `plots/worst_episodes.png` — action traces of the worst episodes"]
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    write_manifest(out, {"stage": "3-detect", "run_dir": str(args.run_dir),
                         "base_controller": controller,
                         "betas": args.thrust_filter_beta,
                         "episodes": args.episodes,
                         "force_freq": args.force_freq,
                         "hf_cutoff": args.hf_cutoff})
    print(f"\nwrote {out}/REPORT.md")


if __name__ == "__main__":
    main()
