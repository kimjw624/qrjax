"""What does the thrust low-pass filter at beta = 0.2 actually do?

The filter is

    u_t = (1 - beta) u_{t-1} + beta * a_t

applied to the THRUST channel of the residual only. This script answers four
separate questions that are easy to conflate:

  A  What does it remove?      Analytic transfer function, verified by
                               injecting sinusoids through the real env.
  B  What does it remove in
     practice?                 Requested vs applied residual for a trained
                               policy, in the time and frequency domains.
  C  What does it cost?        Tracking, saturation and termination with the
                               filter on and off, on matched episodes.
  D  What does it NOT fix?     The attitude-loop instability lives on the
                               moment channels; the filter touches only thrust.
                               If moment chatter and the high-gain limit cycle
                               are unchanged, the filter is not the fix for
                               that problem -- which matters, because it was
                               previously credited with solving it.

A caveat this script cannot remove on its own: a policy trained at beta = 0.2
and evaluated at beta = 1.0 is off its training distribution, so part of any
degradation is distribution shift rather than the filter's doing. Pass
``--run_dir_nofilter`` with a policy trained at beta = 1.0 and the script runs
the full 2x2 (each policy at each setting), which separates the two.

Example
-------
python -m scripts.analyze_lpf \\
    --run_dir experiments/stage1_.../seed_00/pid/train/trial_001 \\
    --episodes 128
"""

import argparse
import csv
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

from qrjax.envs import Config, EnvConfig, VecEnv, DisturbRanges   # noqa: E402
from qrjax.rl import SAC                                          # noqa: E402
from qrjax.rl.curriculum import flat_ranges                       # noqa: E402
from qrjax.utils import load_params, write_manifest               # noqa: E402

BETA = 0.2


def transfer_magnitude(freqs, beta, dt):
    """|H(f)| of u_t = (1-beta) u_{t-1} + beta a_t.

    H(z) = beta / (1 - (1-beta) z^-1), evaluated on the unit circle.
    """
    z = np.exp(-2j * np.pi * np.asarray(freqs) * dt)
    return np.abs(beta / (1.0 - (1.0 - beta) * z))


def cutoff_hz(beta, dt):
    """-3 dB point, found numerically rather than from a continuous-time
    approximation, which is inaccurate at beta this large."""
    f = np.linspace(1e-4, 0.5 / dt, 200000)
    mag = transfer_magnitude(f, beta, dt)
    return float(f[np.argmin(np.abs(mag - 1 / np.sqrt(2)))])


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


def hf_amplitude(sig, mask, dt, cutoff=10.0):
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


def rollout(env_cfg, controller, agent, params, key, episodes, ranges, beta):
    """Return requested action, applied action, motor commands and outcomes."""
    cfg = EnvConfig(**{**env_cfg.__dict__})
    cfg.base_controller = controller
    cfg.thrust_filter_beta = beta
    env = VecEnv(cfg, episodes)
    batched = env.broadcast_ranges(ranges)

    def run(key):
        k_reset, _ = jax.random.split(key)
        state, obs = env.reset(k_reset, batched, stagger=False)

        def body(carry, _):
            state, obs, alive = carry
            requested = jnp.tanh(agent.actor.apply(params, obs)[0])
            state, obs, reward, done, info = env.step(state, requested, batched)
            m = alive.astype(jnp.float32)
            return (state, obs,
                    jnp.logical_and(alive,
                                    jnp.logical_not(info["terminated"]))), (
                requested, state.prev_action_norm, info["motor_cmd"], m,
                jnp.where(alive, info["pos_err_desired"], cfg.term_pos_error),
                info["saturated"] * m)

        (_, _, alive), out = jax.lax.scan(
            body, (state, obs, jnp.ones(episodes, bool)), None,
            length=cfg.episode_steps)
        return out, alive

    (req, app, motor, mask, pos_err, sat), alive = jax.jit(run)(key)
    return (np.asarray(req), np.asarray(app), np.asarray(motor),
            np.asarray(mask), np.asarray(pos_err), np.asarray(sat),
            np.asarray(alive))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True,
                   help="policy trained WITH the filter (beta=0.2)")
    p.add_argument("--run_dir_nofilter", default=None,
                   help="policy trained WITHOUT the filter (beta=1.0). Supply "
                        "this to separate the filter's effect from "
                        "distribution shift")
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--force_freq", type=float, default=0.0)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg, agent, params = load_policy(args.run_dir, args.checkpoint)
    dt = cfg.env.dt
    controller = cfg.env.base_controller
    out = Path(args.out) if args.out else Path("experiments") / (
        f"lpf_beta{BETA:g}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    (out / "plots").mkdir(parents=True, exist_ok=True)

    policies = {"trained@0.2": (agent, params)}
    if args.run_dir_nofilter:
        _, a2, p2 = load_policy(args.run_dir_nofilter, args.checkpoint)
        policies["trained@1.0"] = (a2, p2)

    # ---------------- A: what does the filter remove? ----------------------
    fc = cutoff_hz(BETA, dt)
    print(f"\nA  filter response, beta = {BETA}, dt = {dt} s")
    print(f"   u_t = {1-BETA:g}*u_(t-1) + {BETA:g}*a_t")
    print(f"   -3 dB cutoff : {fc:.2f} Hz")
    for f in (0.5, 1, 2, 5, 10, 25, 50):
        mag = float(transfer_magnitude([f], BETA, dt)[0])
        print(f"   {f:>5.1f} Hz : gain {mag:.4f}  ({20*np.log10(mag):+6.1f} dB)")

    # empirical check: drive the env with a sinusoid, measure what is applied
    print("\n   empirical check (sinusoid injected through the real env):")
    ecfg = EnvConfig(**{**cfg.env.__dict__})
    ecfg.thrust_filter_beta = BETA
    env1 = VecEnv(ecfg, 1)
    r1 = env1.broadcast_ranges(flat_ranges(ecfg, force_freq=0.0, force_dc_prob=0.0))
    step1 = jax.jit(env1.step)
    emp = []
    for f in (1, 5, 10, 25, 50):
        state, obs = env1.reset(jax.random.PRNGKey(0), r1, stagger=False)
        applied, requested = [], []
        for k in range(400):
            # Cosine, not sine: at exactly Nyquist sin(2*pi*f*k*dt) = sin(pi*k)
            # is identically zero for integer k, so a sine probe would report a
            # gain of 0 there regardless of the filter. cos gives (-1)^k, the
            # genuine Nyquist input.
            a = jnp.array([[0.5 * np.cos(2 * np.pi * f * k * dt), 0.0, 0.0, 0.0]])
            state, obs, *_ = step1(state, a, r1)
            applied.append(float(state.prev_action_norm[0, 0]))
            requested.append(float(a[0, 0]))
        # Normalise by the INPUT's own std, not by the sinusoid RMS A/sqrt(2).
        # At Nyquist cos(pi*k) = (-1)^k has std A, not A/sqrt(2), so assuming
        # the sinusoid form overstates the gain there by sqrt(2).
        meas = np.std(applied[100:]) / max(np.std(requested[100:]), 1e-12)
        pred = float(transfer_magnitude([f], BETA, dt)[0])
        emp.append((f, pred, meas))
        print(f"   {f:>5.1f} Hz : predicted {pred:.4f}   measured {meas:.4f}")

    # ---------------- B/C/D: with a real policy ----------------------------
    ranges = flat_ranges(cfg.env, force_freq=args.force_freq, force_dc_prob=0.0)
    key = jax.random.PRNGKey(args.eval_seed)
    rows, data = [], {}

    print(f"\nB/C/D  {args.episodes} matched episodes, {controller.upper()} base")
    header = (f"   {'policy':<14}{'eval beta':>10}{'thrust HF':>11}"
              f"{'moment HF':>11}{'motor HF N':>12}{'med RMSE':>10}"
              f"{'sat %':>8}{'term %':>8}")
    print(header)
    for pname, (ag, pr) in policies.items():
        for beta in (1.0, BETA):
            req, app, motor, mask, pe, sat, alive = rollout(
                cfg.env, controller, ag, pr, key, args.episodes, ranges, beta)
            hf_thr = hf_amplitude(app[:, :, 0], mask, dt)
            hf_mom = np.mean([hf_amplitude(app[:, :, c], mask, dt)
                              for c in (1, 2, 3)], axis=0)
            hf_mot = hf_amplitude(motor[:, :, 0], mask, dt)
            med = np.median(np.sqrt(np.mean(pe ** 2, axis=0)))
            row = {"policy": pname, "eval_beta": beta,
                   "thrust_hf": float(hf_thr.mean()),
                   "moment_hf": float(hf_mom.mean()),
                   "motor_hf_N": float(hf_mot.mean()),
                   "median_rmse": float(med),
                   "saturation_pct": float(100 * sat.mean()),
                   "terminated_pct": float(100 * (1 - alive.mean()))}
            rows.append(row)
            data[(pname, beta)] = (req, app, motor, mask, pe)
            print(f"   {pname:<14}{beta:>10}{row['thrust_hf']:>11.4f}"
                  f"{row['moment_hf']:>11.4f}{row['motor_hf_N']:>12.4f}"
                  f"{row['median_rmse']:>10.4f}{row['saturation_pct']:>8.2f}"
                  f"{row['terminated_pct']:>8.1f}")

    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------------- figures ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    fgrid = np.linspace(0.01, 0.5 / dt, 2000)
    ax.semilogx(fgrid, 20 * np.log10(transfer_magnitude(fgrid, BETA, dt)),
                lw=2, label=f"analytic, beta = {BETA}")
    ax.scatter([e[0] for e in emp], [20 * np.log10(e[2]) for e in emp],
               color="tab:red", zorder=5, label="measured through the env")
    ax.axhline(-3, color="grey", ls=":", lw=1)
    ax.axvline(fc, color="grey", ls=":", lw=1)
    ax.text(fc, -40, f" -3 dB at {fc:.1f} Hz", fontsize=9)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("gain [dB]")
    ax.set_title("A: what the thrust filter removes\n"
                 f"u_t = {1-BETA:g}·u_(t-1) + {BETA:g}·a_t at {1/dt:.0f} Hz")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "plots" / "A_filter_response.png", dpi=150)
    plt.close(fig)

    req, app, motor, mask, pe = data[("trained@0.2", BETA)]
    e = int(np.argmax(hf_amplitude(req[:, :, 0], mask, dt)))
    n = max(int(mask[:, e].sum()), 8)
    z0, nz = max(0, n // 2 - 50), 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    t = np.arange(z0, z0 + nz) * dt
    axes[0].plot(t, req[z0:z0 + nz, e, 0], lw=1.1, marker=".", ms=3,
                 label="requested by policy")
    axes[0].plot(t, app[z0:z0 + nz, e, 0], lw=1.6, label="applied after filter")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("normalized thrust residual")
    axes[0].set_title(f"B: requested vs applied (episode {e}, 1 s)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    freqs = np.fft.rfftfreq(n, d=dt)
    for sig, lab in ((req[:n, e, 0], "requested"), (app[:n, e, 0], "applied")):
        s = np.abs(np.fft.rfft(sig - sig.mean())) ** 2
        axes[1].semilogy(freqs[1:], s[1:] + 1e-16, lw=0.9, label=lab)
    axes[1].axvline(fc, color="grey", ls=":", lw=1)
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("power")
    axes[1].set_title("B: spectrum of the thrust residual")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plots" / "B_requested_vs_applied.png", dpi=150)
    plt.close(fig)

    labels = [f"{r['policy']}\neval @ {r['eval_beta']:g}" for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, field, title in (
            (axes[0], "thrust_hf", "thrust residual HF\n(the filter's target)"),
            (axes[1], "moment_hf", "moment residual HF\n(untouched by the filter)"),
            (axes[2], "median_rmse", "median tracking RMSE [m]\n(the cost)"),
            (axes[3], "terminated_pct", "termination [%]")):
        vals = [r[field] for r in rows]
        colors = ["#bbbbbb" if r["eval_beta"] == 1.0 else "#1f77b4" for r in rows]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("C/D: what the filter buys, and what it leaves alone "
                 "(grey = no filter, blue = beta 0.2)")
    fig.tight_layout()
    fig.savefig(out / "plots" / "CD_effect_summary.png", dpi=150)
    plt.close(fig)

    # ---------------- report -------------------------------------------------
    def get(pname, beta):
        return next(r for r in rows if r["policy"] == pname
                    and r["eval_beta"] == beta)

    off, on = get("trained@0.2", 1.0), get("trained@0.2", BETA)
    L = [f"# Thrust low-pass filter at beta = {BETA}\n",
         f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
         f"- Policy: `{args.run_dir}` ({controller.upper()} base)",
         f"- {args.episodes} matched episodes, eval seed {args.eval_seed}",
         "",
         f"`u_t = {1-BETA:g}·u_(t-1) + {BETA:g}·a_t` on the THRUST channel only.",
         "",
         "## A — what it removes", "",
         f"-3 dB cutoff **{fc:.2f} Hz** (numerically; a continuous-time "
         "approximation is inaccurate at this beta).", "",
         "| frequency | gain | dB | measured |", "|---|---|---|---|"]
    for f, pred, meas in emp:
        L.append(f"| {f:g} Hz | {pred:.4f} | {20*np.log10(pred):+.1f} "
                 f"| {meas:.4f} |")

    L += ["", "## C — what it buys and costs", "",
          "| policy | eval beta | thrust HF | moment HF | motor HF [N] | "
          "median RMSE | sat % | term % |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['policy']} | {r['eval_beta']:g} | {r['thrust_hf']:.4f} "
                 f"| {r['moment_hf']:.4f} | {r['motor_hf_N']:.4f} "
                 f"| {r['median_rmse']:.4f} | {r['saturation_pct']:.2f} "
                 f"| {r['terminated_pct']:.1f} |")

    thr_red = off["thrust_hf"] / max(on["thrust_hf"], 1e-9)
    mom_red = off["moment_hf"] / max(on["moment_hf"], 1e-9)
    L += ["", "## D — what it does NOT fix", "",
          f"- Thrust residual HF falls **{thr_red:.1f}x** with the filter on.",
          f"- Moment residual HF changes by **{mom_red:.2f}x** — essentially "
          "nothing, as expected: the filter never touches those channels.",
          "",
          "That matters, because the severe actuator oscillation is an "
          "ATTITUDE-loop instability driven by the effective gain "
          "`g = kf·arm/k` exceeding the discrete stability limit. It lives on "
          "the moment channels. The thrust filter cannot and does not address "
          "it — it suppresses thrust-channel policy chatter, which is a real "
          "but separate problem.",
          "",
          f"- Median tracking RMSE: {off['median_rmse']:.4f} without the "
          f"filter, {on['median_rmse']:.4f} with. "
          f"{'The filter costs nothing measurable.' if on['median_rmse'] <= off['median_rmse']*1.05 else 'The filter costs some tracking.'}",
          ""]
    if len(policies) == 1:
        L += ["> **Caveat.** Only one policy was supplied, trained at "
              f"beta = {BETA}. Evaluating it at beta = 1.0 puts it off its "
              "training distribution, so part of the difference is "
              "distribution shift rather than the filter itself. Pass "
              "`--run_dir_nofilter` with a policy trained at beta = 1.0 for "
              "the full 2x2.\n"]
    else:
        a = get("trained@1.0", 1.0)
        b = get("trained@0.2", BETA)
        L += ["Both policies were trained at their own beta, so this row "
              "compares like with like:", "",
              f"- trained and evaluated at 1.0: thrust HF {a['thrust_hf']:.4f}, "
              f"RMSE {a['median_rmse']:.4f}",
              f"- trained and evaluated at {BETA}: thrust HF "
              f"{b['thrust_hf']:.4f}, RMSE {b['median_rmse']:.4f}", ""]

    L += ["## Files", "",
          "- `results.csv`",
          "- `plots/A_filter_response.png` — analytic response with measured points",
          "- `plots/B_requested_vs_applied.png` — time and frequency domain",
          "- `plots/CD_effect_summary.png` — benefit, cost, and what is untouched"]
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    write_manifest(out, {"beta": BETA, "run_dir": args.run_dir,
                         "run_dir_nofilter": args.run_dir_nofilter,
                         "episodes": args.episodes, "cutoff_hz": fc})
    print(f"\nwrote {out}/REPORT.md")


if __name__ == "__main__":
    main()
