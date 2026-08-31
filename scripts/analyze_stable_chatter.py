"""Where does actuator chatter come from in the GAIN-STABLE region?

Episodes with g = kf·arm/k below the discrete stability limit are not affected
by the attitude-loop gain-margin failure, so any chatter there has a different
source. This script asks whether that source is the residual policy.

Three arms, all on the same episodes:

    baseline        no residual at all -- the floor
    residual, no LPF
    residual, LPF

and, if a policy trained without the filter is supplied, both policies at both
settings. That matters: a policy trained at beta = 0.2 and evaluated at
beta = 1.0 is off its training distribution, so some of its chatter is the
policy misbehaving under conditions it never saw rather than the filter's
absence. Only a matched pair separates the two.

The central measurement is an ATTRIBUTION. The wrench reaching the mixer is

    u_total = u_base + u_residual

and the env logs both parts, so the high-frequency content of each can be
measured separately:

    if HF(u_residual) accounts for HF(u_total)
        the residual injects chatter directly
    if HF(u_base) is also raised above the no-residual run
        the BASELINE is reacting to a state the residual perturbed, i.e. a
        closed-loop interaction rather than direct injection

Those two have different fixes -- filtering the command helps the first, while
the second needs the loop itself changed -- so distinguishing them is the point.

A dose-response on residual authority is included as a cross-check: direct
injection should scale roughly linearly with authority, a resonant interaction
should not.

Example
-------
python -m scripts.analyze_stable_chatter \\
    --run_dir experiments/stage1_.../seed_00/pd/train/trial_001 \\
    --run_dir_nofilter experiments/stage3_nofilter/seed_00/pd/train/trial_001
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

from qrjax.core.mixer import WRENCH_MAX              # noqa: E402
from qrjax.envs import Config, EnvConfig, VecEnv     # noqa: E402
from qrjax.rl import SAC                             # noqa: E402
from qrjax.rl.curriculum import flat_ranges          # noqa: E402
from qrjax.utils import load_params, write_manifest  # noqa: E402

J_NOM, KR, KOMEGA = 0.022, 8.81, 2.54
CHANNELS = ["thrust", "Mx", "My", "Mz"]


def critical_gain(dt):
    """2J / (dt (kOmega - dt kR/2)); see verify_gain_margin for the derivation."""
    return 2 * J_NOM / (dt * (KOMEGA - dt * KR / 2))


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
    beta = cfg.env.thrust_filter_beta
    return cfg, agent, params, beta


def hp_component(sig, mask, dt, cutoff):
    """High-pass component and its RMS, per episode, over live steps only."""
    T, E = sig.shape
    freqs = np.fft.rfftfreq(T, d=dt)
    band = freqs >= cutoff
    rms = np.zeros(E)
    for e in range(E):
        n = max(int(mask[:, e].sum()), 8)
        x = sig[:n, e] - sig[:n, e].mean()
        X = np.fft.rfft(x, n=T)
        rms[e] = np.sqrt(np.mean(
            np.fft.irfft(np.where(band, X, 0.0), n=T)[:n] ** 2))
    return rms


def mean_psd(sig, mask, dt, nfft=1024):
    """Episode-averaged power spectrum, using each episode's live portion."""
    acc, count = None, 0
    for e in range(sig.shape[1]):
        n = max(int(mask[:, e].sum()), 16)
        x = sig[:n, e] - sig[:n, e].mean()
        if n < nfft:
            x = np.pad(x, (0, nfft - n))
        else:
            x = x[:nfft]
        p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        acc = p if acc is None else acc + p
        count += 1
    return np.fft.rfftfreq(nfft, d=dt), acc / max(count, 1)


def rollout(env_cfg, controller, agent, params, key, episodes, ranges,
            beta, authority=None):
    cfg = EnvConfig(**{**env_cfg.__dict__})
    cfg.base_controller = controller
    cfg.thrust_filter_beta = beta
    if authority is not None:
        cfg.residual_authority = authority
    env = VecEnv(cfg, episodes)
    batched = env.broadcast_ranges(ranges)
    scale = np.asarray(cfg.residual_authority * WRENCH_MAX)

    def run(key):
        k_reset, _ = jax.random.split(key)
        state, obs = env.reset(k_reset, batched, stagger=False)
        g = (state.mixer_true.kf_scale[:, 0] * state.mixer_true.arm_scale[:, 0]
             / state.k)

        def body(carry, _):
            state, obs, alive = carry
            if agent is None:
                action = jnp.zeros((episodes, env.action_dim))
            else:
                action = jnp.tanh(agent.actor.apply(params, obs)[0])
            state, obs, reward, done, info = env.step(state, action, batched)
            m = alive.astype(jnp.float32)
            return (state, obs,
                    jnp.logical_and(alive,
                                    jnp.logical_not(info["terminated"]))), (
                info["motor_cmd"], info["u_base"], info["u_total"],
                state.prev_action_norm, m,
                jnp.where(alive, info["pos_err_desired"], cfg.term_pos_error),
                info["saturated"] * m)

        (_, _, alive), out = jax.lax.scan(
            body, (state, obs, jnp.ones(episodes, bool)), None,
            length=cfg.episode_steps)
        return out, g, alive

    (motor, u_base, u_total, applied, mask, pe, sat), g, alive = jax.jit(run)(key)
    motor, u_base = np.asarray(motor), np.asarray(u_base)
    u_total, applied = np.asarray(u_total), np.asarray(applied)
    # Applied residual wrench in physical units, reconstructed from the
    # normalized action actually applied (post-filter).
    u_res = applied * scale[None, None, :]
    return {"motor": motor, "u_base": u_base, "u_total": u_total,
            "u_res": u_res, "mask": np.asarray(mask), "pos_err": np.asarray(pe),
            "sat": np.asarray(sat), "g": np.asarray(g),
            "alive": np.asarray(alive)}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True,
                   help="policy trained WITH the filter")
    p.add_argument("--run_dir_nofilter", default=None,
                   help="policy trained WITHOUT the filter (beta=1.0). Without "
                        "this the no-filter rows are off-distribution")
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--episodes", type=int, default=192)
    p.add_argument("--hf_cutoff", type=float, default=10.0)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--authority_sweep", default="1.0,0.5,0.25,0.1",
                   type=lambda s: [float(x) for x in s.split(",") if x.strip()],
                   help="residual authority multipliers, as fractions of the "
                        "trained value")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg, agent, params, beta_trained = load_policy(args.run_dir, args.checkpoint)
    dt, controller = cfg.env.dt, cfg.env.base_controller
    gc = critical_gain(dt)
    out = Path(args.out) if args.out else Path("experiments") / (
        f"stable_chatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    (out / "plots").mkdir(parents=True, exist_ok=True)

    variants = [("baseline", None, None, 0.2, "no residual"),
                (f"res trained@{beta_trained:g}", agent, params, 1.0, "LPF off"),
                (f"res trained@{beta_trained:g}", agent, params, 0.2, "LPF on")]
    if args.run_dir_nofilter:
        _, ag2, pr2, b2 = load_policy(args.run_dir_nofilter, args.checkpoint)
        variants += [(f"res trained@{b2:g}", ag2, pr2, 1.0, "LPF off (matched)"),
                     (f"res trained@{b2:g}", ag2, pr2, 0.2, "LPF on")]

    ranges = flat_ranges(cfg.env, force_freq=0.0, force_dc_prob=0.0)
    key = jax.random.PRNGKey(args.eval_seed)

    print(f"policy      : {args.run_dir}  ({controller.upper()} base)")
    print(f"stability   : g_crit = {gc:.2f} at {1/dt:.0f} Hz; "
          "restricting to the STABLE region")
    print(f"episodes    : {args.episodes}\n")

    results, rows = {}, []
    stable = None
    for name, ag, pr, beta, note in variants:
        r = rollout(cfg.env, controller, ag, pr, key, args.episodes,
                    ranges, beta)
        if stable is None:
            stable = r["g"] < gc
            print(f"  {int(stable.sum())} of {args.episodes} episodes are in "
                  f"the stable region\n")
            header = (f"  {'arm':<22}{'LPF':<18}{'motor HF':>10}"
                      f"{'base HF':>10}{'resid HF':>10}{'sat %':>8}"
                      f"{'med RMSE':>10}")
            print(header)
        key_name = f"{name} | {note}"
        results[key_name] = r
        m = r["mask"]
        hf_motor = hp_component(r["motor"][:, :, 0], m, dt, args.hf_cutoff)
        hf_base = hp_component(r["u_base"][:, :, 0], m, dt, args.hf_cutoff)
        hf_res = hp_component(r["u_res"][:, :, 0], m, dt, args.hf_cutoff)
        med = np.median(np.sqrt(np.mean(r["pos_err"][:, stable] ** 2, axis=0)))
        row = {"arm": name, "lpf": note,
               "motor_hf_N": float(hf_motor[stable].mean()),
               "base_wrench_hf_N": float(hf_base[stable].mean()),
               "residual_wrench_hf_N": float(hf_res[stable].mean()),
               "motor_hf_p95_N": float(np.percentile(hf_motor[stable], 95)),
               "saturation_pct": float(100 * r["sat"][:, stable].mean()),
               "median_rmse": float(med)}
        rows.append(row)
        print(f"  {name:<22}{note:<18}{row['motor_hf_N']:>10.4f}"
              f"{row['base_wrench_hf_N']:>10.4f}"
              f"{row['residual_wrench_hf_N']:>10.4f}"
              f"{row['saturation_pct']:>8.2f}{row['median_rmse']:>10.4f}")

    # ---------------- authority dose-response --------------------------------
    print("\n  residual authority sweep (LPF off), stable region:")
    print(f"  {'authority':>10}{'motor HF':>11}{'resid HF':>11}{'base HF':>10}")
    sweep = []
    base_auth = cfg.env.residual_authority
    for mult in args.authority_sweep:
        r = rollout(cfg.env, controller, agent, params, key, args.episodes,
                    ranges, 1.0, authority=base_auth * mult)
        m = r["mask"]
        hm = hp_component(r["motor"][:, :, 0], m, dt, args.hf_cutoff)[stable].mean()
        hr = hp_component(r["u_res"][:, :, 0], m, dt, args.hf_cutoff)[stable].mean()
        hb = hp_component(r["u_base"][:, :, 0], m, dt, args.hf_cutoff)[stable].mean()
        sweep.append({"authority_mult": mult, "motor_hf_N": float(hm),
                      "residual_wrench_hf_N": float(hr),
                      "base_wrench_hf_N": float(hb)})
        print(f"  {mult:>10.2f}{hm:>11.4f}{hr:>11.4f}{hb:>10.4f}")

    with (out / "arms.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (out / "authority_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)

    # ---------------- figures ------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for name, r in results.items():
        sel = np.where(stable)[0]
        freqs, psd = mean_psd(r["motor"][:, sel, 0], r["mask"][:, sel], dt)
        style = "--" if "baseline" in name else "-"
        ax.semilogy(freqs[1:], psd[1:], lw=1.5, ls=style, label=name)
    ax.axvline(args.hf_cutoff, color="grey", ls=":", lw=1)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("mean power, rotor-0 command")
    ax.set_title("Where the extra energy sits (stable region only)\n"
                 "dashed = no residual")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "plots" / "spectra.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    labels = [f"{r['arm']}\n{r['lpf']}" for r in rows]
    x = np.arange(len(rows))
    axes[0].bar(x - 0.2, [r["base_wrench_hf_N"] for r in rows], 0.4,
                label="base wrench HF", color="#888888")
    axes[0].bar(x + 0.2, [r["residual_wrench_hf_N"] for r in rows], 0.4,
                label="residual wrench HF", color="#d62728")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("HF content [N]")
    axes[0].set_title("Attribution: which component carries the chatter?")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    base_line = rows[0]["motor_hf_N"]
    axes[1].bar(x, [r["motor_hf_N"] for r in rows], color="#1f77b4")
    axes[1].axhline(base_line, color="k", ls="--", lw=1.2,
                    label="no-residual floor")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("motor command HF [N]")
    axes[1].set_title("Motor chatter vs the no-residual floor")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plots" / "attribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    mults = [s["authority_mult"] for s in sweep]
    ax.plot(mults, [s["motor_hf_N"] for s in sweep], marker="o",
            label="motor command HF")
    ax.plot(mults, [s["residual_wrench_hf_N"] for s in sweep], marker="s",
            label="residual wrench HF")
    ax.plot(mults, [s["base_wrench_hf_N"] for s in sweep], marker="^",
            label="base wrench HF")
    ax.axhline(rows[0]["motor_hf_N"], color="k", ls="--", lw=1,
               label="no-residual floor")
    ax.set_xlabel("residual authority, as a fraction of the trained value")
    ax.set_ylabel("HF content [N]")
    ax.set_title("Dose-response, LPF off\n"
                 "linear in authority = direct injection")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plots" / "authority_dose_response.png", dpi=150)
    plt.close(fig)

    # ---------------- report --------------------------------------------------
    base_row = rows[0]
    L = ["# Chatter in the gain-stable region\n",
         f"Generated {datetime.now().isoformat(timespec='seconds')}",
         "",
         f"- Policy `{args.run_dir}`, {controller.upper()} base",
         f"- {int(stable.sum())} of {args.episodes} episodes have "
         f"g < g_crit = {gc:.2f}, i.e. the attitude loop is inside its "
         "stability margin. Any chatter here is NOT the gain-margin failure.",
         f"- HF is RMS content above {args.hf_cutoff:g} Hz; the mean rotor "
         "command is about 5 N, so judge the numbers against that.",
         "",
         "## Arms", "",
         "| arm | LPF | motor HF [N] | base wrench HF | residual wrench HF | "
         "sat % | median RMSE |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['arm']} | {r['lpf']} | {r['motor_hf_N']:.4f} "
                 f"| {r['base_wrench_hf_N']:.4f} "
                 f"| {r['residual_wrench_hf_N']:.4f} "
                 f"| {r['saturation_pct']:.2f} | {r['median_rmse']:.4f} |")

    L += ["", "## Is the chatter caused by the residual?", ""]
    worst = max(rows[1:], key=lambda r: r["motor_hf_N"])
    ratio = worst["motor_hf_N"] / max(base_row["motor_hf_N"], 1e-9)
    L += [f"- No-residual floor: **{base_row['motor_hf_N']:.4f} N**. The "
          "baseline controller is not silent; it has its own high-frequency "
          "activity, and that is the number any residual must be judged "
          "against.",
          f"- Worst residual arm: {worst['motor_hf_N']:.4f} N, "
          f"**{ratio:.1f}x** the floor.",
          "",
          "Attribution, from the base and residual wrench columns:",
          "",
          "- If the residual column is large and the base column stays near "
          "its no-residual value, the residual is injecting chatter directly "
          "and filtering the command is the right fix.",
          "- If the base column also rises, the baseline is reacting to a "
          "state the residual perturbed. That is a closed-loop interaction, "
          "and filtering only the residual will not fully remove it.",
          "",
          "## Authority dose-response (LPF off)", "",
          "| authority | motor HF | residual HF | base HF |",
          "|---|---|---|---|"]
    for s in sweep:
        L.append(f"| {s['authority_mult']:.2f} | {s['motor_hf_N']:.4f} "
                 f"| {s['residual_wrench_hf_N']:.4f} "
                 f"| {s['base_wrench_hf_N']:.4f} |")
    L += ["",
          "Roughly linear scaling with authority indicates direct injection. A "
          "sharp threshold or a non-monotone response would indicate a "
          "resonance in the closed loop instead.",
          ""]
    if not args.run_dir_nofilter:
        L += ["> **Caveat.** Only a filter-trained policy was supplied, so the "
              "`LPF off` row is off its training distribution and overstates "
              "the filter's effect. Pass `--run_dir_nofilter` with a policy "
              "trained at beta = 1.0 for a matched comparison.\n"]
    L += ["## Files", "",
          "- `arms.csv`, `authority_sweep.csv`",
          "- `plots/spectra.png` — where the extra energy sits",
          "- `plots/attribution.png` — base vs residual wrench",
          "- `plots/authority_dose_response.png`"]
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    write_manifest(out, {"run_dir": args.run_dir,
                         "run_dir_nofilter": args.run_dir_nofilter,
                         "g_crit": gc, "episodes": args.episodes,
                         "n_stable": int(stable.sum()),
                         "hf_cutoff": args.hf_cutoff})
    print(f"\nwrote {out}/REPORT.md")


if __name__ == "__main__":
    main()
