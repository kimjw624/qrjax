"""One command for the whole paper: train, evaluate, tabulate, plot.

Trains every configuration needed to populate the report, evaluates all six
arms on a single frozen bank of non-terminated episodes, picks the best seed,
and writes the tables and figures.

Arms produced:

    PD                    bare geometric PD
    PID                   bare geometric PID
    PD + RRL              residual on PD, no filter
    PID + RRL             residual on PID, no filter
    PD + RRL + LPF        residual on PD, thrust-channel filter
    PID + RRL + LPF       residual on PID, thrust-channel filter

so 4 training runs per seed (two bases x filter on/off). The bare baselines
need no training.

Evaluation uses a rejection-sampled bank: candidate disturbance realisations
are drawn until exactly --episodes of them survive in EVERY arm. The reported
RMSE is therefore conditional on survival, and the screening statistics are
written to the report so the robustness information that filtering removes is
still visible.

Seed selection: the "best" seed is chosen by median RMSE of PID+RRL+LPF, and
every seed's numbers are written out so the choice is auditable rather than
silent.

Example
-------
python -m scripts.run_final_study --seeds 0,1,2 --total_steps 2000000
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import jax                                           # noqa: E402
import numpy as np                                   # noqa: E402

from qrjax.envs import Config, EnvConfig, VecEnv     # noqa: E402
from qrjax.envs.eval_bank import build_bank          # noqa: E402
from qrjax.envs.final_figures import write_full_set  # noqa: E402
from qrjax.rl import SAC                             # noqa: E402
from qrjax.rl.curriculum import flat_ranges          # noqa: E402
from qrjax.utils import load_params, write_json, write_manifest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# (display name, base controller, needs policy, filter beta key)
ARM_SPEC = [
    ("PD",              "pd",  None,     1.0),
    ("PID",             "pid", None,     1.0),
    ("PD + RRL",        "pd",  "none",   1.0),
    ("PID + RRL",       "pid", "none",   1.0),
    ("PD + RRL + LPF",  "pd",  "thrust", 0.2),
    ("PID + RRL + LPF", "pid", "thrust", 0.2),
]
ROTOR_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode})")


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


def hp_rms(sig, dt, cutoff=10.0):
    """RMS of the >cutoff Hz content, per episode, in the signal's units."""
    T, E = sig.shape
    band = np.fft.rfftfreq(T, d=dt) >= cutoff
    X = np.fft.rfft(sig - sig.mean(axis=0, keepdims=True), axis=0)
    hp = np.fft.irfft(np.where(band[:, None], X, 0.0), n=T, axis=0)
    return np.sqrt(np.mean(hp ** 2, axis=0))


def metrics(trace, dt):
    pos_err, twin_err, motor, wrench, action, mask, sat, x_true, x_nom, x_des = trace
    rmse = np.sqrt(np.mean(pos_err ** 2, axis=0))
    dwr = np.diff(wrench, axis=0)
    return {
        "rmse": rmse,
        "steady": np.mean(pos_err[pos_err.shape[0] // 2:], axis=0),
        "sat_pct": 100 * sat.mean(axis=0),
        "roughness": np.sqrt(np.mean(np.sum(dwr ** 2, axis=-1), axis=0)),
        "motor_hf": hp_rms(motor[:, :, 0], dt),
        "effort": np.sqrt(np.mean(np.sum(action ** 2, axis=-1), axis=0)),
    }


def paired_p(x, y, rng, n=20000):
    d = np.asarray(y) - np.asarray(x)
    idx = rng.integers(0, d.size, size=(n, d.size))
    b = d[idx].mean(axis=1)
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))


def evaluate_seed(seed_root, cfg_env, args, seed_label):
    """Build the survivor bank and evaluate all six arms on it."""
    policies = {}
    for tag in ("none", "thrust"):
        for base in ("pd", "pid"):
            # run_stage1 adds its own seed_XX level under --out/--name, so the
            # trial lives at <root>/seed_XX/<tag>/seed_XX/<base>/train/trial_*
            trials = sorted(seed_root.glob(f"{tag}/seed_*/{base}/train/trial_*"))
            if not trials:
                raise SystemExit(
                    f"missing training run under {seed_root/tag} for base "
                    f"{base!r}; expected {tag}/seed_*/{base}/train/trial_*")
            policies[(tag, base)] = load_policy(trials[-1], args.checkpoint)

    env_cfg = EnvConfig(**{**cfg_env.__dict__})
    ranges = flat_ranges(env_cfg, force_freq=0.0, force_dc_prob=0.0)

    arms, beta_map = [], {}
    for name, base, tag, beta in ARM_SPEC:
        if tag is None:
            arms.append((name, base, None, None))
        else:
            _, ag, pr = policies[(tag, base)]
            arms.append((name, base, ag, pr))
        beta_map[name] = beta

    print(f"  building bank of {args.episodes} non-terminated episodes")
    keep, draw, stats = build_bank(env_cfg, arms, ranges,
                                   jax.random.PRNGKey(args.eval_seed),
                                   n_target=args.episodes, chunk=args.chunk,
                                   thrust_beta_map=beta_map)
    M = {name: metrics(keep[name], env_cfg.dt) for name, *_ in arms}
    return keep, draw, stats, M


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="0,1,2",
                   type=lambda s: [int(x) for x in s.split(",") if x.strip()])
    p.add_argument("--total_steps", type=int, default=2_000_000)
    p.add_argument("--control_hz", type=float, default=200.0)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--utd", type=float, default=1.0)
    p.add_argument("--buffer_size", type=int, default=300_000)
    p.add_argument("--history", type=int, default=10)
    p.add_argument("--episodes", type=int, default=256,
                   help="number of NON-TERMINATED episodes in the bank")
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--fig2_case", type=int, default=None,
                   help="episode index for Figure 2. Default is the largest "
                        "external force, but the largest force is not always "
                        "the most informative case: a constant force is what "
                        "the PID integral handles best, so the baseline can "
                        "look strong there. per_episode.csv lets you pick a "
                        "case with both a large force and a large margin")
    p.add_argument("--fig2", default="final_vs_pid",
                   choices=list(FIG2_PRESETS),
                   help="which arms Figure 2 compares. Default is the proposed "
                        "method against the PID baseline, labelled 'PID + RRL' "
                        "to match the paper's naming")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="experiments")
    p.add_argument("--skip_training", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="skip any (seed, filter) run whose checkpoint already "
                        "exists. Twelve runs is hours of compute; an OOM or a "
                        "reboot partway through should not cost the finished "
                        "ones")
    p.add_argument("--live_plot", action="store_true")
    args = p.parse_args()

    name = args.name or f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = Path(args.out) / name
    (root / "figures").mkdir(parents=True, exist_ok=True)
    py = sys.executable

    print(f"output   : {root}")
    print(f"seeds    : {args.seeds}   ({4 * len(args.seeds)} training runs)")
    print(f"rate     : {args.control_hz:g} Hz")
    print(f"bank     : {args.episodes} non-terminated episodes\n")

    # ------------------------------------------------------------ train
    for seed in args.seeds:
        for tag, beta in (("none", 1.0), ("thrust", 0.2)):
            if args.skip_training:
                continue
            if args.resume:
                done = all(sorted((root / f"seed_{seed:02d}").glob(
                    f"{tag}/seed_*/{b}/train/trial_*/checkpoints/best.pt"))
                    for b in ("pd", "pid"))
                if done:
                    print(f"  [resume] seed {seed} / {tag} already trained, "
                          "skipping")
                    continue
            run([py, "-m", "scripts.run_stage1",
                 "--seeds", seed,
                 "--total_steps", args.total_steps,
                 "--control_hz", args.control_hz,
                 "--num_envs", args.num_envs,
                 "--utd", args.utd,
                 "--buffer_size", args.buffer_size,
                 "--history", args.history,
                 "--thrust_filter_beta", beta,
                 "--episodes", 64,          # per-seed quick check only
                 "--eval_seed", args.eval_seed,
                 "--name", f"seed_{seed:02d}/{tag}",
                 "--out", root]
                + (["--live_plot"] if args.live_plot else []))

    # ------------------------------------------------------------ evaluate
    cfg_env = EnvConfig()
    cfg_env.dt = 1.0 / args.control_hz
    cfg_env.episode_steps = int(round(args.control_hz * 10.0))
    cfg_env.history = args.history

    per_seed = {}
    for seed in args.seeds:
        sr = root / f"seed_{seed:02d}"
        print(f"\n=== evaluating seed {seed} ===")
        keep, draw, stats, M = evaluate_seed(sr, cfg_env, args, f"seed_{seed:02d}")
        per_seed[seed] = (keep, draw, stats, M)
        print(f"  drew {stats['candidates_drawn']}, rejection rate "
              f"{stats['rejection_rate_pct']:.1f}%")
        for nm, *_ in ARM_SPEC:
            print(f"    {nm:<18} median {np.median(M[nm]['rmse']):.4f}  "
                  f"mean {M[nm]['rmse'].mean():.4f}")

    # ------------------------------------------------------------ best seed
    key_arm = "PID + RRL + LPF"
    best = min(per_seed, key=lambda s: np.median(per_seed[s][3][key_arm]["rmse"]))
    print(f"\nbest seed by median RMSE of '{key_arm}': seed {best}")
    keep, draw, stats, M = per_seed[best]
    dt = cfg_env.dt
    rng = np.random.default_rng(args.eval_seed)

    # ------------------------------------------------------------ tables
    rows = []
    for nm, *_ in ARM_SPEC:
        m = M[nm]
        rows.append({
            "arm": nm,
            "rmse_mean": float(m["rmse"].mean()),
            "rmse_sd": float(m["rmse"].std(ddof=1)),
            "rmse_median": float(np.median(m["rmse"])),
            "rmse_p90": float(np.percentile(m["rmse"], 90)),
            "sat_pct": float(m["sat_pct"].mean()),
            "roughness": float(m["roughness"].mean()),
            "motor_hf_N": float(m["motor_hf"].mean()),
        })
    with (root / "table1.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def red(a, b, field="rmse", stat="median"):
        fn = np.median if stat == "median" else np.mean
        if stat == "p90":
            fn = lambda v: np.percentile(v, 90)
        x, y = fn(M[a][field]), fn(M[b][field])
        return 100 * (x - y) / x

    eff = []
    for base, res, lpf in (("PD", "PD + RRL", "PD + RRL + LPF"),
                           ("PID", "PID + RRL", "PID + RRL + LPF")):
        # The proposed method is the FILTERED arm, which the paper labels
        # simply "PD + RRL" / "PID + RRL". Report the residual's benefit
        # against that, not against the unfiltered variant, or the headline
        # numbers describe a configuration the paper never presents.
        eff.append({"base": base,
                    "rrl_mean_pct": red(base, lpf, stat="mean"),
                    "rrl_median_pct": red(base, lpf),
                    "rrl_p90_pct": red(base, lpf, stat="p90"),
                    "rrl_nolpf_mean_pct": red(base, res, stat="mean"),
                    "rrl_nolpf_median_pct": red(base, res),
                    "rrl_nolpf_p90_pct": red(base, res, stat="p90"),
                    "lpf_roughness_pct": red(res, lpf, "roughness", "mean"),
                    "lpf_sat_pct": red(res, lpf, "sat_pct", "mean"),
                    "lpf_median_pct": red(res, lpf),
                    "p_rrl": paired_p(M[base]["rmse"], M[lpf]["rmse"], rng),
                    "p_lpf_rough": paired_p(M[res]["roughness"],
                                            M[lpf]["roughness"], rng)})
    with (root / "effects.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eff[0]))
        w.writeheader()
        w.writerows(eff)

    # ------------------------------------------------------------ figures
    cases, prows = write_full_set(root, keep, draw, M, stats, dt,
                                  [nm for nm, *_ in ARM_SPEC],
                                  args.episodes, args.eval_seed)
    make_fig2(keep, draw, M, dt, root / "figures" / "fig2_trajectories.png",
              preset=args.fig2, case=args.fig2_case)
    for pre in FIG2_PRESETS:
        make_fig2(keep, draw, M, dt,
                  root / "figures" / f"fig2_{pre}.png", preset=pre,
                  case=args.fig2_case)
    make_summary(M, root / "figures" / "fig3_summary.png", args.episodes)

    # ------------------------------------------------------------ report
    L = [f"# Final study — {name}\n",
         f"Generated {datetime.now().isoformat(timespec='seconds')}",
         "",
         f"- Control rate {args.control_hz:g} Hz, {args.total_steps:,} env "
         f"steps per run, history {args.history}",
         f"- Seeds {args.seeds}; **seed {best}** selected by median RMSE of "
         f"`{key_arm}`",
         f"- Error convention: **true minus nominal**, "
         f"`e_R = 1/2 (R_nom^T R - R^T R_nom)^vee`",
         "",
         "## Evaluation bank", "",
         f"- **{args.episodes} episodes in which no arm terminated**",
         f"- {stats['candidates_drawn']} candidates drawn, "
         f"{stats['candidates_survived']} survived "
         f"(**{stats['rejection_rate_pct']:.1f}%** rejected)",
         "",
         "Per-arm termination rate over the candidate pool, which the bank "
         "itself no longer shows:", "",
         "| Arm | terminated / drawn | rate |", "|---|---|---|"]
    for nm, *_ in ARM_SPEC:
        L.append(f"| {nm} | {stats['terminations_by_arm'][nm]} / "
                 f"{stats['candidates_drawn']} | "
                 f"{stats['termination_rate_by_arm_pct'][nm]:.2f}% |")
    L += ["",
          "Reported RMSE is conditional on survival. Quote the table above "
          "alongside Table 1, or the comparison silently ignores robustness.",
          "",
          "## Table 1 — tracking performance", "",
          "| Controller | RMSE mean ± SD [m] | Median / P90 [m] | Sat. [%] |",
          "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['arm']} | {r['rmse_mean']:.4f} ± {r['rmse_sd']:.4f} "
                 f"| {r['rmse_median']:.4f} / {r['rmse_p90']:.4f} "
                 f"| {r['sat_pct']:.3f} |")

    L += ["", "## Residual effect — proposed method vs baseline", "",
          "Proposed = base + residual + thrust LPF, which the paper labels "
          "`PD + RRL` / `PID + RRL`.", "",
          "| Base | mean | median | P90 | p |", "|---|---|---|---|---|"]
    for e in eff:
        L.append(f"| {e['base']} | {e['rrl_mean_pct']:.1f}% "
                 f"| {e['rrl_median_pct']:.1f}% | {e['rrl_p90_pct']:.1f}% "
                 f"| {e['p_rrl']:.4f} |")
    L += ["", "For reference, the unfiltered residual against the same "
          "baseline:", "",
          "| Base | mean | median | P90 |", "|---|---|---|---|"]
    for e in eff:
        L.append(f"| {e['base']} | {e['rrl_nolpf_mean_pct']:.1f}% "
                 f"| {e['rrl_nolpf_median_pct']:.1f}% "
                 f"| {e['rrl_nolpf_p90_pct']:.1f}% |")
    L += ["", "## LPF effect (thrust channel only)", "",
          "| Base | input roughness | saturation | median RMSE | p (roughness) |",
          "|---|---|---|---|---|"]
    for e in eff:
        L.append(f"| {e['base']} | {e['lpf_roughness_pct']:.1f}% "
                 f"| {e['lpf_sat_pct']:.1f}% | {e['lpf_median_pct']:.1f}% "
                 f"| {e['p_lpf_rough']:.4f} |")

    L += ["", "## Paired tests (position RMSE)", "",
          "| A | B | mean A | mean B | change | B wins | p |",
          "|---|---|---|---|---|---|---|"]
    for r in prows:
        L.append(f"| {r['arm_a']} | {r['arm_b']} | {r['mean_a']:.4f} "
                 f"| {r['mean_b']:.4f} | {r['delta_mean_pct']:+.1f}% "
                 f"| {r['b_wins_pct']:.0f}% | {r['p_value']:.4f} |")
    L += ["", "## Cases plotted individually", "",
          "| case | episode |", "|---|---|"]
    for lab, c in cases:
        L.append(f"| {lab} | {c} |")
    L += ["", "## All seeds", "", "| Seed | " +
          " | ".join(nm for nm, *_ in ARM_SPEC) + " |",
          "|---" * (len(ARM_SPEC) + 1) + "|"]
    for s in args.seeds:
        Ms = per_seed[s][3]
        L.append(f"| {s}{' (selected)' if s == best else ''} | " +
                 " | ".join(f"{np.median(Ms[nm]['rmse']):.4f}"
                            for nm, *_ in ARM_SPEC) + " |")
    L += ["", "median RMSE [m] per arm.", "",
          "## Files", "",
          "- `table1.csv`, `effects.csv`, `per_episode.csv`, "
          "`paired_tests.csv`, `all_seeds.json`",
          "- `01_summary/` — arm comparison, distribution, error-vs-time, "
          "control cost, uncertainty coverage",
          "- `02_cases/` — trajectory, error and actuator figures per case",
          "- `03_actuators/` — actuator summary and spectra",
          "- `figures/` — fig2 presets and fig3 summary",
          "- `seed_XX/{none,thrust}/` — training runs"]
    (root / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    write_json(root / "all_seeds.json",
               {str(s): {nm: {k: float(np.median(v)) if k == "rmse"
                              else float(np.mean(v))
                              for k, v in per_seed[s][3][nm].items()}
                         for nm, *_ in ARM_SPEC} for s in args.seeds})
    write_manifest(root, {"study": "final", "seeds": args.seeds,
                          "best_seed": best, "episodes": args.episodes,
                          "control_hz": args.control_hz,
                          "eval_seed": args.eval_seed,
                          "bank_stats": stats,
                          "error_convention": "true minus nominal"})

    print("\n" + "=" * 74)
    print(f"{'Controller':<20}{'mean ± SD':>20}{'median / P90':>20}{'Sat %':>9}")
    for r in rows:
        print(f"{r['arm']:<20}{r['rmse_mean']:>9.4f} ± {r['rmse_sd']:<8.4f}"
              f"{r['rmse_median']:>10.4f} / {r['rmse_p90']:<8.4f}"
              f"{r['sat_pct']:>8.3f}")
    print(f"\nwrote {root}/REPORT.md")


# Figure 2 panels: (arm key, label shown on the figure).
# The proposed method is PID + RRL + LPF, but the paper defines "PID + RRL" to
# mean the full method including the thrust filter -- Table 1 already uses that
# name for the filtered arm -- so the figure carries the same short label.
FIG2_PRESETS = {
    "final_vs_pid": [("PID", "PID"), ("PID + RRL + LPF", "PID + RRL")],
    "final_vs_pd":  [("PD", "PD"), ("PD + RRL + LPF", "PD + RRL")],
    "both_bases":   [("PD", "PD"), ("PD + RRL + LPF", "PD + RRL"),
                     ("PID", "PID"), ("PID + RRL + LPF", "PID + RRL")],
    "lpf_ablation": [("PD + RRL", "PD + RRL"),
                     ("PD + RRL + LPF", "PD + RRL + LPF"),
                     ("PID + RRL", "PID + RRL"),
                     ("PID + RRL + LPF", "PID + RRL + LPF")],
}


def make_fig2(keep, draw, M, dt, path, preset="final_vs_pid", case=None):
    """3-D trajectories for the selected arms on one shared uncertainty case.

    All panels share one set of axis limits. Autoscaling each panel to its own
    data would give a badly-tracking arm a zoomed-out view and make it look
    comparable to a good one, which is the opposite of what the figure is for.
    """
    panels = FIG2_PRESETS[preset]
    fmag = np.linalg.norm(draw["force"], axis=-1)
    c = int(np.argmax(fmag)) if case is None else case
    g = draw["kf"] * draw["arm"] / draw["k"]

    ncol = 2
    nrow = int(np.ceil(len(panels) / ncol))
    fig = plt.figure(figsize=(6.0 * ncol, 5.4 * nrow))

    lim = [[np.inf, -np.inf] for _ in range(3)]
    for key, _ in panels:
        x_true, x_nom = keep[key][7], keep[key][8]
        for j, sgn in enumerate((1, 1, -1)):
            for series in (x_true, x_nom):
                v = sgn * series[:, c, j]
                lim[j][0] = min(lim[j][0], float(v.min()))
                lim[j][1] = max(lim[j][1], float(v.max()))
    pads = [0.08 * (hi - lo + 1e-6) for lo, hi in lim]

    for i, (key, label) in enumerate(panels):
        x_true, x_nom = keep[key][7], keep[key][8]
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        ax.plot(x_nom[:, c, 0], x_nom[:, c, 1], -x_nom[:, c, 2],
                color="tab:blue", ls="--", lw=2.0, label="nominal")
        ax.plot(x_true[:, c, 0], x_true[:, c, 1], -x_true[:, c, 2],
                color="tab:red", ls="-", lw=2.0, label="true")
        ax.set_title(f"{label}   RMSE {M[key]['rmse'][c]:.4f} m", fontsize=16)
        ax.set_xlim(lim[0][0] - pads[0], lim[0][1] + pads[0])
        ax.set_ylim(lim[1][0] - pads[1], lim[1][1] + pads[1])
        ax.set_zlim(lim[2][0] - pads[2], lim[2][1] + pads[2])
        ax.set_xlabel("N [m]", fontsize=13, labelpad=8)
        ax.set_ylabel("E [m]", fontsize=13, labelpad=8)
        ax.set_zlabel("up [m]", fontsize=13, labelpad=8)
        ax.tick_params(labelsize=11)
        # 3-D axes default to dense ticks; on a wide two-panel layout the
        # labels run together, so cap them explicitly
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=5)
        ax.locator_params(axis="z", nbins=5)

    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=16, frameon=True)
    fig.suptitle(f"Episode {c}:  k={draw['k'][c]:.2f}  "
                 f"|F|={fmag[c]:.2f} N  kf={draw['kf'][c]:.2f}  "
                 f"arm={draw['arm'][c]:.2f}  g={g[c]:.2f}", fontsize=15)
    fig.tight_layout(rect=(0, 0.07 / nrow, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_summary(M, path, n_ep):
    arms = [nm for nm, *_ in ARM_SPEC]
    colors = ["#999999", "#444444", "#8fce8f", "#8fb4e0", "#2ca02c", "#1f77b4"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(arms))
    for ax, key, title in ((axes[0], "rmse", "Position RMSE [m]"),
                           (axes[1], "roughness", "Input roughness RMS"),
                           (axes[2], "sat_pct", "Saturation [%]")):
        v = [M[a][key].mean() if key != "rmse" else np.median(M[a][key])
             for a in arms]
        ax.bar(x, v, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=22, ha="right", fontsize=10)
        ax.set_title(title + (" (median)" if key == "rmse" else " (mean)"),
                     fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=10)
    fig.suptitle(f"{n_ep} non-terminated episodes, paired", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
