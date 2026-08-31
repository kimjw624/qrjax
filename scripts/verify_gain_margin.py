"""Is the oscillation a sampled-data gain-margin violation of the attitude loop?

The hypothesis: the flight controller allocates using NOMINAL rotor constants
and arm lengths and assumes NOMINAL inertia, so when the true vehicle differs
it produces

    g = kf * arm / k

times the commanded moment per unit of true inertia. Above a critical g the
discrete-time attitude loop loses stability and limit-cycles near Nyquist.

Correlation and a dose-response curve do not establish this. Six tests do, and
each can independently falsify it:

  T1  dose-response      oscillation must rise monotonically with g and knee
                         near the value linear theory predicts
  T2  theory scaling     the critical g must scale as 1/dt. This is the sharp
                         prediction: continuous-time analysis says the loop is
                         stable for ALL g, so any g-threshold that moves with
                         the sample rate is sampled-data in origin
  T3  product invariance  only the product g should matter. Different
                         (kf, arm, k) triples with the same g must behave
                         identically. If they do not, g is the wrong variable
  T4  true allocation     tell the controller the true rotor constants and
                         arm lengths, which drops g from kf*arm/k to 1/k --
                         below the limit for any k in range. Oscillation must
                         vanish. This removes
                         the hypothesised cause directly rather than reducing
                         it, so it is the decisive test
  T5  PD vs PID           identical at matched g, ruling out integral action
  T6  residual on/off     identical at matched g, ruling out the learned policy

T4 is the one that can kill the hypothesis outright. If oscillation survives
correct allocation, the mismatch was never the cause.

Example
-------
python -m scripts.verify_gain_margin --out experiments/gain_margin
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

from qrjax.core.dynamics import Inertia               # noqa: E402
from qrjax.core.mixer import MixerParams             # noqa: E402
from qrjax.envs import EnvConfig, VecEnv, DisturbRanges   # noqa: E402
from qrjax.utils import write_json, write_manifest   # noqa: E402

J_NOM_ROLL = 0.022
KR, KOMEGA = 8.81, 2.54
BASE_FORCE = np.array([1.3123, -1.1827, -0.762])


# ------------------------------------------------------------ linear theory

def max_closed_loop_pole(g, dt):
    """Largest |pole| of the linearised, ZOH-discretised attitude loop.

    Small-angle: eR -> theta, eOmega -> theta_dot, so
        theta_ddot = (g / J_nom) * M_cmd,   M_cmd = -kR theta - kOmega theta_dot
    Control is held across the step, matching the simulator.
    """
    a = g / J_NOM_ROLL
    Ad = np.array([[1.0, dt], [0.0, 1.0]])
    Bd = np.array([[0.5 * a * dt ** 2], [a * dt]])
    K = np.array([[KR, KOMEGA]])
    return max(abs(np.linalg.eigvals(Ad - Bd @ K)))


def critical_gain(dt):
    lo, hi = 1e-3, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if max_closed_loop_pole(mid, dt) < 1.0:
            lo = mid
        else:
            hi = mid
    return lo


# ------------------------------------------------------------- measurement

def hf_amplitude(x, dt, cutoff=10.0):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    T = len(x)
    freqs = np.fft.rfftfreq(T, d=dt)
    X = np.fft.rfft(x)
    return float(np.sqrt(np.mean(
        np.fft.irfft(np.where(freqs >= cutoff, X, 0.0), n=T)[:T] ** 2)))


def peak_frequency(x, dt, floor=3.0):
    x = np.asarray(x, dtype=float) - np.mean(x)
    freqs = np.fft.rfftfreq(len(x), d=dt)
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec[freqs < floor] = 0.0
    return float(freqs[np.argmax(spec)])


_ENV_CACHE = {}


def _get_env(dt, controller, true_allocation, thrust_beta, duration):
    """Cache one compiled env per distinct CONFIG.

    Plant parameters (mass, inertia, rotor constants, arm) are set through the
    STATE rather than the config, so the whole sweep shares a handful of
    compiled graphs instead of recompiling for every parameter combination.
    Config-level fields are the ones that change the graph; state-level fields
    are just values.
    """
    key = (dt, controller, bool(true_allocation), thrust_beta, duration)
    if key not in _ENV_CACHE:
        cfg = EnvConfig()
        cfg.base_controller = controller
        cfg.dt = dt
        cfg.episode_steps = int(duration / dt)
        cfg.disturbances = ("massmoi",)
        cfg.k_min = cfg.k_max = 1.0
        cfg.allocation_uses_true_params = bool(true_allocation)
        cfg.thrust_filter_beta = thrust_beta
        env = VecEnv(cfg, 1)
        _ENV_CACHE[key] = (cfg, env,
                           env.broadcast_ranges(DisturbRanges.from_config(cfg)),
                           jax.jit(env.step))
    return _ENV_CACHE[key]


def simulate(k, kf, arm, moment=1.0, dt=0.01, controller="pid",
             true_allocation=False, duration=10.0, force=None,
             agent=None, params=None, thrust_beta=0.2):
    """One episode with fully specified plant parameters. Returns diagnostics."""
    cfg, env, ranges, step = _get_env(dt, controller, true_allocation,
                                      thrust_beta, duration)
    state, obs = env.reset(jax.random.PRNGKey(0), ranges, stagger=False)

    J_nom = jnp.diag(jnp.asarray(cfg.J_nom, dtype=jnp.float32))
    state = state._replace(
        external_force=jnp.array([BASE_FORCE if force is None else force]),
        inertia_true=Inertia(mass=jnp.array([k * cfg.mass_nom]),
                             J=jnp.array([k * J_nom])),
        mixer_true=MixerParams(kf_scale=jnp.full((1, 4), kf),
                               moment_scale=jnp.full((1, 4), moment),
                               arm_scale=jnp.full((1, 4), arm)))

    thrust, motor, terminated = [], [], False
    for _ in range(cfg.episode_steps):
        if agent is None:
            action = jnp.zeros((1, env.action_dim))
        else:
            action = jnp.tanh(agent.actor.apply(params, obs)[0])
        state, obs, reward, done, info = step(state, action, ranges)
        thrust.append(float(info["u_total"][0][0]))
        motor.append(np.asarray(info["motor_cmd"][0]))
        if bool(info["terminated"][0]):
            terminated = True
            break

    motor = np.array(motor)
    return {
        "g": float(kf * arm / k),
        "hf_thrust": hf_amplitude(thrust, dt),
        "hf_motor0": hf_amplitude(motor[:, 0], dt),
        "peak_hz": peak_frequency(motor[:, 0], dt),
        "motor_std": float(motor[:, 0].std()),
        "steps": len(thrust),
        "terminated": terminated,
        "trace_motor": motor,
        "dt": dt,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--run_dir", default=None,
                   help="optional trained policy, for the residual on/off test")
    args = p.parse_args()

    out = Path(args.out) if args.out else Path("experiments") / (
        f"gain_margin_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    (out / "plots").mkdir(parents=True, exist_ok=True)

    agent = params = None
    if args.run_dir:
        import json
        from qrjax.envs import Config
        from qrjax.rl import SAC
        from qrjax.utils import load_params
        rd = Path(args.run_dir)
        pcfg = Config.from_dict(json.loads((rd / "config.json").read_text()))
        probe = VecEnv(pcfg.env, 1)
        agent = SAC(probe.obs_dim, probe.action_dim, pcfg.sac)
        params = load_params(rd / "checkpoints" / "best.pt",
                             agent.init(jax.random.PRNGKey(0)).actor_params)
        print(f"loaded policy {rd}")

    rows = []
    sim = lambda **kw: simulate(duration=args.duration, **kw)

    # ---------------- T1 + T2: dose-response at several control rates -------
    print("\nT1/T2  dose-response vs control rate")
    print(f"  {'dt':>7}{'rate':>7}{'g':>7}{'theory':>9}{'HF motor [N]':>14}{'peak Hz':>9}")
    dts = [0.02, 0.01, 0.005]
    gains = [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.6]
    for dt in dts:
        gc = critical_gain(dt)
        for g in gains:
            # realise g with arm alone, holding kf and k at nominal
            r = sim(k=1.0, kf=1.0, arm=g, dt=dt, controller="pid")
            r.update({"test": "T1_dose", "dt": dt, "critical_g": gc,
                      "predicted_unstable": g > gc})
            rows.append({kk: vv for kk, vv in r.items() if kk != "trace_motor"})
            flag = "UNSTABLE" if g > gc else "stable"
            print(f"  {dt:>7}{1/dt:>7.0f}{g:>7.2f}{gc:>9.2f}"
                  f"{r['hf_motor0']:>14.4f}{r['peak_hz']:>9.1f}   theory: {flag}")

    # ---------------- T3: product invariance --------------------------------
    print("\nT3  product invariance: same g, different (kf, arm, k)")
    triples = [(1.0, 2.2, 1.0), (2.2, 1.0, 1.0), (1.1, 1.4, 0.7),
               (1.54, 1.0, 0.7), (1.3, 1.3, 0.768)]
    for kf, arm, k in triples:
        r = sim(k=k, kf=kf, arm=arm, controller="pid")
        r.update({"test": "T3_invariance", "kf": kf, "arm": arm, "k": k})
        rows.append({kk: vv for kk, vv in r.items() if kk != "trace_motor"})
        print(f"  kf={kf:<5} arm={arm:<5} k={k:<6} g={r['g']:.2f}"
              f"   HF {r['hf_motor0']:.4f} N   peak {r['peak_hz']:.1f} Hz")

    # ---------------- T4: true allocation removes the cause -----------------
    print("\nT4  DECISIVE: allocate with true parameters (g -> 1/k, "
          "below the limit)")
    t4 = []
    for g in (2.2, 2.6, 3.0):
        for true_alloc in (False, True):
            r = sim(k=0.71, kf=1.27, arm=g * 0.71 / 1.27, controller="pid",
                    true_allocation=true_alloc)
            r.update({"test": "T4_allocation", "true_allocation": true_alloc})
            rows.append({kk: vv for kk, vv in r.items() if kk != "trace_motor"})
            t4.append((g, true_alloc, r["hf_motor0"]))
            print(f"  g={g:.1f}  true_allocation={str(true_alloc):<5}  "
                  f"HF {r['hf_motor0']:.4f} N   peak {r['peak_hz']:.1f} Hz")

    # ---------------- T5: PD vs PID -----------------------------------------
    print("\nT5  PD vs PID at matched g (rules out integral action)")
    for g in (1.0, 2.2, 3.0):
        line = f"  g={g:.1f}  "
        for ctrl in ("pd", "pid"):
            r = sim(k=1.0, kf=1.0, arm=g, controller=ctrl)
            r.update({"test": "T5_controller", "controller": ctrl})
            rows.append({kk: vv for kk, vv in r.items() if kk != "trace_motor"})
            line += f"{ctrl.upper()} {r['hf_motor0']:.4f} N   "
        print(line)

    # ---------------- T6: residual on/off ------------------------------------
    if agent is not None:
        print("\nT6  residual on/off at matched g (rules out the learned policy)")
        for g in (1.0, 2.2, 3.0):
            line = f"  g={g:.1f}  "
            for use, lab in ((False, "baseline"), (True, "residual")):
                r = sim(k=1.0, kf=1.0, arm=g, controller="pid",
                        agent=agent if use else None,
                        params=params if use else None)
                r.update({"test": "T6_residual", "with_residual": use})
                rows.append({kk: vv for kk, vv in r.items() if kk != "trace_motor"})
                line += f"{lab} {r['hf_motor0']:.4f} N   "
            print(line)

    # ---------------- LPF sweep ---------------------------------------------
    print("\nLPF  thrust filter beta at matched g (how much does it help?)")
    for g in (1.0, 2.6):
        for beta in (1.0, 0.5, 0.2, 0.05):
            r = sim(k=1.0, kf=1.0, arm=g, controller="pid",
                    agent=agent, params=params, thrust_beta=beta)
            r.update({"test": "LPF", "thrust_beta": beta})
            rows.append({kk: vv for kk, vv in r.items() if kk != "trace_motor"})
            print(f"  g={g:.1f}  beta={beta:<5} HF {r['hf_motor0']:.4f} N")

    with (out / "results.csv").open("w", newline="") as f:
        fields, seen = [], set()
        for r in rows:
            for kk in r:
                if kk not in seen:
                    seen.add(kk)
                    fields.append(kk)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---------------- figures -----------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for dt, colour in zip(dts, ["#d62728", "#1f77b4", "#2ca02c"]):
        sel = [r for r in rows if r["test"] == "T1_dose" and r["dt"] == dt]
        ax.semilogy([r["g"] for r in sel], [r["hf_motor0"] for r in sel],
                    marker="o", color=colour, label=f"{1/dt:.0f} Hz measured")
        ax.axvline(critical_gain(dt), color=colour, ls="--", alpha=0.8)
        ax.text(critical_gain(dt), ax.get_ylim()[1], f" theory {1/dt:.0f} Hz",
                rotation=90, va="top", fontsize=8, color=colour)
    ax.set_xlabel("effective attitude gain  g = kf·arm / k")
    ax.set_ylabel("rotor-0 command, RMS above 10 Hz [N]")
    ax.set_title("T1/T2: oscillation onset moves with the control rate\n"
                 "dashed lines = stability limit predicted by linear "
                 "sampled-data theory (no fitting)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "plots" / "T1_T2_dose_response.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sel = [r for r in rows if r["test"] == "T3_invariance"]
    ax.scatter([r["g"] for r in sel], [r["hf_motor0"] for r in sel], s=70)
    for r in sel:
        ax.annotate(f"kf={r['kf']:g}\narm={r['arm']:g}\nk={r['k']:g}",
                    (r["g"], r["hf_motor0"]), fontsize=7,
                    textcoords="offset points", xytext=(8, -6))
    ax.set_xlabel("g = kf·arm / k")
    ax.set_ylabel("rotor-0 HF [N]")
    ax.set_yscale("log")
    ax.set_title("T3: different parameter triples, same g\n"
                 "points at equal g must coincide if g is the right variable")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "plots" / "T3_product_invariance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    gs = sorted({g for g, _, _ in t4})
    width = 0.35
    for i, ta in enumerate((False, True)):
        vals = [next(v for g2, t2, v in t4 if g2 == g and t2 == ta) for g in gs]
        ax.bar(np.arange(len(gs)) + i * width, vals, width,
               label="true allocation" if ta else "nominal allocation")
    ax.set_xticks(np.arange(len(gs)) + width / 2)
    ax.set_xticklabels([f"g = {g:g}" for g in gs])
    ax.set_ylabel("rotor-0 HF [N]")
    ax.set_yscale("log")
    ax.set_title("T4: telling the controller the true parameters\n"
                 "if the gain mismatch is the cause, the right bars collapse")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "plots" / "T4_true_allocation.png", dpi=150)
    plt.close(fig)

    # ---------------- report -------------------------------------------------
    def get(test, **kw):
        for r in rows:
            if r["test"] == test and all(r.get(a) == b for a, b in kw.items()):
                return r
        return None

    L = ["# Is the oscillation a sampled-data gain-margin violation?\n",
         f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
         f"- Episode length {args.duration:g} s, zero residual unless stated",
         "",
         "Hypothesis: the controller allocates with NOMINAL rotor constants and "
         "arm lengths and assumes NOMINAL inertia, so the true loop gain is "
         "`g = kf·arm/k`. Above a critical `g` the discrete attitude loop "
         "limit-cycles. Continuous-time analysis says the loop is stable for "
         "ALL `g`, so any threshold that moves with the sample rate must be "
         "sampled-data in origin.",
         "",
         "## Predicted stability limits (linear theory, no fitting)", "",
         "| control rate | critical g |", "|---|---|"]
    for dt in dts:
        L.append(f"| {1/dt:.0f} Hz | {critical_gain(dt):.2f} |")
    L += ["", "Critical g scales as 1/dt, which is the falsifiable prediction.",
          "", "## T1/T2 — dose response vs control rate", "",
          "| g | " + " | ".join(f"{1/dt:.0f} Hz" for dt in dts) + " |",
          "|---" * (len(dts) + 1) + "|"]
    for g in gains:
        cells = []
        for dt in dts:
            r = next((x for x in rows if x["test"] == "T1_dose"
                      and x["dt"] == dt and abs(x["g"] - g) < 1e-6), None)
            cells.append(f"{r['hf_motor0']:.4f}" if r else "-")
        L.append(f"| {g:g} | " + " | ".join(cells) + " |")

    L += ["", "## T3 — product invariance", "",
          "| kf | arm | k | g | HF [N] | peak Hz |", "|---|---|---|---|---|---|"]
    for r in [x for x in rows if x["test"] == "T3_invariance"]:
        L.append(f"| {r['kf']:g} | {r['arm']:g} | {r['k']:g} | {r['g']:.2f} "
                 f"| {r['hf_motor0']:.4f} | {r['peak_hz']:.1f} |")

    L += ["", "## T4 — true allocation (decisive)", "",
          "| g | nominal allocation | true allocation | reduction |",
          "|---|---|---|---|"]
    for g in gs:
        a = next(v for g2, t2, v in t4 if g2 == g and not t2)
        b = next(v for g2, t2, v in t4 if g2 == g and t2)
        L.append(f"| {g:g} | {a:.4f} | {b:.4f} | {a/max(b,1e-9):.1f}x |")
    L += ["",
          "If the right column is not far below the left, the gain mismatch is "
          "NOT the cause and the hypothesis is dead.",
          "", "## T5 — PD vs PID", "", "| g | PD | PID |", "|---|---|---|"]
    for g in (1.0, 2.2, 3.0):
        a = get("T5_controller", controller="pd", g=g)
        b = get("T5_controller", controller="pid", g=g)
        if a and b:
            L.append(f"| {g:g} | {a['hf_motor0']:.4f} | {b['hf_motor0']:.4f} |")

    lpf = [x for x in rows if x["test"] == "LPF"]
    if lpf:
        L += ["", "## Thrust LPF — how much does it help, and where?", "",
              "| g | beta | HF [N] |", "|---|---|---|"]
        for r in lpf:
            L.append(f"| {r['g']:.1f} | {r['thrust_beta']:g} "
                     f"| {r['hf_motor0']:.4f} |")
        L += ["",
              "The filter acts on the THRUST channel, while the instability is "
              "in the ATTITUDE loop. Expect it to reduce thrust-channel "
              "chatter without moving the attitude limit cycle much -- which is "
              "the point: it treats a symptom on a different channel.",
              ]

    L += ["", "## Files", "",
          "- `results.csv` — every run",
          "- `plots/T1_T2_dose_response.png` — measured onset vs predicted limit",
          "- `plots/T3_product_invariance.png`",
          "- `plots/T4_true_allocation.png`"]
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    write_manifest(out, {"stage": "3-verify", "duration": args.duration,
                         "run_dir": args.run_dir,
                         "critical_gains": {f"{1/dt:.0f}Hz": critical_gain(dt)
                                            for dt in dts}})
    print(f"\nwrote {out}/REPORT.md")


if __name__ == "__main__":
    main()
