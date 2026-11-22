"""Stage 6 -- why the unfiltered residual sometimes oscillates violently, and
where the small one-step rotor spikes come from.

Part 1  Which episodes go extreme (from the evaluation cache): disturbance
        draws of the extreme-chatter episodes against the rest.
Part 2  Case studies. The worst unfiltered paper episode (original code, PID
        and PD) and a typical high-authority fixed-code episode are re-flown
        with the controller internals logged and with one path cut at a time:
          +thrust LPF   the trained policy unchanged, thrust LPF added at test time
          Δf = 0        residual thrust channel zeroed
          ΔM = 0        residual moment channels zeroed
          analytic / finite-difference   the other desired-rate feedforward
          FF accel -> model   analytic feedforward fed the model acceleration,
                        so the residual thrust cannot reach omega_d through it
          no rate FF    omega_d = omega_d_dot = 0
        and the base moment is split into its terms (-kR eR, -kΩ eΩ,
        J ω̇_d, -kI eI) to see which one carries the chatter.
        Caveat: cutting a path also moves the closed loop away from what the
        policy was trained on; "+thrust LPF" is the only cut that leaves
        tracking intact, so read the others together with it.
Part 3  One-step rotor spikes: every rotor jump in N episodes is labelled by
        what happened at that step (episode start, translational-integrator
        clip onset, saturation onset, anti-windup toggle, other). 3b: which
        episodes the spikes after the start come from, and the two worst
        episodes per original-code arm re-flown with the allocator fix and the
        desired-rate fix one at a time.

Writes <results>/05_diagnosis/.
"""

import argparse
from pathlib import Path

import numpy as np

from . import common as C
from .s5_figures import (Arms, SLOT, INK, INK2, MUTED, BASE_GRAY, SURFACE, plain_log, save)
from .s5_figures import plt  # noqa: F401  (style already applied)

import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
import matplotlib.pyplot as plt                      # noqa: E402,F811
from matplotlib.lines import Line2D                  # noqa: E402

from qrjax.core.controller import Gains              # noqa: E402

FIELDS = ("pos_err_desired", "motor_cmd", "u_base", "residual", "saturated", "terminated",
          "ctrl_A", "ctrl_omega_d", "ctrl_omega_d_dot", "ctrl_eR", "ctrl_eOmega", "ctrl_ei",
          "ctrl_eI", "hold")
EXTREME_N = 0.3
SKIP = 100            # steps (0.5 s) excluded from the chatter statistics: start transient


# ------------------------------------------------------------------ instrumented flight
_FN = {}


def fly_instr(env_cfg, run_dir, eval_seed, ids, mask=None, chunk=32, fields=None):
    """Per-step internals for bank episodes ``ids`` (dict field -> (T, E, ...))."""
    agent, params = (None, None) if run_dir is None else C.load_policy(run_dir)[1:]
    ids = [int(i) for i in ids]
    chunk = int(min(chunk, max(1, len(ids))))
    FIELDS_ = FIELDS if fields is None else tuple(fields)
    sig = (repr(sorted(env_cfg.__dict__.items())), agent is not None, chunk,
           None if mask is None else tuple(mask), FIELDS_)
    if sig not in _FN:
        env = C.VecEnv(env_cfg, chunk)
        ranges = env.broadcast_ranges(C.flat_ranges(env_cfg, force_freq=0.0, force_dc_prob=0.0))
        m = None if mask is None else jnp.asarray(mask, jnp.float32)

        def run(params, keys):
            st = env._reset(keys, ranges, jnp.zeros(chunk, jnp.int32))
            ob = env._reset_obs(st)

            def body(c, _):
                st, ob = c
                act = (jnp.zeros((chunk, 4)) if agent is None
                       else jnp.tanh(agent.actor.apply(params, ob)[0]))
                if m is not None:
                    act = act * m
                st, ob, _, _, info = env.step(st, act, ranges)
                return (st, ob), {k: info[k] for k in FIELDS_}
            _, tr = jax.lax.scan(body, (st, ob), None, length=env_cfg.episode_steps)
            return tr
        _FN[sig] = jax.jit(run)
    fn = _FN[sig]
    out = {k: [] for k in FIELDS_}
    for c0 in range(0, len(ids), chunk):
        part = ids[c0:c0 + chunk]
        kk = C.bank_keys(eval_seed, part + [part[-1]] * (chunk - len(part)))
        tr = fn(params, kk)
        for k in FIELDS_:
            out[k].append(np.asarray(tr[k])[:, :len(part)])
    return {k: np.concatenate(v, axis=1) for k, v in out.items()}


def hp(x, dt, cut=10.0):
    """RMS of the > cut Hz content along axis 0."""
    X = np.fft.rfft(x - x.mean(axis=0), axis=0)
    X[np.fft.rfftfreq(x.shape[0], dt) < cut] = 0
    return np.sqrt(np.mean(np.fft.irfft(X, n=x.shape[0], axis=0) ** 2, axis=0))


def dominant_hz(x, dt):
    X = np.abs(np.fft.rfft(x - x.mean()))
    f = np.fft.rfftfreq(len(x), dt)
    X[f < 10] = 0
    return float(f[np.argmax(X)])


def period2(x):
    """Fraction of steps on which x reverses direction (1 = flips every step)."""
    d = np.sign(np.diff(x))
    return float(np.mean(d[:-1] * d[1:] < 0))


def case_metrics(tr, env_cfg, e=0):
    dt = env_cfg.dt
    g = Gains.make(env_cfg.base_controller, mass=env_cfg.mass_nom, gravity=env_cfg.gravity,
                   dt=dt, J=env_cfg.J_nom)
    J = np.diag(np.asarray(g.J))
    w = slice(SKIP, None)
    s = lambda k: tr[k][w, e]
    A = s("ctrl_A")
    b3d = A / np.linalg.norm(A, axis=1, keepdims=True)
    terms = {"-kR eR": -g.kR * s("ctrl_eR"), "-kΩ eΩ": -g.kOmega * s("ctrl_eOmega"),
             "J ω̇_d": J * s("ctrl_omega_d_dot"), "-kI eI": -g.kI * s("ctrl_eI")}
    df = s("residual")[:, 0]
    r = {"motor_hf": float(np.mean(hp(s("motor_cmd"), dt))),
         "sat_pct": 100 * float(np.mean(s("saturated"))),
         "rmse_cm": 100 * float(np.sqrt(np.mean(tr["pos_err_desired"][:, e] ** 2))),
         "terminated": bool(tr["terminated"][:, e].any()),
         "df_hf": float(hp(df, dt)), "df_period2": period2(df), "df_hz": dominant_hz(df, dt),
         "b3d_hf_mrad": 1e3 * float(np.linalg.norm(hp(b3d[:, :2], dt))),
         "wd_hf": float(np.linalg.norm(hp(s("ctrl_omega_d")[:, :2], dt))),
         "wdd_hf": float(np.linalg.norm(hp(s("ctrl_omega_d_dot")[:, :2], dt))),
         "Mbase_hf": float(np.linalg.norm(hp(s("u_base")[:, 1:3], dt))),
         "Mres_hf": float(np.linalg.norm(hp(s("residual")[:, 1:3], dt)))}
    for k, v in terms.items():
        r[f"term {k}"] = float(np.linalg.norm(hp(v[:, :2], dt)))
    return r


# ------------------------------------------------------------------ part 1
def part_where(A, out):
    rows, L = [], []
    sets = [("PID+RRL, no LPF (paper, original code)", "PID_paper_nolpf"),
            ("PD+RRL, no LPF (paper, original code)", "PD_paper_nolpf"),
            ("PID+RRL, no LPF (paper policy, every fix)", "PID_paperfx_nolpf"),
            ("PID+RRL, no LPF, τ 0.05, α 0.6 (fixed code)", "hi_nolpf")]
    figdata = {}
    for lab, k in sets:
        if not A.has(k):
            continue
        M = [A.m(k, s) for s in A.seeds if A.has(k, s)]
        cat = lambda f: np.concatenate([m[f] for m in M])
        al = cat("terminated") < 0.5
        hf = cat("motor_hf")
        ext, oth = al & (hf > EXTREME_N), al & (hf <= EXTREME_N)
        r = {"set": lab, "episodes": int(al.sum()), "extreme": int(ext.sum())}
        for f in ("k", "kf", "arm", "moment", "force", "g_eff"):
            v = cat(f)
            r[f"{f} extreme"] = float(np.median(v[ext])) if ext.any() else np.nan
            r[f"{f} others"] = float(np.median(v[oth]))
        rows.append(r)
        figdata[lab] = (cat("g_eff")[al], cat("force")[al], hf[al])
    C.write_csv(out / "where_extreme.csv", rows)
    show = [{"set": r["set"], "n": f"{r['extreme']} / {r['episodes']}",
             **{f: f"{r[f + ' extreme']:.2f} vs {r[f + ' others']:.2f}"
                for f in ("k", "kf", "arm", "force", "g_eff")}} for r in rows]
    L += ["## 1. Which episodes go extreme", "",
          f"Extreme = per-episode motor-command HF RMS > {EXTREME_N} N. Medians of the "
          "disturbance draw, extreme vs other episodes (all seeds). `g = kf·arm/k` is the "
          "attitude loop's gain relative to nominal.", "",
          C.md_table(show, [("set", "set"), ("extreme / episodes", "n"),
                            ("mass·inertia k", "k"), ("thrust coeff kf", "kf"),
                            ("arm", "arm"), ("|F| [N]", "force"), ("g = kf·arm/k", "g_eff")]),
          ""]
    if figdata:
        fig, axes = plt.subplots(1, len(figdata), figsize=(4.4 * len(figdata), 3.9),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, (lab, (g, F, hf)) in zip(axes[0], figdata.items()):
            e = hf > EXTREME_N
            ax.scatter(g[~e], F[~e], s=6, color=MUTED, alpha=0.35, lw=0, label="other")
            ax.scatter(g[e], F[e], s=22, color=SLOT[1], edgecolor=SURFACE, lw=0.6,
                       label=f"extreme ({e.sum()})")
            ax.set_title(lab, fontsize=9)
            ax.set_xlabel("g = kf·arm/k")
        axes[0, 0].set_ylabel("|external force| [N]")
        axes[0, 0].legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        L += [f"![]({save(fig, out, 'diag1_where_extreme')})", ""]
    return L


# ------------------------------------------------------------------ part 2
def pick_cases(A):
    cases = []
    for base in ("PID", "PD"):
        k = f"{base}_paper_nolpf"
        best = None
        for s in A.seeds:
            if A.has(k, s):
                m = A.m(k, s)
                hf = np.where(m["terminated"] < 0.5, m["motor_hf"], -1)
                e = int(np.argmax(hf))
                if best is None or hf[e] > best[2]:
                    best = (s, e, hf[e])
        if best:
            s, e, _ = best
            run = A.spec[k]["runs"][s]
            lpf_run = A.spec[f"{base}_paper_lpf"]["runs"].get(s)
            cases.append({"name": f"{base} paper, original code, worst episode",
                          "tag": f"{base.lower()}_paper", "seed": s, "ep": e, "run": run,
                          "env": A.spec[k]["envs"][s], "lpf_run": lpf_run, "code": "original"})
    if A.has("hi_nolpf"):
        cand = sorted((float(np.mean(A.m("hi_nolpf", s)["motor_hf"][
            A.m("hi_nolpf", s)["terminated"] < 0.5])), s) for s in A.seeds if A.has("hi_nolpf", s))
        _, s = cand[len(cand) // 2]
        m = A.m("hi_nolpf", s)
        al = m["terminated"] < 0.5
        e = int(np.argsort(np.where(al, m["motor_hf"], np.inf))[al.sum() // 2])
        cases.append({"name": "fixed code, τ 0.05, α 0.6, typical episode", "tag": "hi_fixed",
                      "seed": s, "ep": e, "run": A.spec["hi_nolpf"]["runs"][s],
                      "env": A.spec["hi_nolpf"]["envs"][s],
                      "lpf_run": A.spec["hi_lpf"]["runs"].get(s) if A.has("hi_lpf", s) else None,
                      "code": "fixed"})
    return cases


def variants_for(case):
    v = [("as flown", {}, None), ("+thrust LPF (test time)", {"thrust_filter_beta": 0.2}, None),
         ("Δf = 0", {}, [0, 1, 1, 1]), ("ΔM = 0", {}, [1, 0, 0, 0])]
    if case["code"] == "original":
        v += [("analytic rates", {"rate_ff": "analytic"}, None),
              ("no rate FF (ω_d = 0)", {"rate_ff": "none"}, None)]
    else:
        v += [("FF accel -> model", {"accel_source": "model"}, None),
              ("no rate FF (ω_d = 0)", {"rate_ff": "none"}, None),
              ("finite-difference rates", {"rate_ff": "finite_difference"}, None)]
    return v


def chain_figure(case, trs, out):
    """Zoomed time series around the strongest chatter: as flown vs +thrust LPF."""
    tr0 = trs["as flown"]
    df = tr0["residual"][:, 0, 0]
    T = len(df)
    act = np.convolve(np.abs(np.diff(df, 2)), np.ones(40), "same")
    act[:SKIP] = 0
    c = int(np.clip(np.argmax(act), 30, T - 31))
    w = slice(c - 30, c + 30)
    dt = case["env"].dt
    g = Gains.make(case["env"].base_controller, mass=2.0, gravity=9.807, dt=dt,
                   J=case["env"].J_nom)
    t = np.arange(T)[w] * dt
    shown = [k for k in ("as flown", "+thrust LPF (test time)") if k in trs]
    rows = [("residual thrust Δf [N]", lambda tr: tr["residual"][w, 0, 0]),
            ("desired rate ω_d,x [rad/s]", lambda tr: tr["ctrl_omega_d"][w, 0, 0]),
            ("base moment terms, x [N·m]", None),
            ("rotor commands [N]", lambda tr: tr["motor_cmd"][w, 0, :])]
    fig, axes = plt.subplots(len(rows), len(shown), figsize=(6.2 * len(shown), 8.2),
                             sharex=True, squeeze=False)
    for j, name in enumerate(shown):
        tr = trs[name]
        for i, (lab, fn) in enumerate(rows):
            ax = axes[i, j]
            if fn is None:
                ax.plot(t, -g.kOmega * tr["ctrl_eOmega"][w, 0, 0], color=SLOT[0], lw=1.4,
                        label="-kΩ eΩ")
                ax.plot(t, np.asarray(g.J)[0, 0] * tr["ctrl_omega_d_dot"][w, 0, 0],
                        color=SLOT[1], lw=1.4, label="J ω̇_d")
                ax.plot(t, -g.kR * tr["ctrl_eR"][w, 0, 0], color=SLOT[2], lw=1.4,
                        label="-kR eR")
                ax.plot(t, tr["u_base"][w, 0, 1], color=INK, lw=1.0, ls="--", label="M_base")
                if j == 0:
                    ax.legend(fontsize=7, ncol=4, loc="upper right")
            else:
                y = fn(tr)
                if y.ndim == 1:
                    ax.plot(t, y, color=INK, lw=1.2, marker="o", ms=2)
                else:
                    for r in range(y.shape[1]):
                        ax.plot(t, y[:, r], color=SLOT[r], lw=1.0)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
        axes[0, j].set_title(name, fontsize=10)
        axes[-1, j].set_xlabel("time [s]")
    fig.suptitle(f"{case['name']} (seed {case['seed']}, episode {case['ep']}): "
                 "the chain around the strongest chatter", fontsize=10, fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, out, f"diag2_chain_{case['tag']}")


def part_cases(A, out):
    L = ["## 2. Case studies: cut one path at a time", "",
         "Statistics after the first 0.5 s. `Δf period-2` = fraction of steps on which the "
         "residual thrust reverses (1 = flips every step, a 100 Hz limit cycle at 200 Hz). "
         "HF = RMS above 10 Hz; moment quantities are the norm over body x and y.", ""]
    all_rows = []
    for case in pick_cases(A):
        trs, rows = {}, []
        for name, ov, mask in variants_for(case):
            env = C.EnvConfig(**{**case["env"].__dict__})
            for k, v in ov.items():
                setattr(env, k, v)
            env.validate()
            tr = fly_instr(env, case["run"], A.a.eval_seed, [case["ep"]], mask, chunk=1)
            trs[name] = tr
            rows.append({"case": case["name"], "variant": name, **case_metrics(tr, env)})
        if case.get("lpf_run"):
            tr = fly_instr(C.env_from_run(case["lpf_run"]), case["lpf_run"], A.a.eval_seed,
                           [case["ep"]], None, chunk=1)
            rows.append({"case": case["name"], "variant": "trained-with-LPF policy",
                         **case_metrics(tr, C.env_from_run(case["lpf_run"]))})
        all_rows += rows
        L += [f"### {case['name']} — seed {case['seed']}, episode {case['ep']}", "",
              C.md_table(rows, [("variant", "variant"), ("motor HF [N]", "motor_hf"),
                                ("sat %", "sat_pct"), ("RMSE [cm]", "rmse_cm"),
                                ("Δf HF [N]", "df_hf"), ("Δf period-2", "df_period2"),
                                ("Δf peak [Hz]", "df_hz"), ("b3d HF [mrad]", "b3d_hf_mrad"),
                                ("ω_d HF", "wd_hf"), ("ω̇_d HF", "wdd_hf"),
                                ("M_base HF", "Mbase_hf"), ("-kΩ eΩ", "term -kΩ eΩ"),
                                ("J ω̇_d", "term J ω̇_d"), ("-kR eR", "term -kR eR"),
                                ("ΔM HF", "Mres_hf")],
                         {"motor_hf": "{:.3f}", "sat_pct": "{:.2f}", "rmse_cm": "{:.2f}",
                          "df_hf": "{:.3f}", "df_period2": "{:.2f}", "df_hz": "{:.0f}",
                          "b3d_hf_mrad": "{:.2f}", "wd_hf": "{:.3f}", "wdd_hf": "{:.1f}",
                          "Mbase_hf": "{:.3f}", "term -kΩ eΩ": "{:.3f}",
                          "term J ω̇_d": "{:.3f}", "term -kR eR": "{:.3f}",
                          "Mres_hf": "{:.3f}"}), ""]
        try:
            L += [f"![]({chain_figure(case, trs, out)})", ""]
        except Exception as e:                               # noqa: BLE001
            L += [f"_chain figure failed: {type(e).__name__}: {e}_", ""]
    C.write_csv(out / "case_studies.csv", all_rows)
    return L


# ------------------------------------------------------------------ part 3
def label_spikes(tr, dt, jump=0.3, limit=1.0):
    mot = tr["motor_cmd"]                         # (T, E, 4)
    T, E, _ = mot.shape
    dm = np.abs(np.diff(mot, axis=0)).max(axis=2)  # (T-1, E): jump into step t+1
    ei = np.abs(tr["ctrl_ei"])
    clip_on = np.zeros((T, E), bool)
    clip_on[1:] = ((ei[1:] >= limit - 1e-6) & (ei[:-1] < limit - 1e-6)).any(axis=2)
    sat = tr["saturated"].astype(bool)
    sat_on = np.zeros((T, E), bool)
    sat_on[1:] = sat[1:] & ~sat[:-1]
    hold = tr["hold"].astype(bool)
    hold_tg = np.zeros((T, E), bool)
    hold_tg[1:] = hold[1:] != hold[:-1]
    dead = np.cumsum(tr["terminated"], axis=0) > 0
    res = tr["residual"]
    dres = np.zeros((T, E), bool)
    dres[1:] = ((np.abs(np.diff(res[..., 0], axis=0)) > 0.3)
                | (np.abs(np.diff(res[..., 1:], axis=0)).max(axis=2) > 0.05))
    events = []
    for tt, e in zip(*np.where(dm > jump)):
        s = tt + 1
        if dead[s, e]:
            continue
        near = slice(max(0, s - 2), min(T, s + 2))
        if s * dt < 0.1:
            cause = "episode start (disturbance switched on)"
        elif clip_on[near, e].any():
            cause = "translational integrator reaches its limit"
        elif sat_on[near, e].any():
            cause = "rotor saturation onset"
        elif hold_tg[near, e].any():
            cause = "anti-windup toggle"
        elif dres[near, e].any():
            cause = "residual command jump (policy)"
        else:
            cause = "other"
        events.append((cause, float(dm[tt, e]), int(e), s * dt))
    # event-centred: the kick at EVERY integrator-clip onset, however small
    kicks = []
    for s, e in zip(*np.where(clip_on)):
        if s * dt < 0.1 or dead[s, e]:
            continue
        near = slice(max(0, s - 2), min(T - 1, s + 2))
        base = np.median(dm[max(0, s - 40):max(1, s - 5), e])
        kicks.append((float(dm[near, e].max()), float(base), int(e), s * dt))
    return events, kicks


def part_spikes(A, out, n_ep):
    L = ["## 3. One-step rotor spikes", "",
         f"Every jump of a rotor command larger than 0.3 N between two steps, in "
         f"{n_ep} episodes, labelled by what happened within ±2 steps. Steps after a "
         "termination are ignored.", ""]
    ids = list(range(n_ep))
    rows, krows, example, brk = [], [], None, []
    arms = [("PID baseline", "PID_orig", None), ("PID baseline", "PID_fixed", None),
            ("PID+RRL+LPF (paper)", "PID_paper_lpf", "original"),
            ("PID+RRL+LPF (fixed, retrained)", "PID_fixed_lpf", "fixed")]
    for lab, k, _ in arms:
        if not A.has(k):
            continue
        s = next(s for s in A.seeds if A.has(k, s))
        env = A.spec[k]["envs"][s]
        tr = fly_instr(env, A.spec[k]["runs"].get(s), A.a.eval_seed, ids, None, chunk=32)
        ev, kicks = label_spikes(tr, env.dt)
        code = A.spec[k]["code"]
        if kicks:
            krows.append({"arm": lab, "code": code, "clip onsets": len(kicks),
                          "median kick [N]": float(np.median([x[0] for x in kicks])),
                          "max kick [N]": float(np.max([x[0] for x in kicks])),
                          "typical step change before [N]":
                              float(np.median([x[1] for x in kicks]))})
        for cause in sorted({c for c, *_ in ev}):
            j = [x for x in ev if x[0] == cause]
            rows.append({"arm": lab, "code": code, "cause": cause, "events": len(j),
                         "median jump [N]": float(np.median([x[1] for x in j])),
                         "max jump [N]": float(np.max([x[1] for x in j]))})
        if example is None and k == "PID_orig" and kicks:
            example = (tr, max(kicks)[2], env)
        late = [x for x in ev if not x[0].startswith("episode start")]
        if late:
            brk.append((lab, k, s, env, tr, late, code))
    C.write_csv(out / "spike_events.csv", rows)
    C.write_csv(out / "integrator_kicks.csv", krows)
    L += [C.md_table(rows, [("arm", "arm"), ("code", "code"), ("cause", "cause"),
                            ("events", "events"), ("median jump [N]", "median jump [N]"),
                            ("max jump [N]", "max jump [N]")],
                     {"median jump [N]": "{:.2f}", "max jump [N]": "{:.2f}"}), "",
          "Event-centred: the largest rotor-command change within ±2 steps of EVERY "
          "translational-integrator clip onset (any axis reaching |e_i| = 1), against the "
          "typical step-to-step change just before it.", "",
          C.md_table(krows, [("arm", "arm"), ("code", "code"), ("clip onsets", "clip onsets"),
                             ("median kick [N]", "median kick [N]"),
                             ("max kick [N]", "max kick [N]"),
                             ("typical change before [N]", "typical step change before [N]")],
                     {"median kick [N]": "{:.3f}", "max kick [N]": "{:.3f}",
                      "typical step change before [N]": "{:.3f}"})
          if krows else "_no integrator clip onsets in these episodes_", ""]
    if example:
        tr, e, env = example
        t = np.arange(tr["motor_cmd"].shape[0]) * env.dt
        ei = tr["ctrl_ei"][:, e]
        fig, axes = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True)
        for r in range(4):
            axes[0].plot(t, tr["motor_cmd"][:, e, r], color=SLOT[r], lw=0.9)
        axes[0].set_ylabel("rotor cmd [N]")
        for r, lab in enumerate("xyz"):
            axes[1].plot(t, ei[:, r], color=SLOT[r], lw=1.4, label=f"e_i,{lab}")
        axes[1].axhline(1.0, color=MUTED, ls=":", lw=1)
        axes[1].axhline(-1.0, color=MUTED, ls=":", lw=1)
        axes[1].set_ylabel("translational integral e_i")
        axes[1].legend(fontsize=8, ncol=3, loc="lower right")
        axes[2].plot(t, np.linalg.norm(tr["ctrl_omega_d_dot"][:, e], axis=1), color=INK, lw=1)
        axes[2].set_yscale("log")
        plain_log(axes[2])
        axes[2].set_ylabel("|ω̇_d| [rad/s²]")
        axes[2].set_xlabel("time [s]")
        fig.suptitle(f"PID baseline, original code, episode {e}: the spike when the "
                     "integrator hits its limit", fontsize=10, fontweight="bold", x=0.01,
                     ha="left")
        fig.tight_layout()
        L += [f"![]({save(fig, out, 'diag3_integrator_spike')})", ""]
    L += spike_episodes(A, out, brk)
    return L


# knock-outs for the episodes behind the non-start spikes (original-code arms only)
_SPIKE_VARIANTS = (("as flown", {}),
                   ("+prioritized allocator", {"allocator": "prioritized"}),
                   ("+analytic rates", {"rate_ff": "analytic", "fd_counter_fix": True}),
                   ("+both", {"allocator": "prioritized", "rate_ff": "analytic",
                              "fd_counter_fix": True}))


def _draws(A, k, s, n):
    try:
        m = A.m(k, s)
        return {f: np.asarray(m[f])[:n] for f in ("k", "kf", "arm", "force", "g_eff")}
    except Exception:                                    # noqa: BLE001
        return None


def _flight_row(tr, e, dt):
    term = np.where(tr["terminated"][:, e])[0]
    end = int(term[0]) if len(term) else tr["motor_cmd"].shape[0]
    dm = np.abs(np.diff(tr["motor_cmd"][:, e], axis=0)).max(axis=1)
    lo = int(0.1 / dt)
    wdd = np.linalg.norm(tr["ctrl_omega_d_dot"][:, e], axis=1)
    return {"terminated [s]": end * dt if len(term) else np.nan,
            "jumps > 0.3 N": int((dm[lo:max(lo, end - 1)] > 0.3).sum()),
            "sat %": 100 * float(tr["saturated"][:end, e].mean()),
            "max |ω̇_d|": float(wdd[lo:end].max()) if end > lo else np.nan}


def spike_episodes(A, out, brk, top=3):
    """Which episodes the non-start spikes come from, and which fix removes them."""
    if not brk:
        return []
    import dataclasses
    L = ["### 3b. Episodes behind the spikes after the start", "",
         "Spikes after the first 0.1 s, per episode: are they spread over many episodes "
         "or concentrated in a few (and do those episodes end in a termination)? "
         "`g = kf·arm/k`; 'sat %' = share of steps with a rotor at its limit.", ""]
    rows, ko = [], []
    for lab, k, s, env, tr, late, code in brk:
        dr = _draws(A, k, s, tr["motor_cmd"].shape[1])
        eps = {}
        for cause, j, e, t in late:
            eps.setdefault(e, []).append((cause, j, t))
        order = sorted(eps, key=lambda e: -len(eps[e]))
        for e in order[:top]:
            ev = eps[e]
            causes = {}
            for c, *_ in ev:
                causes[c] = causes.get(c, 0) + 1
            r = {"arm": lab, "code": code, "episode": e, "events": len(ev),
                 "share of arm's late events %": 100 * len(ev) / len(late),
                 "max jump [N]": max(j for _, j, _ in ev),
                 "t range [s]": f"{min(t for *_, t in ev):.2f}–{max(t for *_, t in ev):.2f}",
                 "causes": "; ".join(f"{c} ×{n}" for c, n in
                                     sorted(causes.items(), key=lambda x: -x[1]))}
            r.update(_flight_row(tr, e, env.dt))
            if dr is not None:
                r.update({("arm_scale" if f == "arm" else f): float(dr[f][e]) for f in dr})
            rows.append(r)
        r0 = {"arm": lab, "code": code, "episode": "all",
              "events": len(late), "share of arm's late events %": 100.0,
              "causes": f"{len(eps)} episodes"}
        rows.append(r0)
        if code != "original":
            continue
        run = A.spec[k]["runs"].get(s)
        for e in order[:2]:
            for name, ov in _SPIKE_VARIANTS:
                try:
                    e2 = dataclasses.replace(env, **ov)
                    t2 = fly_instr(e2, run, A.a.eval_seed, [e], None, chunk=1)
                except Exception as exc:                 # noqa: BLE001
                    ko.append({"arm": lab, "episode": e, "variant": name,
                               "terminated [s]": f"failed: {exc}"})
                    continue
                ko.append({"arm": lab, "episode": e, "variant": name,
                           **_flight_row(t2, 0, env.dt)})
    C.write_csv(out / "spike_episodes.csv", rows)
    C.write_csv(out / "spike_episode_knockouts.csv", ko)
    f3 = "{:.2f}"
    L += [C.md_table(rows, [("arm", "arm"), ("code", "code"), ("episode", "episode"),
                            ("late events", "events"),
                            ("share %", "share of arm's late events %"),
                            ("causes", "causes"), ("t [s]", "t range [s]"),
                            ("max jump [N]", "max jump [N]"),
                            ("terminated [s]", "terminated [s]"), ("sat %", "sat %"),
                            ("k", "k"), ("kf", "kf"), ("arm scale", "arm_scale"), ("|F| [N]", "force"),
                            ("g", "g_eff")],
                     {"share of arm's late events %": "{:.0f}", "max jump [N]": f3,
                      "terminated [s]": f3, "sat %": "{:.0f}", "k": f3, "kf": f3,
                      "arm_scale": f3, "force": f3, "g_eff": f3}), ""]
    if ko:
        L += ["The two episodes with the most late spikes per original-code arm, re-flown "
              "(same policy) with the allocator fix and the desired-rate fix switched on one "
              "at a time.", "",
              C.md_table(ko, [("arm", "arm"), ("episode", "episode"), ("variant", "variant"),
                              ("terminated [s]", "terminated [s]"),
                              ("jumps > 0.3 N", "jumps > 0.3 N"), ("sat %", "sat %"),
                              ("max |ω̇_d| [rad/s²]", "max |ω̇_d|")],
                         {"terminated [s]": f3, "sat %": "{:.0f}", "max |ω̇_d|": "{:.1f}"}),
              "", "_'–' under terminated = survived the full 10 s_", ""]
    return L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="study_results")
    p.add_argument("--paper_dir", default="auto")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--eval_seed", type=int, default=20260829)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--spike_episodes", type=int, default=64)
    p.add_argument("--parts", default="1,2,3")
    a = p.parse_args()
    a.case = None
    out = C.results_dir(Path(a.results) / "05_diagnosis")
    A = Arms(a)
    parts = set(a.parts.split(","))
    L = ["# 05 — Diagnosis: violent chatter and one-step spikes", ""]
    for key, fn in (("1", lambda: part_where(A, out)), ("2", lambda: part_cases(A, out)),
                    ("3", lambda: part_spikes(A, out, a.spike_episodes))):
        if key in parts:
            try:
                L += fn()
            except Exception as e:                           # noqa: BLE001
                import traceback
                traceback.print_exc()
                L += [f"_part {key} failed: {type(e).__name__}: {e}_", ""]
            (out / "DIAGNOSIS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
