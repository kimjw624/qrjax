"""Shared evaluation machinery for the fixes study.

One evaluation = one arm (env config + optional policy) flown on one fixed
bank of episodes. The bank is defined by (eval_seed, n_episodes) only: each
episode gets its own PRNG key, so episode i is the same plant, force and start
in every arm, every variant and every chunk size. Arms are therefore paired
episode by episode, and the draws are asserted identical across arms.

Unlike the paper's survivor bank, nothing is filtered: every episode is kept,
terminations are recorded, and statistics are reported both over all episodes
(failures charged at the 2 m threshold) and over the episodes every compared
arm survives (the paper's convention, made explicit).

Results are cached per arm under <results>/cache so that every stage can ask
for the same arm without re-flying it.
"""

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402

from qrjax.envs import Config, EnvConfig, VecEnv     # noqa: E402
from qrjax.envs.config import FIXED_ENV, LEGACY_ENV, legacy_env  # noqa: E402
from qrjax.rl import SAC                             # noqa: E402
from qrjax.rl.curriculum import flat_ranges          # noqa: E402
from qrjax.utils import load_params                  # noqa: E402

CACHE_VERSION = "v1"
HF_CUTOFF_HZ = 10.0          # same cutoff as run_final_study.hp_rms
PAPER_HZ = 200.0
PAPER_EPISODE_S = 10.0

# The paper's Table 1 and effect sizes (KSAS 2026 Fall, final 0910).
PAPER_TABLE1 = {
    # arm in run_final_study naming: (mean, sd, median, p90, sat%)
    "PD": (0.1346, 0.0698, 0.1181, 0.2342, 0.362),
    "PID": (0.0617, 0.0519, 0.0390, 0.1364, 0.508),
    "PD + RRL + LPF": (0.0458, 0.0254, 0.0406, 0.0792, 0.270),
    "PID + RRL + LPF": (0.0230, 0.0108, 0.0211, 0.0321, 0.356),
}
PAPER_EFFECTS = {
    # base: rrl mean/median/p90 reduction %, lpf roughness/sat/median %
    "PD": {"rrl_mean_pct": 66.0, "rrl_median_pct": 65.6,
           "lpf_roughness_pct": 53.4, "lpf_sat_pct": 35.7, "lpf_median_pct": 17.8},
    "PID": {"rrl_mean_pct": 62.6, "rrl_median_pct": 46.0, "rrl_p90_pct": 76.4,
            "lpf_roughness_pct": 49.0, "lpf_sat_pct": 16.4, "lpf_median_pct": 3.2},
}

# Single fixes, in the order they are reported. "terminations" changes only
# which episodes count as failed.
SINGLE_FIXES = {
    "mixer_frd": {"mixer_geometry": "frd"},
    "fd_counter": {"fd_counter_fix": True},
    "analytic_rates": {"rate_ff": "analytic"},
    "prioritized_alloc": {"allocator": "prioritized"},
    "start_on_ref": {"start_on_reference": True},
    "terminations": {"term_ref_error": 3.0, "term_heading_deg": 90.0},
}
FIX_LABELS = {
    "original": "original code",
    "mixer_frd": "mixer roll/pitch signs (FRD)",
    "fd_counter": "FD: two-sample gate for omega_d_dot",
    "analytic_rates": "analytic desired rates (+accelerometer)",
    "prioritized_alloc": "prioritized allocation",
    "start_on_ref": "start on reference attitude/rate",
    "terminations": "reference-error + heading terminations",
    "all_fixes": "all fixes",
}


def all_fixes():
    return dict(FIXED_ENV)


def leave_one_out(name):
    d = all_fixes()
    for k in SINGLE_FIXES[name]:
        d[k] = LEGACY_ENV[k]
    return d


# ------------------------------------------------------------------ configs
def paper_env(**overrides):
    """EnvConfig exactly as the paper's run_final_study built it, plus overrides.

    Starts from the ORIGINAL defaults (legacy_env): since v2 the EnvConfig
    defaults are the fixes, and every study variant here is "original code +
    named overrides"."""
    cfg = legacy_env()
    cfg.dt = 1.0 / PAPER_HZ
    cfg.episode_steps = int(round(PAPER_HZ * PAPER_EPISODE_S))
    cfg.history = 10
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown EnvConfig field {k!r}")
        setattr(cfg, k, v)
    return cfg.validate()


def env_from_run(run_dir, **overrides):
    """The env a trained run used (its config.json), plus overrides."""
    cfg = Config.from_dict(json.loads((Path(run_dir) / "config.json").read_text()))
    env = EnvConfig(**{**cfg.env.__dict__})
    for k, v in overrides.items():
        setattr(env, k, v)
    return env.validate()


def load_policy(run_dir, checkpoint="best"):
    run_dir = Path(run_dir)
    cfg = Config.from_dict(json.loads((run_dir / "config.json").read_text()))
    ckpt = run_dir / "checkpoints" / (
        checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt")
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    agent = SAC(cfg.env.obs_dim(), cfg.env.action_dim(), cfg.sac)
    params = load_params(ckpt, agent.init(jax.random.PRNGKey(0)).actor_params)
    return cfg, agent, params


# ------------------------------------------------------------------ paper runs
def paper_policies(paper_dir):
    """{(seed, tag, base): trial_dir} for a run_final_study experiment."""
    out = {}
    paper_dir = Path(paper_dir)
    for sd in sorted(paper_dir.glob("seed_*")):
        try:
            seed = int(sd.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        for tag in ("none", "thrust"):
            for base in ("pd", "pid"):
                trials = sorted(sd.glob(f"{tag}/seed_*/{base}/train/trial_*"))
                trials = [t for t in trials if (t / "checkpoints" / "best.pt").is_file()]
                if trials:
                    out[(seed, tag, base)] = trials[-1]
    return out


def paper_best_seed(paper_dir):
    m = Path(paper_dir) / "manifest.json"
    if m.is_file():
        try:
            return int(json.loads(m.read_text()).get("best_seed"))
        except Exception:
            pass
    return None


# ------------------------------------------------------------------ bank
def bank_keys(eval_seed, ids):
    """Episode keys. Episode i's key is fold_in(PRNGKey(eval_seed), i), so it
    does not depend on how many episodes are flown or how they are chunked."""
    base = jax.random.PRNGKey(int(eval_seed))
    ids = jnp.arange(int(ids)) if np.isscalar(ids) else jnp.asarray(ids)
    return jax.vmap(lambda i: jax.random.fold_in(base, i))(ids)


def _hp_rms(x, dt, cutoff=HF_CUTOFF_HZ):
    """RMS of the > cutoff content of a 1-D signal (demeaned)."""
    n = x.shape[0]
    if n < 64:
        return np.nan
    X = np.fft.rfft(x - x.mean())
    X[np.fft.rfftfreq(n, d=dt) < cutoff] = 0.0
    return float(np.sqrt(np.mean(np.fft.irfft(X, n=n) ** 2)))


METRIC_FIELDS = (
    "terminated", "term_step", "rmse_charged", "rmse_alive", "sat_pct",
    "roughness", "motor_hf", "motor_hf_r0", "res_f_hf", "res_m_hf",
    "res_f_slew", "res_f_rms", "res_m_rms", "wdd_max", "heading_min_cos",
    "k", "force", "kf", "moment", "arm", "g_eff",
)


def _episode_metrics(pos_err, motor, wrench, residual, sat, term, wdd, hcos, dt,
                     term_pos_error):
    """Per-episode metrics from per-step arrays of ONE chunk (T, E, ...)."""
    T, E = pos_err.shape
    dead = np.cumsum(term, axis=0) > 0          # True from the terminating step on
    alive_prev = np.vstack([np.ones((1, E), bool), ~dead[:-1]])
    terminated = dead[-1]
    term_step = np.where(terminated, np.argmax(term, axis=0), -1)
    out = {k: np.full(E, np.nan) for k in METRIC_FIELDS}
    out["terminated"] = terminated.astype(float)
    out["term_step"] = term_step.astype(float)
    charged = np.where(alive_prev, pos_err, term_pos_error)
    out["rmse_charged"] = np.sqrt(np.mean(charged ** 2, axis=0))
    for e in range(E):
        L = T if not terminated[e] else int(term_step[e]) + 1
        a = slice(0, L)
        out["rmse_alive"][e] = np.sqrt(np.mean(pos_err[a, e] ** 2))
        out["sat_pct"][e] = 100.0 * np.mean(sat[a, e])
        dw = np.diff(wrench[a, e], axis=0)
        out["roughness"][e] = np.sqrt(np.mean(np.sum(dw ** 2, axis=-1))) if L > 1 else np.nan
        hf = [_hp_rms(motor[a, e, j], dt) for j in range(4)]
        out["motor_hf"][e] = float(np.mean(hf))
        out["motor_hf_r0"][e] = hf[0]
        out["res_f_hf"][e] = _hp_rms(residual[a, e, 0], dt)
        out["res_m_hf"][e] = float(np.sqrt(np.sum(
            [_hp_rms(residual[a, e, j], dt) ** 2 for j in (1, 2, 3)])))
        df = np.diff(residual[a, e, 0])
        out["res_f_slew"][e] = float(np.mean(np.abs(df)) / dt) if L > 1 else np.nan
        out["res_f_rms"][e] = float(np.sqrt(np.mean(residual[a, e, 0] ** 2)))
        out["res_m_rms"][e] = float(np.sqrt(np.mean(np.sum(residual[a, e, 1:] ** 2, -1))))
        out["wdd_max"][e] = float(np.max(wdd[a, e]))
        out["heading_min_cos"][e] = float(np.min(hcos[a, e]))
    return out


_COMPILED = {}


def _runner(env_cfg, agent, chunk):
    """Compiled (params, keys) -> per-step arrays for one chunk. Cached per config."""
    sig = json.dumps(env_cfg.__dict__, sort_keys=True, default=str) + \
        f"|{agent is not None}|{chunk}|{None if agent is None else agent.cfg.hidden}"
    if sig in _COMPILED:
        return _COMPILED[sig]
    env = VecEnv(env_cfg, chunk)
    ranges = env.broadcast_ranges(flat_ranges(env_cfg, force_freq=0.0, force_dc_prob=0.0))
    T = env_cfg.episode_steps

    def run(params, keys):
        state = env._reset(keys, ranges, jnp.zeros(chunk, jnp.int32))
        obs = env._reset_obs(state)
        draw = {"k": state.k,
                "force": jnp.linalg.norm(state.external_force, axis=-1),
                "kf": state.mixer_true.kf_scale[:, 0],
                "moment": state.mixer_true.moment_scale[:, 0],
                "arm": state.mixer_true.arm_scale[:, 0]}

        def body(carry, _):
            state, obs = carry
            if agent is None:
                action = jnp.zeros((chunk, 4))
            else:
                action = jnp.tanh(agent.actor.apply(params, obs)[0])
            state, obs, _, _, info = env.step(state, action, ranges)
            return (state, obs), (info["pos_err_desired"], info["motor_cmd"],
                                  info["u_total"], info["residual"],
                                  info["saturated"], info["terminated"],
                                  info["omega_d_dot_norm"], info["heading_cos"])

        _, tr = jax.lax.scan(body, (state, obs), None, length=T)
        return tr, draw

    fn = jax.jit(run)
    _COMPILED[sig] = fn
    return fn


def fly(env_cfg, agent=None, params=None, eval_seed=20260829, n_episodes=512,
        chunk=256, trace_ids=()):
    """Fly one arm on the bank. Returns (per-episode metrics dict, traces).

    ``traces`` maps a requested episode id to its per-step arrays
    (pos_err, motor_cmd, u_total, residual, saturated).
    """
    chunk = int(min(chunk, n_episodes))
    fn = _runner(env_cfg, agent, chunk)
    res = {k: [] for k in METRIC_FIELDS}
    traces = {}
    trace_ids = set(int(i) for i in trace_ids)
    for c0 in range(0, n_episodes, chunk):
        idx = np.arange(c0, min(c0 + chunk, n_episodes))
        kk = bank_keys(eval_seed, idx)
        if idx.size < chunk:                       # pad the last chunk
            kk = jnp.concatenate([kk, jnp.repeat(kk[-1:], chunk - idx.size, 0)])
        tr, draw = fn(params, kk)
        tr = [np.asarray(x)[:, :idx.size] for x in tr]
        pos_err, motor, wrench, residual, sat, term, wdd, hcos = tr
        m = _episode_metrics(pos_err, motor, wrench, residual, sat, term, wdd,
                             hcos, env_cfg.dt, env_cfg.term_pos_error)
        d = {k: np.asarray(v)[:idx.size] for k, v in draw.items()}
        m["k"], m["force"], m["kf"] = d["k"], d["force"], d["kf"]
        m["moment"], m["arm"] = d["moment"], d["arm"]
        m["g_eff"] = d["kf"] * d["arm"] / d["k"]
        for k in METRIC_FIELDS:
            res[k].append(np.asarray(m[k], dtype=float))
        for j, e in enumerate(idx):
            if int(e) in trace_ids:
                traces[int(e)] = {"pos_err": pos_err[:, j], "motor": motor[:, j],
                                  "wrench": wrench[:, j], "residual": residual[:, j],
                                  "sat": sat[:, j]}
    return {k: np.concatenate(v) for k, v in res.items()}, traces


def fly_traces(env_cfg, run_dir=None, checkpoint="best", eval_seed=20260829,
               episode_ids=(), chunk=256, n_episodes=512):
    """Per-step traces of selected bank episodes (same compiled function as fly)."""
    ids = [int(i) for i in episode_ids]
    if not ids:
        return {}
    agent, params = (None, None) if run_dir is None else load_policy(run_dir, checkpoint)[1:]
    chunk = int(min(chunk, n_episodes))
    fn = _runner(env_cfg, agent, chunk)
    kk = bank_keys(eval_seed, ids + [ids[-1]] * (chunk - len(ids)))
    tr, _ = fn(params, kk)
    tr = [np.asarray(x) for x in tr]
    names = ("pos_err", "motor", "wrench", "residual", "sat", "term")
    return {e: {n: tr[k][:, j] for k, n in enumerate(names)} for j, e in enumerate(ids)}


_LATE_FIELDS = {"accel_source": "accelerometer", "pid_ei_limit": 1.0}


def cached_fly(results, label, env_cfg, run_dir=None, checkpoint="best",
               eval_seed=20260829, n_episodes=512, chunk=256):
    """``fly`` with a per-arm cache under <results>/cache. Returns metrics dict."""
    # Fields added to EnvConfig after the cache was first written are left out of
    # the key while at their default, so existing cache entries stay valid.
    env_key = {k: v for k, v in env_cfg.__dict__.items()
               if not (k in _LATE_FIELDS and v == _LATE_FIELDS[k])}
    ident = {"v": CACHE_VERSION, "env": env_key,
             "run": None if run_dir is None else str(Path(run_dir).resolve()),
             "ckpt": checkpoint, "seed": int(eval_seed), "n": int(n_episodes)}
    if run_dir is not None:
        ck = Path(run_dir) / "checkpoints" / f"{checkpoint}.pt"
        ident["ckpt_mtime"] = ck.stat().st_mtime if ck.is_file() else None
    h = hashlib.sha1(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = Path(results) / "cache" / f"{h}.npz"
    if path.is_file():
        z = np.load(path)
        return {k: z[k] for k in METRIC_FIELDS}
    t0 = time.time()
    if run_dir is None:
        agent, params = None, None
    else:
        _, agent, params = load_policy(run_dir, checkpoint)
    m, _ = fly(env_cfg, agent, params, eval_seed, n_episodes, chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **m)
    (path.with_suffix(".json")).write_text(json.dumps(
        {"label": label, **ident, "seconds": round(time.time() - t0, 1)},
        indent=1, default=str))
    print(f"    flew {label:<55} {time.time() - t0:6.1f} s", flush=True)
    return m


def check_same_draws(arms):
    """All arms must have flown identical episodes."""
    ref = None
    for name, m in arms.items():
        d = np.stack([m["k"], m["force"], m["kf"], m["moment"], m["arm"]])
        if ref is None:
            ref, ref_name = d, name
        elif d.shape != ref.shape or not np.allclose(d, ref, atol=1e-6):
            raise AssertionError(f"episode draws differ between {ref_name} and {name}")


# ------------------------------------------------------------------ statistics
def summarize(m, mask=None):
    """Statistics of one arm. ``mask`` selects episodes for conditional stats."""
    n = m["rmse_charged"].size
    surv = m["terminated"] < 0.5
    sel = surv if mask is None else (mask & surv)
    r = m["rmse_alive"][sel]

    def q(x, p):
        return float(np.percentile(x, p)) if x.size else np.nan

    def mean(x):
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else np.nan

    return {
        "n": n, "n_cond": int(sel.sum()),
        "term_pct": 100.0 * float(np.mean(~surv)),
        "charged_mean": float(np.mean(m["rmse_charged"])),
        "charged_median": float(np.median(m["rmse_charged"])),
        "cond_mean": mean(r), "cond_sd": float(np.std(r, ddof=1)) if r.size > 1 else np.nan,
        "cond_median": q(r, 50), "cond_p90": q(r, 90),
        "sat_pct": mean(m["sat_pct"][sel]),
        "roughness": mean(m["roughness"][sel]),
        "motor_hf": mean(m["motor_hf"][sel]),
        "motor_hf_p95": q(m["motor_hf"][sel][np.isfinite(m["motor_hf"][sel])], 95),
        "motor_hf_max": float(np.nanmax(m["motor_hf"][sel])) if sel.any() else np.nan,
        "res_f_hf": mean(m["res_f_hf"][sel]),
        "res_m_hf": mean(m["res_m_hf"][sel]),
        "res_f_slew": mean(m["res_f_slew"][sel]),
        "sat_gt5_pct": 100.0 * float(np.mean(m["sat_pct"][sel] > 5.0)) if sel.any() else np.nan,
    }


def chatter_tail(m, ref, mask=None):
    """Per-episode motor HF relative to a reference arm on the same episodes."""
    ok = (m["terminated"] < 0.5) & (ref["terminated"] < 0.5)
    if mask is not None:
        ok &= mask
    ratio = m["motor_hf"][ok] / np.maximum(ref["motor_hf"][ok], 1e-6)
    if ratio.size == 0:
        return {"hf_ratio_median": np.nan, "hf_ratio_p95": np.nan,
                "hf_ratio_max": np.nan, "n_gt3x": 0, "n_gt10x": 0, "n_paired": 0}
    return {"hf_ratio_median": float(np.median(ratio)),
            "hf_ratio_p95": float(np.percentile(ratio, 95)),
            "hf_ratio_max": float(np.max(ratio)),
            "n_gt3x": int(np.sum(ratio > 3)), "n_gt10x": int(np.sum(ratio > 10)),
            "n_paired": int(ratio.size)}


def joint_survivors(arms):
    ms = list(arms.values())
    s = np.ones(ms[0]["terminated"].size, bool)
    for m in ms:
        s &= m["terminated"] < 0.5
    return s


def own_failures(m, ref):
    """Episodes the arm terminated that the reference arm survived."""
    return int(np.sum((m["terminated"] > 0.5) & (ref["terminated"] < 0.5)))


# ------------------------------------------------------------------ io
def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def write_per_episode(path, arms):
    rows = []
    for name, m in arms.items():
        for e in range(m["rmse_charged"].size):
            # "label", not "arm": METRIC_FIELDS has an "arm" (arm-length scale) column
            rows.append({"label": name, "episode": e,
                         **{k: float(m[k][e]) for k in METRIC_FIELDS}})
    write_csv(path, rows)


def md_table(rows, cols, fmt=None):
    fmt = fmt or {}
    L = ["| " + " | ".join(h for h, _ in cols) + " |",
         "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for _, k in cols:
            v = r.get(k, "")
            f = fmt.get(k)
            if isinstance(v, float):
                cells.append("–" if not np.isfinite(v) else (f.format(v) if f else f"{v:.4g}"))
            else:
                cells.append(str(v))
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def results_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def device_info():
    try:
        x64 = bool(jax.config.jax_enable_x64)
    except Exception:                                        # noqa: BLE001
        x64 = None
    return {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
            "x64": x64,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")}
