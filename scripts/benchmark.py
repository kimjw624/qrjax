"""Find good --num_envs / --utd settings for THIS machine.

Two phases, both optional.

Phase 1 (default, ~2-5 min) measures throughput. It sweeps a grid of
``num_envs`` x ``utd``, times a few compiled iterations of the real training
function, records peak device memory, and extrapolates to a wall-clock estimate
for a full run. Configurations that run out of memory are caught and reported
rather than crashing the sweep.

Phase 2 (``--learning_check``, ~10-25 min) answers the question throughput
cannot: does a cheap ``utd`` still learn? It runs a short real training job at
each candidate ``utd`` on an identical seed and budget, and reports the eval
position RMSE reached. Throughput alone will always recommend the lowest
``utd``, which is exactly the setting most likely to be undertrained, so treat
phase 1 as "what is affordable" and phase 2 as "what is sufficient".

The first timed configuration in each row absorbs XLA compilation, so the
script always runs one untimed warm-up iteration before measuring.

Examples
--------
# Throughput sweep with sensible defaults.
python -m scripts.benchmark

# Wider sweep, and estimate against a 2M-step budget.
python -m scripts.benchmark --num_envs 64,128,256,512,1024 \
    --utd 1.0,0.5,0.25,0.0625 --target_steps 2000000

# Add the learning-quality check.
python -m scripts.benchmark --learning_check --check_steps 60000
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402
import numpy as np                                           # noqa: E402

from qrjax.envs import Config, VecEnv                        # noqa: E402
from qrjax.rl import SAC, buffer as buffer_mod               # noqa: E402
from qrjax.rl.curriculum import flat_ranges                  # noqa: E402
from qrjax.train.loop import (                               # noqa: E402
    build_iteration_fn, _stack_ranges, _assign_ranges,
)
from qrjax.utils import write_json                           # noqa: E402


OOM_MARKERS = ("RESOURCE_EXHAUSTED", "out of memory", "OutOfMemory",
               "Out of memory", "failed to allocate")


def device_report():
    dev = jax.devices()[0]
    info = {"platform": dev.platform, "device": str(dev),
            "device_kind": getattr(dev, "device_kind", "unknown")}
    try:
        stats = dev.memory_stats() or {}
        limit = stats.get("bytes_limit")
        if limit:
            info["memory_limit_gb"] = round(limit / 1e9, 2)
    except Exception:
        pass
    return info


def peak_memory_gb():
    try:
        stats = jax.devices()[0].memory_stats() or {}
        peak = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use")
        return round(peak / 1e9, 3) if peak else None
    except Exception:
        return None


def is_oom(exc):
    text = f"{type(exc).__name__}: {exc}"
    return any(m in text for m in OOM_MARKERS)


def time_config(num_envs, utd, rollout_len, buffer_size, batch_size, repeats):
    """Time one compiled training iteration. Returns a result dict."""
    cfg = Config()
    cfg.train.num_envs = num_envs
    cfg.train.rollout_len = rollout_len
    cfg.train.utd = utd
    cfg.sac.buffer_size = buffer_size
    cfg.sac.batch_size = batch_size

    steps_per_iter = rollout_len * num_envs
    n_updates = max(1, int(round(utd * steps_per_iter)))

    env = VecEnv(cfg.env, num_envs)
    agent = SAC(env.obs_dim, env.action_dim, cfg.sac)
    it_fn = build_iteration_fn(cfg.env, cfg.train, agent, env, n_updates)

    key = jax.random.PRNGKey(0)
    agent_state = agent.init(key)
    buf = buffer_mod.init(buffer_size, env.obs_dim, env.action_dim)
    stacked = _stack_ranges([flat_ranges(cfg.env)])
    ranges = _assign_ranges(stacked, jnp.zeros(num_envs, jnp.int32))
    env_state, obs = env.reset(key, ranges)
    carry = (env_state, obs, agent_state, buf, key)

    # Warm-up absorbs XLA compilation, which otherwise lands entirely on the
    # first timed sample and makes the whole row look slow.
    t_compile0 = time.perf_counter()
    carry, metrics = it_fn(carry, ranges, True)
    jax.block_until_ready(metrics)
    compile_s = time.perf_counter() - t_compile0

    t0 = time.perf_counter()
    for _ in range(repeats):
        carry, metrics = it_fn(carry, ranges, True)
    jax.block_until_ready(metrics)
    iter_s = (time.perf_counter() - t0) / repeats

    result = {
        "num_envs": num_envs, "utd": utd, "rollout_len": rollout_len,
        "steps_per_iter": steps_per_iter, "updates_per_iter": n_updates,
        "iter_ms": round(iter_s * 1000, 1),
        "sps": round(steps_per_iter / iter_s, 1),
        "compile_s": round(compile_s, 1),
        "peak_mem_gb": peak_memory_gb(),
        "ok": True,
    }
    del carry, it_fn, agent, env, buf
    gc.collect()
    return result


def fmt_hours(seconds):
    if not np.isfinite(seconds):
        return "n/a"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    return f"{seconds/3600:.2f} h"


def phase1(args):
    print("=" * 78)
    print("PHASE 1 - throughput")
    print("=" * 78)
    header = (f"{'envs':>6}{'utd':>8}{'steps/it':>10}{'updates':>9}"
              f"{'iter ms':>10}{'env steps/s':>13}{'mem GB':>9}"
              f"{'  ' + str(args.target_steps // 1000) + 'k steps':>13}")
    print(header)
    print("-" * 78)

    results = []
    for num_envs in args.num_envs:
        for utd in args.utd:
            try:
                r = time_config(num_envs, utd, args.rollout_len,
                                args.buffer_size, args.batch_size, args.repeats)
            except Exception as exc:
                kind = "OOM" if is_oom(exc) else type(exc).__name__
                print(f"{num_envs:>6}{utd:>8}{'':>10}{'':>9}{'':>10}"
                      f"{kind:>13}{'':>9}{'':>13}")
                results.append({"num_envs": num_envs, "utd": utd,
                                "ok": False, "error": kind, "detail": str(exc)[:300]})
                gc.collect()
                continue

            est = args.target_steps / r["sps"]
            r["target_steps"] = args.target_steps
            r["estimated_seconds"] = round(est, 1)
            mem = f"{r['peak_mem_gb']:.2f}" if r["peak_mem_gb"] else "-"
            print(f"{r['num_envs']:>6}{r['utd']:>8}{r['steps_per_iter']:>10}"
                  f"{r['updates_per_iter']:>9}{r['iter_ms']:>10.1f}"
                  f"{r['sps']:>13,.0f}{mem:>9}{fmt_hours(est):>13}")
            results.append(r)
    return results


def phase2(args, candidates):
    """Short real training runs to check that a cheap utd still learns."""
    from qrjax.train import train

    print()
    print("=" * 78)
    print(f"PHASE 2 - learning quality ({args.check_steps:,} env steps per utd)")
    print("=" * 78)
    print("Lower eval RMSE is better. Compare against the utd=1.0 row: if a")
    print("cheaper utd reaches a similar RMSE, it is the better setting.")
    print()

    out_root = Path(args.out) / "learning_check"
    rows = []
    for utd in candidates:
        cfg = Config()
        cfg.train.use_curriculum = False
        cfg.train.total_steps = args.check_steps
        cfg.train.num_envs = args.check_num_envs
        cfg.train.rollout_len = args.rollout_len
        cfg.train.utd = utd
        cfg.train.seed = args.seed
        cfg.sac.buffer_size = args.buffer_size
        cfg.sac.batch_size = args.batch_size
        cfg.sac.learning_starts = min(5000, args.check_steps // 8)
        cfg.train.eval_every_iters = max(
            1, (args.check_steps // (args.check_num_envs * args.rollout_len)) // 6)
        cfg.train.eval_episodes = 16
        cfg.train.checkpoint_every_iters = 10 ** 9
        cfg.train.runs_root = str(out_root)
        cfg.train.run_name = f"utd_{str(utd).replace('.', 'p')}"

        t0 = time.perf_counter()
        try:
            run_dir = train(cfg, live_plot=False, verbose=False)
        except Exception as exc:
            kind = "OOM" if is_oom(exc) else type(exc).__name__
            print(f"  utd={utd:<8} FAILED ({kind})")
            rows.append({"utd": utd, "ok": False, "error": kind})
            continue
        elapsed = time.perf_counter() - t0

        from qrjax.utils import read_jsonl
        records = read_jsonl(Path(run_dir) / "metrics.jsonl")
        evals = [r["eval_pos_rmse"] for r in records
                 if r.get("eval_pos_rmse") is not None
                 and np.isfinite(r["eval_pos_rmse"])]
        best = min(evals) if evals else float("nan")
        final = evals[-1] if evals else float("nan")
        print(f"  utd={utd:<8} best eval RMSE {best:.4f} m   "
              f"final {final:.4f} m   wallclock {fmt_hours(elapsed)}")
        rows.append({"utd": utd, "ok": True, "best_eval_pos_rmse": float(best),
                     "final_eval_pos_rmse": float(final),
                     "wallclock_s": round(elapsed, 1), "run_dir": str(run_dir)})
    return rows


def recommend(results, learning_rows, target_steps):
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return ["Every configuration failed. If the errors say OOM, lower "
                "--buffer_size (try 100000) and --num_envs."]

    lines = []
    fastest = max(ok, key=lambda r: r["sps"])
    utd1 = [r for r in ok if r["utd"] == 1.0]
    best_utd1 = max(utd1, key=lambda r: r["sps"]) if utd1 else None

    if best_utd1:
        lines.append(
            f"Matching your PyTorch runs (utd=1.0): "
            f"--num_envs {best_utd1['num_envs']} --utd 1.0  "
            f"-> {best_utd1['sps']:,.0f} steps/s, "
            f"{fmt_hours(target_steps / best_utd1['sps'])} for "
            f"{target_steps:,} steps."
        )
    lines.append(
        f"Fastest measured: --num_envs {fastest['num_envs']} "
        f"--utd {fastest['utd']}  -> {fastest['sps']:,.0f} steps/s, "
        f"{fmt_hours(target_steps / fastest['sps'])}."
    )

    if learning_rows:
        good = [r for r in learning_rows if r.get("ok")
                and np.isfinite(r.get("best_eval_pos_rmse", np.nan))]
        if good:
            ref = next((r for r in good if r["utd"] == 1.0), None)
            if ref:
                tol = ref["best_eval_pos_rmse"] * 1.15
                cheap = [r for r in good if r["best_eval_pos_rmse"] <= tol]
                if cheap:
                    pick = min(cheap, key=lambda r: r["utd"])
                    lines.append(
                        f"Cheapest utd within 15% of utd=1.0 quality: "
                        f"--utd {pick['utd']} "
                        f"(RMSE {pick['best_eval_pos_rmse']:.4f} vs "
                        f"{ref['best_eval_pos_rmse']:.4f}).")
            else:
                pick = min(good, key=lambda r: r["best_eval_pos_rmse"])
                lines.append(f"Best learning quality measured: --utd {pick['utd']} "
                             f"(RMSE {pick['best_eval_pos_rmse']:.4f}).")
    else:
        lines.append("Throughput alone always favours the lowest utd, which is "
                     "also the most likely to be undertrained. Run with "
                     "--learning_check before trusting a low value.")

    mems = [r["peak_mem_gb"] for r in ok if r.get("peak_mem_gb")]
    if mems:
        lines.append(f"Peak device memory across the sweep: {max(mems):.2f} GB.")
    return lines


def parse_list(text, cast):
    return [cast(x) for x in str(text).split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_envs", default="64,128,256,512",
                   type=lambda s: parse_list(s, int))
    p.add_argument("--utd", default="1.0,0.25,0.0625",
                   type=lambda s: parse_list(s, float))
    p.add_argument("--rollout_len", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_size", type=int, default=300_000,
                   help="lower this first if the sweep reports OOM")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--target_steps", type=int, default=1_000_000,
                   help="budget used for the wall-clock estimate column")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--learning_check", action="store_true",
                   help="also run short real training jobs per utd (slow)")
    p.add_argument("--check_steps", type=int, default=60_000)
    p.add_argument("--check_num_envs", type=int, default=128)

    p.add_argument("--out", default="benchmarks")
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out = str(Path(args.out) / stamp)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    dev = device_report()
    print()
    print(f"device       : {dev['device']}  ({dev.get('device_kind', '?')})")
    if "memory_limit_gb" in dev:
        print(f"device memory: {dev['memory_limit_gb']} GB")
    if dev["platform"] == "cpu":
        print()
        print("  WARNING: JAX is using the CPU. If this laptop has a CUDA GPU,")
        print("  install the GPU build first or these numbers are meaningless:")
        print('      pip install -U "jax[cuda13]"')
    print(f"buffer       : {args.buffer_size:,} transitions")
    print(f"batch        : {args.batch_size}")
    print()

    results = phase1(args)
    learning_rows = phase2(args, args.utd) if args.learning_check else []

    print()
    print("=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    for line in recommend(results, learning_rows, args.target_steps):
        print("  " + line)
    print()

    write_json(Path(args.out) / "benchmark.json", {
        "created": datetime.now().isoformat(timespec="seconds"),
        "device": dev,
        "args": vars(args),
        "throughput": results,
        "learning_check": learning_rows,
    })
    print(f"wrote {Path(args.out) / 'benchmark.json'}")


if __name__ == "__main__":
    main()
