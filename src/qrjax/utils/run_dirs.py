"""Run directory layout and serialization.

Every training run and every evaluation gets its own directory with an
auto-incrementing trial index, so re-running the same name never overwrites
earlier results:

    runs/<run_name>/trial_003/
        config.json          full resolved config, exactly as used
        manifest.json        git commit, command line, device, timestamps
        curriculum.toml      copy of the curriculum actually used (if any)
        metrics.jsonl        one JSON object per logged iteration
        progress.png         training curve, rewritten every log interval
        checkpoints/
            best.pt          highest eval score so far
            last.pt          most recent
            step_000250000.pt
        evaluation/
            <eval_name>/     written by scripts/evaluate.py

Checkpoints use ``.pt`` only for familiarity; they are msgpack-serialized Flax
parameter trees, not torch files.
"""

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def git_commit(default="unknown"):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return default


def jsonify(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): jsonify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonify(x) for x in v]
    try:
        import jax.numpy as jnp
        if isinstance(v, jnp.ndarray):
            return np.asarray(v).tolist()
    except Exception:
        pass
    return v


def write_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonify(obj), indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonify(obj)) + "\n")


def read_jsonl(path: Path):
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A partially-flushed final line while training is live.
                continue
    return out


def next_trial_dir(runs_root, run_name, create=True) -> Path:
    """Return ``<runs_root>/<run_name>/trial_NNN`` with the next free index."""
    base = Path(runs_root) / run_name
    base.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("trial_")]
    used = set()
    for name in existing:
        try:
            used.add(int(name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    idx = 1
    while idx in used:
        idx += 1
    run_dir = base / f"trial_{idx:03d}"
    if create:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(run_dir: Path, extra=None):
    import jax
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "hostname": platform.node(),
    }
    if extra:
        manifest.update(extra)
    write_json(Path(run_dir) / "manifest.json", manifest)
    return manifest


def save_params(path: Path, pytree, meta=None):
    """Serialize a Flax parameter tree with msgpack, plus a JSON sidecar."""
    from flax.serialization import to_bytes
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_bytes(pytree))
    if meta is not None:
        write_json(path.with_suffix(".meta.json"), meta)


def load_params(path: Path, target):
    """Restore into ``target``, which supplies the tree structure and shapes."""
    from flax.serialization import from_bytes
    return from_bytes(target, Path(path).read_bytes())
