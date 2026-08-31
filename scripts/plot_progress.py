"""Plot training curves from a run's metrics.jsonl.

Useful for three things training itself does not cover: watching a run started
in another terminal, re-plotting after a run finishes, and overlaying several
runs to compare curriculum against flat training.

Examples
--------
# Watch a run that is currently training, refreshing every 5 s.
python -m scripts.plot_progress --run_dir runs/residual_sac_flat_pd/trial_001 --watch

# Overlay two finished runs.
python -m scripts.plot_progress \
    --run_dir runs/residual_sac_curriculum_pd/trial_001 \
              runs/residual_sac_flat_pd/trial_001 \
    --out comparison.png
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib                                    # noqa: E402
import numpy as np                                   # noqa: E402

from qrjax.utils import read_jsonl                   # noqa: E402

PANELS = [
    ("eval_pos_rmse", "Eval position RMSE [m]", True),
    ("episode_return", "Episode return", False),
    ("critic_loss", "Critic loss", True),
    ("alpha", "Entropy temperature", True),
    ("entropy", "Policy entropy", False),
    ("sps", "Env steps / s", False),
]


def series(records, key):
    xs, ys = [], []
    for r in records:
        v = r.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        xs.append(r.get("env_steps", len(xs)))
        ys.append(v)
    return np.asarray(xs), np.asarray(ys)


def smooth(y, w):
    if w <= 1 or y.size < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def draw(run_dirs, out_path, window, fig=None, axes=None):
    import matplotlib.pyplot as plt

    if fig is None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    else:
        for ax in axes.flat:
            ax.clear()

    for run_dir in run_dirs:
        records = read_jsonl(Path(run_dir) / "metrics.jsonl")
        if not records:
            continue
        label = Path(run_dir).parent.name + "/" + Path(run_dir).name
        for ax, (key, title, logy) in zip(axes.flat, PANELS):
            x, y = series(records, key)
            if y.size == 0:
                continue
            line, = ax.plot(x, y, lw=0.7, alpha=0.3)
            ys = smooth(y, window)
            if ys.size:
                ax.plot(x[-ys.size:], ys, lw=1.8, color=line.get_color(), label=label)
            if logy and np.all(y > 0):
                ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("env steps")
            ax.grid(alpha=0.3)

    if len(run_dirs) > 1:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center",
                       ncol=min(len(labels), 4), fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=130)
    return fig, axes


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", nargs="+", required=True)
    p.add_argument("--out", default=None,
                   help="output PNG; default <run_dir>/progress_replot.png "
                        "for a single run")
    p.add_argument("--window", type=int, default=9, help="smoothing window")
    p.add_argument("--watch", action="store_true",
                   help="re-read and redraw until interrupted")
    p.add_argument("--interval", type=float, default=5.0)
    args = p.parse_args()

    out = args.out
    if out is None and len(args.run_dir) == 1:
        out = str(Path(args.run_dir[0]) / "progress_replot.png")

    if not args.watch:
        matplotlib.use("Agg")
        draw(args.run_dir, out, args.window)
        print(f"wrote {out}")
        return

    try:
        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt
        plt.ion()
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        interactive = True
    except Exception as exc:
        print(f"[plot] no interactive backend ({type(exc).__name__}); "
              f"rewriting {out} every {args.interval:g}s instead")
        matplotlib.use("Agg", force=True)
        fig = axes = None
        interactive = False

    print("watching; Ctrl-C to stop")
    try:
        while True:
            fig, axes = draw(args.run_dir, out, args.window, fig, axes)
            if interactive:
                import matplotlib.pyplot as plt
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(args.interval)
            else:
                import matplotlib.pyplot as plt
                plt.close(fig)
                fig = axes = None
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
