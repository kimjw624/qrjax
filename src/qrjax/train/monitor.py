"""Live training curves.

Three surfaces, in order of robustness:

1. ``metrics.jsonl`` -- appended every log interval, always. Nothing can break
   this, and it survives a crash or a disconnected session.
2. ``progress.png`` -- redrawn every log interval, always. Open it in any image
   viewer; most viewers (eog, feh, VS Code) auto-refresh when the file changes,
   which gives a live curve with no GUI dependency at all.
3. An interactive matplotlib window -- only with ``--live_plot``, and only if a
   display is actually available.

The third is the one that breaks on headless machines, over SSH, or under
Wayland with a missing backend, so it is opt-in and degrades quietly: if the
window cannot be created, training continues and says so once, rather than
dying 40 minutes into a run.

Plotting happens on the host between compiled iterations, so it never appears
inside a traced graph and never blocks the accelerator.
"""

import os
import warnings
from pathlib import Path

import numpy as np

import matplotlib


def _display_available() -> bool:
    if os.environ.get("QRJAX_FORCE_HEADLESS"):
        return False
    if os.name == "nt" or sys.platform == "darwin":  # noqa: F821
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


import sys  # noqa: E402  (needed by _display_available)


PANELS = [
    ("eval_pos_rmse", "Eval position RMSE [m]", True),
    ("eval_term_frac", "Eval termination rate", False),
    ("episode_return", "Episode return", False),
    ("critic_loss", "Critic loss", True),
    ("alpha", "Entropy temperature", True),
    ("entropy", "Policy entropy", False),
]


class ProgressMonitor:
    """Accumulates metrics and redraws the training curve."""

    def __init__(self, run_dir, live=False, smooth=9):
        self.run_dir = Path(run_dir)
        self.png_path = self.run_dir / "progress.png"
        self.history = []
        self.smooth = int(smooth)
        self.live = False
        self._fig = None
        self._warned = False

        if live:
            if not _display_available():
                print("[monitor] no display detected; falling back to "
                      "progress.png only (it refreshes every log interval)")
            else:
                try:
                    matplotlib.use("TkAgg", force=True)
                    import matplotlib.pyplot as plt
                    plt.ion()
                    self._fig, self._axes = plt.subplots(2, 3, figsize=(15, 7))
                    self._fig.canvas.manager.set_window_title("qrjax training")
                    self.live = True
                except Exception as exc:
                    print(f"[monitor] live window unavailable ({type(exc).__name__}: "
                          f"{exc}); using progress.png only")

        if not self.live:
            matplotlib.use("Agg", force=True)

    def append(self, record: dict):
        self.history.append(dict(record))

    def _series(self, key):
        xs, ys = [], []
        for r in self.history:
            v = r.get(key)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            xs.append(r.get("env_steps", len(xs)))
            ys.append(float(v))
        return np.asarray(xs), np.asarray(ys)

    def _smoothed(self, y):
        if self.smooth <= 1 or y.size < self.smooth:
            return y
        kernel = np.ones(self.smooth) / self.smooth
        return np.convolve(y, kernel, mode="valid")

    def draw(self):
        """Redraw all panels. Cheap relative to an iteration, so called often."""
        import matplotlib.pyplot as plt

        if self.live:
            fig, axes = self._fig, self._axes
            for ax in axes.flat:
                ax.clear()
        else:
            fig, axes = plt.subplots(2, 3, figsize=(15, 7))

        for ax, (key, label, logy) in zip(axes.flat, PANELS):
            x, y = self._series(key)
            if y.size == 0:
                ax.set_title(f"{label} (no data)")
                ax.grid(alpha=0.3)
                continue
            ax.plot(x, y, lw=0.8, alpha=0.35, color="tab:blue")
            ys = self._smoothed(y)
            if ys.size:
                ax.plot(x[-ys.size:], ys, lw=1.8, color="tab:blue")
            if logy and np.all(y > 0):
                ax.set_yscale("log")
            ax.set_title(label)
            ax.set_xlabel("env steps")
            ax.grid(alpha=0.3)

        last = self.history[-1] if self.history else {}
        fig.suptitle(
            f"{self.run_dir.name}  |  step {last.get('env_steps', 0):,}"
            f"  |  eval RMSE {last.get('eval_pos_rmse', float('nan')):.4f} m",
            fontsize=11,
        )
        fig.tight_layout()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(self.png_path, dpi=110)

        if self.live:
            try:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except Exception:
                if not self._warned:
                    print("[monitor] live window closed; continuing with progress.png")
                    self._warned = True
                self.live = False
        else:
            plt.close(fig)

    def close(self):
        if self.live:
            try:
                import matplotlib.pyplot as plt
                plt.ioff()
                plt.close(self._fig)
            except Exception:
                pass
