"""Fixed-step curriculum over disturbance distributions.

A stage is just a set of disturbance ranges plus a duration. Because
``DisturbRanges`` is a traced argument to the env rather than a static closure
value, advancing a stage costs nothing -- no recompile, no graph switch, no
pause in training.

Two modes:

*Curriculum* (``--curriculum``): stages run in order, and after the first stage
a fraction of episodes rehearse an earlier stage to limit forgetting. Rehearsal
is implemented per-env rather than per-episode: at each reset an env
independently draws which stage it belongs to, so a single batch of parallel
envs naturally contains a mixture.

*Flat* (``--no_curriculum``): every disturbance type is active over its full
range from the first step. Equivalent to running the final curriculum stage for
the whole budget.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from ..envs.config import EnvConfig
from ..envs.residual_env import DisturbRanges


ALL_DISTURBANCES = ("massmoi", "force", "motor_coeff", "moment_coeff", "arm_length")


@dataclass
class Stage:
    name: str
    timesteps: int
    disturbances: Tuple[str, ...]
    overrides: dict = field(default_factory=dict)

    def ranges(self, base: EnvConfig) -> DisturbRanges:
        cfg = EnvConfig(**{**base.__dict__})
        for k, v in self.overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.disturbances = tuple(self.disturbances)
        return DisturbRanges.from_config(cfg)


@dataclass
class Curriculum:
    stages: List[Stage]
    rehearsal_probability: float = 0.25

    @property
    def total_timesteps(self) -> int:
        return sum(s.timesteps for s in self.stages)

    def stage_index_at(self, step: int) -> int:
        """Scheduled stage for a given global env-step count."""
        acc = 0
        for i, s in enumerate(self.stages):
            acc += s.timesteps
            if step < acc:
                return i
        return len(self.stages) - 1

    def all_ranges(self, base: EnvConfig) -> List[DisturbRanges]:
        return [s.ranges(base) for s in self.stages]


def load_curriculum(path) -> Curriculum:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"curriculum file not found: {path}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    stages = []
    for entry in raw.get("stages", []):
        known = {"name", "timesteps", "disturbances"}
        overrides = {k: v for k, v in entry.items() if k not in known}
        stages.append(Stage(
            name=str(entry["name"]),
            timesteps=int(entry["timesteps"]),
            disturbances=tuple(entry.get("disturbances", ("massmoi",))),
            overrides=overrides,
        ))
    if not stages:
        raise ValueError(f"{path} defines no [[stages]]")

    return Curriculum(
        stages=stages,
        rehearsal_probability=float(raw.get("rehearsal_probability", 0.25)),
    )


def flat_ranges(base: EnvConfig, force_freq=None, force_dc_prob=None) -> DisturbRanges:
    """Every disturbance type active over its full configured range.

    This is what ``--no_curriculum`` trains on: no staging, no rehearsal, the
    hardest distribution from step zero.
    """
    cfg = EnvConfig(**{**base.__dict__})
    cfg.disturbances = ALL_DISTURBANCES
    return DisturbRanges.from_config(cfg, force_freq=force_freq,
                                     force_dc_prob=force_dc_prob)
