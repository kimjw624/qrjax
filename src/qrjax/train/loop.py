"""Training loop.

Shape of one iteration:

    collect  rollout_len steps across num_envs envs   (lax.scan, jitted)
    insert   rollout_len * num_envs transitions       (single scatter)
    learn    n_updates SAC gradient steps             (lax.scan, jitted)

All three are inside ONE compiled function, so a whole iteration is a single
dispatch. Metrics come back between iterations, which is where logging,
plotting, evaluation, and checkpointing happen on the host. That split is what
gives both a fully-fused inner loop and a live training curve -- a single
end-to-end scan over all of training would be marginally faster but would emit
nothing until it finished.

Update-to-data ratio. ``n_updates = round(utd * rollout_len * num_envs)``.
``utd = 1.0`` reproduces the sequential reference (one gradient step per env
step) and is the setting to use when comparing sample efficiency. It is also
usually the slowest useful setting: with many parallel envs the same number of
gradient steps sees far more diverse data, so ``utd`` well below 1 often
reaches the same performance in a fraction of the wall time.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..envs.residual_env import VecEnv, DisturbRanges
from ..rl import buffer as buffer_mod
from ..rl.sac import SAC
from ..rl.curriculum import load_curriculum, flat_ranges
from ..utils.run_dirs import (
    append_jsonl, next_trial_dir, save_params, write_json, write_manifest,
)
from .monitor import ProgressMonitor


def _stack_ranges(range_list):
    """Stack a list of scalar DisturbRanges into arrays indexed by stage."""
    return jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]),
                        *range_list)


def _assign_ranges(stacked, stage_idx):
    """Gather per-env ranges from the stacked table using per-env stage ids."""
    return jax.tree.map(lambda arr: arr[stage_idx], stacked)


def build_eval_fn(cfg, agent, num_episodes):
    """Deterministic evaluation: one full episode per env, greedy policy.

    Two things this has to get right that a naive scan does not.

    *Auto-reset contaminates the measurement.* The env resets itself on
    termination, so a plain scan over episode_steps silently splices a failed
    episode together with a fresh one that has a DIFFERENT disturbance draw,
    and reports the mixture as one episode. An ``alive`` mask freezes each env's
    contribution at its first termination.

    *Failures must not be rewarded.* Masking alone would give an env that
    diverges at step 5 an excellent RMSE over those 5 steps. Instead the error
    after termination is held at the termination threshold, which is what
    "this episode failed" actually means in position terms. The resulting mean
    is then monotone in both tracking quality and failure rate, so it can be
    used directly for checkpoint selection without inventing a weighting
    between the two.

    Median and p90 are reported alongside because the mean over episodes is
    dominated by the few that fail.
    """
    eval_env = VecEnv(cfg, num_episodes)

    def run(agent_state, key, ranges):
        k_reset, k_run = jax.random.split(key)
        # stagger=False: every eval episode starts at t=0 so arms are
        # directly comparable and the whole trajectory is always covered.
        state, obs = eval_env.reset(k_reset, ranges, stagger=False)

        def body(carry, _):
            state, obs, alive = carry
            action = jnp.tanh(agent.actor.apply(agent_state.actor_params, obs)[0])
            state, obs, reward, done, info = eval_env.step(state, action, ranges)
            err = jnp.where(alive, info["pos_err_desired"], cfg.term_pos_error)
            alive = jnp.logical_and(alive, jnp.logical_not(info["terminated"]))
            return (state, obs, alive), (err, reward * alive, info["terminated"])

        alive0 = jnp.ones(num_episodes, dtype=bool)
        (_, _, alive), (pos_err, reward, term) = jax.lax.scan(
            body, (state, obs, alive0), None, length=cfg.episode_steps
        )
        per_ep = jnp.sqrt(jnp.mean(pos_err ** 2, axis=0))
        return {
            "eval_pos_rmse": jnp.mean(per_ep),
            "eval_pos_rmse_median": jnp.median(per_ep),
            "eval_pos_rmse_p90": jnp.quantile(per_ep, 0.9),
            "eval_return": jnp.mean(jnp.sum(reward, axis=0)),
            "eval_term_frac": 1.0 - jnp.mean(alive.astype(jnp.float32)),
        }

    return jax.jit(run)


def build_iteration_fn(cfg, train_cfg, agent, env, n_updates):
    """Compile collect + insert + learn into one function."""

    def iteration(carry, ranges, learn: bool):
        env_state, obs, agent_state, buf, key = carry

        # ---------------- collect ----------------
        def collect_step(c, _):
            env_state, obs, agent_state, key = c
            key, k_act = jax.random.split(key)
            action = agent.act(agent_state, obs, k_act, deterministic=False)
            env_state, next_obs, reward, done, info = env.step(env_state, action, ranges)

            # Store info["final_obs"], not next_obs: on a done step next_obs is
            # already the reset observation of a NEW episode with a different
            # disturbance, while final_obs is the true successor state.
            #
            # Store info["terminated"], not done. An episode that merely ran out
            # of clock has not reached an absorbing state, so its value must be
            # bootstrapped normally. Masking it as terminal teaches the critic
            # that the world ends every 1000 steps, which with gamma=0.99 puts a
            # systematic downward bias on every value estimate near a boundary.
            transition = (obs, action, reward, info["final_obs"],
                          info["terminated"])
            stats = (info["episode_return"], info["episode_pos_mae"],
                     info["done"], info["terminated"])
            return (env_state, next_obs, agent_state, key), (transition, stats)

        (env_state, obs, agent_state, key), (transitions, stats) = jax.lax.scan(
            collect_step, (env_state, obs, agent_state, key), None,
            length=train_cfg.rollout_len,
        )

        flat = jax.tree.map(lambda a: a.reshape((-1,) + a.shape[2:]), transitions)
        buf = buffer_mod.add_batch(buf, *flat)

        # ---------------- learn ----------------
        def learn_step(agent_state, _):
            return agent.update(agent_state, buf)

        def do_learn(agent_state):
            agent_state, metrics = jax.lax.scan(
                learn_step, agent_state, None, length=n_updates
            )
            return agent_state, jax.tree.map(jnp.mean, metrics)

        def skip_learn(agent_state):
            zeros = {k: jnp.float32(jnp.nan) for k in
                     ("critic_loss", "actor_loss", "alpha_loss", "alpha",
                      "q1_mean", "q2_mean", "entropy")}
            return agent_state, zeros

        agent_state, learn_metrics = jax.lax.cond(
            learn, do_learn, skip_learn, agent_state
        )

        ep_return, ep_rmse, done, term = stats
        finished = done.astype(jnp.float32)
        n_finished = finished.sum()
        # Episode statistics are only defined once an episode actually ends.
        # Reporting 0.0 when none has would draw a flat zero line on the
        # training curve that looks like a real (and broken) measurement, so
        # emit NaN instead and let the monitor skip those points.
        safe_n = jnp.maximum(n_finished, 1.0)
        none_yet = n_finished < 0.5
        mean_or_nan = lambda v: jnp.where(none_yet, jnp.nan, (v * finished).sum() / safe_n)
        rollout_metrics = {
            "episode_return": mean_or_nan(ep_return),
            "episode_pos_mae": mean_or_nan(ep_rmse),
            "episodes_finished": n_finished,
            "terminated_frac": jnp.where(none_yet, jnp.nan,
                                         term.astype(jnp.float32).sum() / safe_n),
        }
        carry = (env_state, obs, agent_state, buf, key)
        return carry, {**learn_metrics, **rollout_metrics}

    return jax.jit(iteration, static_argnums=(2,))


def train(cfg_all, live_plot=False, verbose=True):
    """Run training end to end. Returns the run directory."""
    cfg, sac_cfg, train_cfg = cfg_all.env, cfg_all.sac, cfg_all.train

    run_dir = next_trial_dir(train_cfg.runs_root, train_cfg.run_name)
    write_json(run_dir / "config.json", cfg_all.to_dict())

    # ---------------- curriculum or flat ----------------
    if train_cfg.use_curriculum:
        curriculum = load_curriculum(train_cfg.curriculum_path)
        stage_ranges = [s.ranges(cfg) for s in curriculum.stages]
        stage_names = [s.name for s in curriculum.stages]
        total_steps = curriculum.total_timesteps
        import shutil
        shutil.copyfile(train_cfg.curriculum_path, run_dir / "curriculum.toml")
    else:
        curriculum = None
        stage_ranges = [flat_ranges(cfg)]
        stage_names = ["flat_all_disturbances"]
        total_steps = train_cfg.total_steps

    stacked = _stack_ranges(stage_ranges)

    # ---------------- build ----------------
    env = VecEnv(cfg, train_cfg.num_envs)
    agent = SAC(env.obs_dim, env.action_dim, sac_cfg)

    key = jax.random.PRNGKey(train_cfg.seed)
    key, k_agent, k_env = jax.random.split(key, 3)
    agent_state = agent.init(k_agent)
    buf = buffer_mod.init(sac_cfg.buffer_size, env.obs_dim, env.action_dim)

    init_ranges = _assign_ranges(stacked, jnp.zeros(train_cfg.num_envs, jnp.int32))
    env_state, obs = env.reset(k_env, init_ranges)

    steps_per_iter = train_cfg.rollout_len * train_cfg.num_envs
    n_updates = max(1, int(round(train_cfg.utd * steps_per_iter)))
    total_iters = max(1, total_steps // steps_per_iter)

    iteration_fn = build_iteration_fn(cfg, train_cfg, agent, env, n_updates)
    eval_fn = build_eval_fn(cfg, agent, train_cfg.eval_episodes)

    monitor = ProgressMonitor(run_dir, live=live_plot)
    write_manifest(run_dir, {
        "obs_dim": env.obs_dim, "action_dim": env.action_dim,
        "num_envs": train_cfg.num_envs, "rollout_len": train_cfg.rollout_len,
        "steps_per_iteration": steps_per_iter, "updates_per_iteration": n_updates,
        "total_iterations": total_iters, "total_env_steps": total_steps,
        "mode": "curriculum" if curriculum else "flat",
        "stages": stage_names,
    })

    if verbose:
        print(f"run dir     : {run_dir}")
        print(f"device      : {jax.devices()[0]}")
        print(f"mode        : {'curriculum (' + str(len(stage_names)) + ' stages)' if curriculum else 'flat, all disturbances'}")
        print(f"envs        : {train_cfg.num_envs} x {train_cfg.rollout_len} = "
              f"{steps_per_iter} steps/iter")
        print(f"updates     : {n_updates}/iter  (utd={train_cfg.utd})")
        print(f"budget      : {total_steps:,} env steps = {total_iters:,} iterations")
        print()

    rng = np.random.default_rng(train_cfg.seed)
    best_score = np.inf
    env_steps = 0
    t_start = time.time()

    for it in range(1, total_iters + 1):
        # --- per-env stage assignment (rehearsal mixes stages within a batch) ---
        if curriculum is not None:
            scheduled = curriculum.stage_index_at(env_steps)
            stage_idx = np.full(train_cfg.num_envs, scheduled, dtype=np.int32)
            if scheduled > 0 and curriculum.rehearsal_probability > 0:
                mask = rng.random(train_cfg.num_envs) < curriculum.rehearsal_probability
                stage_idx[mask] = rng.integers(0, scheduled, size=int(mask.sum()))
        else:
            scheduled = 0
            stage_idx = np.zeros(train_cfg.num_envs, dtype=np.int32)

        ranges = _assign_ranges(stacked, jnp.asarray(stage_idx))
        learn = env_steps >= sac_cfg.learning_starts

        carry = (env_state, obs, agent_state, buf, key)
        carry, metrics = iteration_fn(carry, ranges, bool(learn))
        env_state, obs, agent_state, buf, key = carry
        env_steps += steps_per_iter

        # --- evaluation ---
        eval_metrics = {}
        if it % train_cfg.eval_every_iters == 0 or it == total_iters:
            key, k_eval = jax.random.split(key)
            eval_ranges = jax.tree.map(
                lambda v: jnp.broadcast_to(v[scheduled], (train_cfg.eval_episodes,)),
                stacked,
            )
            eval_metrics = {k: float(v) for k, v in
                            eval_fn(agent_state, k_eval, eval_ranges).items()}

            score = eval_metrics["eval_pos_rmse"]
            if np.isfinite(score) and score < best_score:
                best_score = score
                save_params(run_dir / "checkpoints" / "best.pt",
                            agent_state.actor_params,
                            {"env_steps": env_steps, "iteration": it, **eval_metrics})

        # --- logging ---
        if it % train_cfg.log_every_iters == 0 or eval_metrics:
            elapsed = time.time() - t_start
            record = {
                "iteration": it,
                "env_steps": env_steps,
                "stage": stage_names[scheduled],
                "wallclock_s": round(elapsed, 2),
                "sps": round(env_steps / max(elapsed, 1e-9), 1),
                **{k: float(v) for k, v in metrics.items()},
                **eval_metrics,
            }
            append_jsonl(run_dir / "metrics.jsonl", record)
            monitor.append(record)
            monitor.draw()

            if verbose and (eval_metrics or it % (train_cfg.log_every_iters * 10) == 0):
                ret = record["episode_return"]
                ret_s = "   --  " if not np.isfinite(ret) else f"{ret:>7.2f}"
                msg = (f"it {it:>6}/{total_iters}  steps {env_steps:>10,}  "
                       f"{record['sps']:>8,.0f} sps  ret {ret_s}")
                if eval_metrics:
                    # Print the median as well as the mean. The mean folds in a
                    # 2.0 m charge per failed episode, so with 64 eval episodes
                    # a single extra failure shifts it by ~0.03 -- most of the
                    # visible bounce is that Bernoulli count, not tracking
                    # quality. The median is unaffected by a few failures and
                    # is the number to read for tracking; term% is the number
                    # to read for robustness.
                    msg += (f"  RMSE mean {eval_metrics['eval_pos_rmse']:.4f}"
                            f" med {eval_metrics['eval_pos_rmse_median']:.4f}"
                            f"  term {100*eval_metrics['eval_term_frac']:.1f}%")
                print(msg)

        # --- periodic checkpoint ---
        if it % train_cfg.checkpoint_every_iters == 0:
            save_params(run_dir / "checkpoints" / f"step_{env_steps:09d}.pt",
                        agent_state.actor_params,
                        {"env_steps": env_steps, "iteration": it})

    save_params(run_dir / "checkpoints" / "last.pt", agent_state.actor_params,
                {"env_steps": env_steps, "iteration": total_iters})
    monitor.draw()
    monitor.close()

    if verbose:
        dt = time.time() - t_start
        print(f"\ndone in {dt/60:.1f} min  ({env_steps/max(dt,1e-9):,.0f} env steps/s)")
        print(f"best eval position RMSE: {best_score:.4f} m")
        print(f"artifacts: {run_dir}")
    return run_dir
