"""Evaluation on a bank of exactly N episodes in which NO arm terminates.

Ordinary evaluation draws N disturbance realisations and reports over all of
them, charging terminated episodes at the termination threshold. That mixes
tracking quality with failure rate in one number and leaves a ragged episode
count once failures are excluded.

This module instead draws candidate realisations in chunks, runs every arm on
each candidate, and keeps only those where no arm terminated, until exactly N
survivors are collected. Every arm therefore sees the same N episodes and every
comparison stays paired.

What this costs, stated plainly: the resulting RMSE is CONDITIONAL ON SURVIVAL.
Any robustness difference between arms disappears from the table, because the
episodes that would have exposed it are the ones removed. The screening
statistics -- how many candidates were drawn and which arm rejected them -- are
therefore recorded alongside and must be reported with the results, or the
table silently flatters whichever arm fails most.

The rejection test is applied across ALL arms jointly rather than per arm. Per
arm screening would give each arm a different episode set and destroy the
pairing, which is the property the whole evaluation rests on.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from qrjax.envs import EnvConfig, VecEnv


def _rollout_chunk(env_cfg, controller, agent, params, key, episodes, ranges,
                   thrust_beta):
    """Run one arm over one chunk of candidate episodes."""
    cfg = EnvConfig(**{**env_cfg.__dict__})
    cfg.base_controller = controller
    cfg.thrust_filter_beta = thrust_beta
    env = VecEnv(cfg, episodes)
    batched = env.broadcast_ranges(ranges)

    def run(key):
        k_reset, _ = jax.random.split(key)
        state, obs = env.reset(k_reset, batched, stagger=False)
        draw = {"k": state.k,
                "force": state.external_force,
                "kf": state.mixer_true.kf_scale[:, 0],
                "moment": state.mixer_true.moment_scale[:, 0],
                "arm": state.mixer_true.arm_scale[:, 0]}

        def body(carry, _):
            state, obs, alive = carry
            if agent is None:
                action = jnp.zeros((episodes, env.action_dim))
            else:
                action = jnp.tanh(agent.actor.apply(params, obs)[0])
            state, obs, reward, done, info = env.step(state, action, batched)
            m = alive.astype(jnp.float32)
            alive_next = jnp.logical_and(alive,
                                         jnp.logical_not(info["terminated"]))
            return (state, obs, alive_next), (
                jnp.where(alive, info["pos_err_desired"], cfg.term_pos_error),
                jnp.where(alive, info["pos_err_twin"], cfg.term_pos_error),
                info["motor_cmd"], info["u_total"], action, m,
                info["saturated"] * m, info["x_true"], info["x_nom"],
                info["x_des"])

        alive0 = jnp.ones(episodes, dtype=bool)
        (_, _, alive), out = jax.lax.scan(
            body, (state, obs, alive0), None, length=cfg.episode_steps)
        return out, draw, alive

    out, draw, alive = jax.jit(run)(key)
    return ([np.asarray(x) for x in out],
            {k: np.asarray(v) for k, v in draw.items()},
            np.asarray(alive))


def build_bank(env_cfg, arms, ranges, key, n_target=256, chunk=128,
               thrust_beta_map=None, max_chunks=40, verbose=True):
    """Collect exactly `n_target` episodes in which no arm terminates.

    arms: list of (name, controller, agent, params) — agent None for a bare
          baseline.
    thrust_beta_map: name -> thrust filter beta for that arm.

    Returns (per_arm_traces, draw, stats). Traces are truncated to n_target
    episodes and are index-aligned across arms.
    """
    thrust_beta_map = thrust_beta_map or {}
    keep = {name: None for name, *_ in arms}
    keep_draw = None
    n_kept = 0
    n_drawn = 0
    n_survived = 0
    rejected_by = {name: 0 for name, *_ in arms}

    for c in range(max_chunks):
        ck = jax.random.fold_in(key, c)
        chunk_out, chunk_alive = {}, {}
        draw = None
        for name, controller, agent, params in arms:
            out, d, alive = _rollout_chunk(
                env_cfg, controller, agent, params, ck, chunk, ranges,
                thrust_beta_map.get(name, env_cfg.thrust_filter_beta))
            chunk_out[name] = out
            chunk_alive[name] = alive
            if draw is None:
                draw = d

        survive = np.ones(chunk, dtype=bool)
        for name in chunk_alive:
            survive &= chunk_alive[name]
        for name, a in chunk_alive.items():
            rejected_by[name] += int(np.sum(~a))
        n_drawn += chunk
        n_survived += int(survive.sum())

        idx = np.where(survive)[0]
        need = n_target - n_kept
        take = idx[:need]
        if take.size:
            for name in keep:
                sel = [x[:, take] if x.ndim >= 2 else x[take]
                       for x in chunk_out[name]]
                if keep[name] is None:
                    keep[name] = sel
                else:
                    keep[name] = [np.concatenate([a, b], axis=1)
                                  if a.ndim >= 2 else np.concatenate([a, b])
                                  for a, b in zip(keep[name], sel)]
            d = {k: v[take] for k, v in draw.items()}
            keep_draw = d if keep_draw is None else {
                k: np.concatenate([keep_draw[k], d[k]]) for k in d}
            n_kept += take.size

        if verbose:
            print(f"    chunk {c + 1}: drew {chunk}, survived "
                  f"{int(survive.sum())}, kept {n_kept}/{n_target}")
        if n_kept >= n_target:
            break

    if n_kept < n_target:
        raise SystemExit(
            f"only {n_kept} of {n_target} non-terminated episodes after "
            f"{n_drawn} draws. Either the termination rate is very high or "
            f"--max_chunks is too small.")

    # rejection rate counts only episodes an arm actually terminated; survivors
    # left over in the final chunk are unused, not rejected
    stats = {"candidates_drawn": n_drawn,
             "candidates_survived": n_survived,
             "kept": n_kept,
             "rejection_rate_pct": 100.0 * (1 - n_survived / n_drawn),
             "terminations_by_arm": {k: int(v) for k, v in rejected_by.items()},
             "termination_rate_by_arm_pct": {
                 k: 100.0 * v / n_drawn for k, v in rejected_by.items()}}
    return keep, keep_draw, stats
