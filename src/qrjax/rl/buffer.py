"""Replay buffer held entirely in device memory.

A Python-side buffer would force a host round trip every iteration and destroy
the point of a jitted training loop. Instead the buffer is a pytree of fixed
arrays that lives on the accelerator and is updated functionally, so the whole
collect-and-learn cycle stays inside one compiled graph.

Insertion is a batched scatter: a rollout of shape (T, N, ...) is flattened to
T*N transitions and written at wrapped indices in a single ``.at[].set()``.

Sizing note: capacity is in TRANSITIONS, and with N parallel envs a single
iteration contributes ``rollout_len * num_envs`` of them. A 1e6-capacity buffer
of 156-D float32 observations is roughly 1.3 GB with both obs and next_obs
stored, so on an 8 GB laptop GPU keep capacity at or below a few hundred
thousand unless observations are small.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Buffer(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray
    ptr: jnp.ndarray       # next write index
    size: jnp.ndarray      # number of valid entries


def init(capacity, obs_dim, action_dim) -> Buffer:
    return Buffer(
        obs=jnp.zeros((capacity, obs_dim), jnp.float32),
        action=jnp.zeros((capacity, action_dim), jnp.float32),
        reward=jnp.zeros((capacity,), jnp.float32),
        next_obs=jnp.zeros((capacity, obs_dim), jnp.float32),
        done=jnp.zeros((capacity,), jnp.float32),
        ptr=jnp.int32(0),
        size=jnp.int32(0),
    )


def add_batch(buf: Buffer, obs, action, reward, next_obs, done) -> Buffer:
    """Insert a flattened batch of transitions, wrapping at capacity."""
    capacity = buf.obs.shape[0]
    n = obs.shape[0]
    if n > capacity:
        raise ValueError(
            f"inserting {n} transitions into a buffer of capacity {capacity}: "
            f"the wrapped indices would collide and silently drop data. "
            f"Raise buffer_size above rollout_len * num_envs."
        )
    idx = (buf.ptr + jnp.arange(n)) % capacity
    return buf._replace(
        obs=buf.obs.at[idx].set(obs),
        action=buf.action.at[idx].set(action),
        reward=buf.reward.at[idx].set(reward),
        next_obs=buf.next_obs.at[idx].set(next_obs),
        done=buf.done.at[idx].set(done.astype(jnp.float32)),
        ptr=(buf.ptr + n) % capacity,
        size=jnp.minimum(buf.size + n, capacity),
    )


def sample(buf: Buffer, key, batch_size):
    """Uniform sample from the valid region.

    ``maxval=buf.size`` keeps the sampler off never-written slots while the
    buffer is still filling, without needing a Python-side branch.
    """
    idx = jax.random.randint(key, (batch_size,), 0, jnp.maximum(buf.size, 1))
    return (buf.obs[idx], buf.action[idx], buf.reward[idx],
            buf.next_obs[idx], buf.done[idx])
