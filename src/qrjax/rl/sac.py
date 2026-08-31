"""Soft Actor-Critic, fully jitted.

The whole update -- critic step, delayed actor step, entropy-temperature step,
and Polyak target sync -- is one compiled function of an ``AgentState`` pytree.
That is where the wall-clock win comes from: in the sequential PyTorch
reference the update is ~72% of runtime and is dominated by Python dispatch for
a small MLP, not by arithmetic.

Delayed actor updates (``policy_frequency``) use ``lax.cond``, not a Python
branch, because the update count is traced. ``lax.cond`` genuinely skips the
gradient computation on off-steps; ``jnp.where`` on the resulting parameters
would compute it and discard it, which measured ~40% slower.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .networks import Actor, Critic
from . import buffer as buffer_mod


class AgentState(NamedTuple):
    actor_params: dict
    critic_params: dict
    critic_target: dict
    log_alpha: jnp.ndarray
    actor_opt: optax.OptState
    critic_opt: optax.OptState
    alpha_opt: optax.OptState
    updates: jnp.ndarray
    key: jnp.ndarray


class SAC:
    """Container for the networks, optimizers, and jitted update."""

    def __init__(self, obs_dim, action_dim, cfg):
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.actor = Actor(action_dim=action_dim, hidden=tuple(cfg.hidden),
                           log_std_min=cfg.log_std_min, log_std_max=cfg.log_std_max)
        self.critic = Critic(hidden=tuple(cfg.hidden), layernorm=cfg.critic_layernorm)

        self.actor_tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adam(cfg.lr_actor),
        )
        self.critic_tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adam(cfg.lr_critic),
        )
        self.alpha_tx = optax.adam(cfg.lr_alpha)
        self.target_entropy = -cfg.target_entropy_scale * action_dim

    # ------------------------------------------------------------------ init
    def init(self, key) -> AgentState:
        k_actor, k_critic, k_state = jax.random.split(key, 3)
        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_act = jnp.zeros((1, self.action_dim))

        actor_params = self.actor.init(k_actor, dummy_obs)
        critic_params = self.critic.init(k_critic, dummy_obs, dummy_act)
        log_alpha = jnp.zeros(())

        return AgentState(
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target=critic_params,
            log_alpha=log_alpha,
            actor_opt=self.actor_tx.init(actor_params),
            critic_opt=self.critic_tx.init(critic_params),
            alpha_opt=self.alpha_tx.init(log_alpha),
            updates=jnp.int32(0),
            key=k_state,
        )

    # ------------------------------------------------------------------- act
    def act(self, state: AgentState, obs, key, deterministic=False):
        action, _, mean = self.actor.sample(state.actor_params, obs, key)
        return jnp.where(deterministic, mean, action)

    # ---------------------------------------------------------------- update
    def update(self, state: AgentState, buf):
        """One gradient step on a freshly sampled batch. Returns (state, metrics)."""
        cfg = self.cfg
        key, k_sample, k_next, k_pi = jax.random.split(state.key, 4)
        obs, action, reward, next_obs, done = buffer_mod.sample(
            buf, k_sample, cfg.batch_size
        )
        alpha = jnp.exp(state.log_alpha)

        # --- critic target ---
        next_action, next_logp, _ = self.actor.sample(
            state.actor_params, next_obs, k_next
        )
        q1_t, q2_t = self.critic.apply(state.critic_target, next_obs, next_action)
        target_q = jnp.minimum(q1_t, q2_t) - alpha * next_logp
        backup = reward + cfg.gamma * (1.0 - done) * target_q
        backup = jax.lax.stop_gradient(backup)

        def critic_loss_fn(params):
            q1, q2 = self.critic.apply(params, obs, action)
            loss = ((q1 - backup) ** 2).mean() + ((q2 - backup) ** 2).mean()
            return loss, (q1.mean(), q2.mean())

        (critic_loss, (q1m, q2m)), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(state.critic_params)
        critic_updates, critic_opt = self.critic_tx.update(
            critic_grads, state.critic_opt, state.critic_params
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        # --- actor (delayed) ---
        def actor_loss_fn(params):
            a, logp, _ = self.actor.sample(params, obs, k_pi)
            q1, q2 = self.critic.apply(critic_params, obs, a)
            return (alpha * logp - jnp.minimum(q1, q2)).mean(), logp

        def do_actor_step(_):
            (loss, lp), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                state.actor_params
            )
            upd, opt = self.actor_tx.update(grads, state.actor_opt, state.actor_params)
            return optax.apply_updates(state.actor_params, upd), opt, loss, lp

        def skip_actor_step(_):
            # Still need log-probs for the temperature loss, but no gradient.
            _, lp, _ = self.actor.sample(state.actor_params, obs, k_pi)
            return (state.actor_params, state.actor_opt,
                    jnp.zeros(()), lp)

        # lax.cond genuinely skips the actor gradient on off-steps. jnp.where
        # would compute it and throw it away, wasting ~40% of the update.
        actor_params, actor_opt, actor_loss, logp = jax.lax.cond(
            (state.updates % cfg.policy_frequency) == 0,
            do_actor_step, skip_actor_step, operand=None,
        )

        # --- entropy temperature ---
        def alpha_loss_fn(log_alpha):
            return (-jnp.exp(log_alpha)
                    * (jax.lax.stop_gradient(logp) + self.target_entropy)).mean()

        alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(state.log_alpha)
        alpha_updates, alpha_opt = self.alpha_tx.update(alpha_grad, state.alpha_opt)
        log_alpha = optax.apply_updates(state.log_alpha, alpha_updates)

        # --- Polyak target ---
        do_target = (state.updates % cfg.target_frequency) == 0
        critic_target = jax.tree.map(
            lambda t, s: jnp.where(do_target, (1 - cfg.tau) * t + cfg.tau * s, t),
            state.critic_target, critic_params,
        )

        new_state = AgentState(
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target=critic_target,
            log_alpha=log_alpha,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            alpha_opt=alpha_opt,
            updates=state.updates + 1,
            key=key,
        )
        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": alpha,
            "q1_mean": q1m,
            "q2_mean": q2m,
            "entropy": -logp.mean(),
        }
        return new_state, metrics
