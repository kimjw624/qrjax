"""Actor and critic networks (Flax linen).

Actor is a tanh-squashed diagonal Gaussian. The squash correction uses the
numerically stable form

    log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))

rather than ``log(1 - tanh(u)**2 + eps)``, which loses all precision once
|u| exceeds about 8 and silently biases the entropy term.

Critic is a twin Q network with optional LayerNorm on the hidden layers, which
markedly stabilizes off-policy value learning when observations are stacked
histories.
"""

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):
    action_dim: int
    hidden: Sequence[int] = (256, 256)
    log_std_min: float = LOG_STD_MIN
    log_std_max: float = LOG_STD_MAX

    @nn.compact
    def __call__(self, obs):
        x = obs
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        # Zero-init the mean head so the initial residual is ~0 and the policy
        # starts as the untouched base controller rather than a random wrench.
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.zeros,
                        bias_init=nn.initializers.zeros)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, params, obs, key):
        """Return (action, log_prob, deterministic_action)."""
        mean, log_std = self.apply(params, obs)
        std = jnp.exp(log_std)
        noise = jax.random.normal(key, mean.shape)
        u = mean + std * noise
        action = jnp.tanh(u)

        # Gaussian log-prob minus the tanh Jacobian correction.
        log_prob = (-0.5 * ((noise ** 2) + 2.0 * log_std + jnp.log(2.0 * jnp.pi)))
        correction = 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))
        log_prob = (log_prob - correction).sum(axis=-1)
        return action, log_prob, jnp.tanh(mean)


class Critic(nn.Module):
    hidden: Sequence[int] = (256, 256)
    layernorm: bool = True

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)

        def q_net(y):
            for h in self.hidden:
                y = nn.Dense(h)(y)
                if self.layernorm:
                    y = nn.LayerNorm()(y)
                y = nn.relu(y)
            return nn.Dense(1)(y).squeeze(-1)

        return q_net(x), q_net(x)
