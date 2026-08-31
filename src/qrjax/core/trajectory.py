"""Analytic desired trajectories in NED coordinates.

Returns (x, v, a, b1d) as a plain tuple so it is a pytree and can be produced
inside a scan. Trajectory type is a static Python value chosen at construction
time, not a traced one, so each variant compiles to straight-line code.
"""

import jax.numpy as jnp

B1D = jnp.array([1.0, 0.0, 0.0])   # fixed zero-yaw heading


def figure8(t, a=1.0, b=1.0, z_amp=1.0, omega=2.0 * jnp.pi / 10.0, z0=0.0):
    """3-D figure eight: x = a sin(wt), y = (b/2) sin(2wt), z = z0 + z_amp sin(wt)."""
    th = omega * t
    s1, c1 = jnp.sin(th), jnp.cos(th)
    s2, c2 = jnp.sin(2.0 * th), jnp.cos(2.0 * th)
    x = jnp.array([a * s1, 0.5 * b * s2, z0 + z_amp * s1])
    v = jnp.array([a * omega * c1, b * omega * c2, z_amp * omega * c1])
    acc = jnp.array([
        -a * omega ** 2 * s1,
        -2.0 * b * omega ** 2 * s2,
        -z_amp * omega ** 2 * s1,
    ])
    return x, v, acc, B1D


def circle(t, radius=0.79, speed=0.5, z0=0.0):
    """Circle starting at (0, 0, z0), moving North, centred at (0, radius, z0)."""
    w = speed / radius
    s, c = jnp.sin(w * t), jnp.cos(w * t)
    x = jnp.array([radius * s, -radius * c + radius, z0])
    v = jnp.array([radius * w * c, radius * w * s, 0.0])
    acc = jnp.array([-radius * w ** 2 * s, radius * w ** 2 * c, 0.0])
    return x, v, acc, B1D


def hover(t, z0=0.0):
    return jnp.array([0.0, 0.0, z0]), jnp.zeros(3), jnp.zeros(3), B1D


_TRAJECTORIES = {"figure8": figure8, "circle": circle, "hover": hover}


def make_trajectory(kind="figure8", **kwargs):
    """Return a closure ``t -> (x, v, a, b1d)``.

    ``kind`` is static: it selects the function at trace time, so no branch
    survives into the compiled graph.
    """
    kind = str(kind).lower()
    if kind not in _TRAJECTORIES:
        raise ValueError(
            f"unknown trajectory {kind!r}; expected one of {sorted(_TRAJECTORIES)}"
        )
    fn = _TRAJECTORIES[kind]
    return lambda t: fn(t, **kwargs)
