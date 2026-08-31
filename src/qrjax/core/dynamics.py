"""Quadrotor rigid-body dynamics in NED coordinates, as pure JAX functions.

    x_dot     = v
    v_dot     = g e3 - (f / m) R e3 + f_ext / m
    R_dot     = R hat(omega)
    omega_dot = J^-1 (M - omega x J omega)

Fixed-step RK4, matching the NumPy reference. The rigid-body state is a
NamedTuple so it is a pytree and can be carried through scan without manual
flattening.
"""

from typing import NamedTuple

import jax.numpy as jnp

from .so3 import E3, hat, project_to_so3


class RigidState(NamedTuple):
    """Position, velocity, attitude, body angular velocity."""
    x: jnp.ndarray      # (3,)
    v: jnp.ndarray      # (3,)
    R: jnp.ndarray      # (3, 3)
    omega: jnp.ndarray  # (3,)


class Inertia(NamedTuple):
    """Plant parameters. Held per-env so each parallel env can be disturbed."""
    mass: jnp.ndarray   # scalar
    J: jnp.ndarray      # (3, 3)


def initial_state(x, v):
    return RigidState(x=x, v=v, R=jnp.eye(3), omega=jnp.zeros(3))


def _pack(s: RigidState):
    return jnp.concatenate([s.x, s.v, s.R.reshape(9), s.omega])


def _unpack(y):
    return RigidState(x=y[0:3], v=y[3:6], R=y[6:15].reshape(3, 3), omega=y[15:18])


def _deriv(y, f, M, inertia: Inertia, gravity, fext):
    s = _unpack(y)
    v_dot = gravity * E3 - (f / inertia.mass) * (s.R @ E3) + fext / inertia.mass
    R_dot = s.R @ hat(s.omega)
    omega_dot = jnp.linalg.solve(inertia.J, M - jnp.cross(s.omega, inertia.J @ s.omega))
    return jnp.concatenate([s.v, v_dot, R_dot.reshape(9), omega_dot])


def step(state: RigidState, f, M, inertia: Inertia, gravity, fext, dt) -> RigidState:
    """One fixed-step RK4 update, with the attitude re-projected onto SO(3)."""
    y = _pack(state)
    k1 = _deriv(y, f, M, inertia, gravity, fext)
    k2 = _deriv(y + 0.5 * dt * k1, f, M, inertia, gravity, fext)
    k3 = _deriv(y + 0.5 * dt * k2, f, M, inertia, gravity, fext)
    k4 = _deriv(y + dt * k3, f, M, inertia, gravity, fext)
    yn = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    out = _unpack(yn)
    return out._replace(R=project_to_so3(out.R))


def accel_at(state: RigidState, f, M, inertia: Inertia, gravity, fext):
    """Instantaneous (v_dot, omega_dot) without advancing the state.

    Used to build residual-acceleration targets: evaluate once with the true
    inertia and active disturbance, once with the nominal inertia and no
    disturbance, at the same state, and subtract.
    """
    v_dot = gravity * E3 - (f / inertia.mass) * (state.R @ E3) + fext / inertia.mass
    omega_dot = jnp.linalg.solve(
        inertia.J, M - jnp.cross(state.omega, inertia.J @ state.omega)
    )
    return v_dot, omega_dot
