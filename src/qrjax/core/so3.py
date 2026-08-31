"""SO(3) helpers in JAX.

Every function here is traceable: no Python control flow depends on a traced
value, so all of it survives jit / vmap / scan.

The subtlety is NaN safety. Several formulas below have a removable
singularity (theta -> 0 in the log map, ||v|| -> 0 in normalize). A single
``jnp.where`` is not enough, because JAX evaluates BOTH branches: if the
unsafe branch produces a NaN or an inf, ``where`` will happily select the good
value but the NaN can still poison gradients, and 0/0 in the unselected branch
shows up as NaN in the primal on some backends. The fix used throughout is the
"double where" pattern -- sanitize the input to the unsafe expression so it can
never produce a NaN, then select.
"""

import jax.numpy as jnp

E3 = jnp.array([0.0, 0.0, 1.0])


def hat(w):
    """Map a 3-vector to its skew-symmetric matrix, so hat(w) @ v == cross(w, v)."""
    return jnp.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


def vee(S):
    """Inverse of :func:`hat`. Matches the NumPy reference implementation."""
    return jnp.array([-S[1, 2], S[0, 2], -S[0, 1]])


def project_to_so3(R):
    """Orthonormalize onto SO(3) via SVD, guaranteeing det = +1.

    The reference implementation branches on ``det < 0`` and flips a column of
    U. Here the flip is folded into a multiply so the function stays
    branch-free.
    """
    U, _, Vt = jnp.linalg.svd(R)
    d = jnp.linalg.det(U @ Vt)
    U = U.at[:, -1].multiply(jnp.where(d < 0.0, -1.0, 1.0))
    return U @ Vt


def rotation_error(R, Rd):
    """Geometric attitude error e_R = 0.5 * vee(Rd^T R - R^T Rd)."""
    return 0.5 * vee(Rd.T @ R - R.T @ Rd)


def so3_log(R):
    """Matrix logarithm of R as a skew-symmetric matrix.

    Safe as theta -> 0, where theta / (2 sin theta) is 0/0.
    """
    cos_theta = jnp.clip((jnp.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    small = theta < 1e-6
    safe_sin = jnp.where(small, 1.0, jnp.sin(theta))
    coef = jnp.where(small, 0.5, theta / (2.0 * safe_sin))
    return coef * (R - R.T)


def so3_log_vector(R):
    """Rotation vector (axis * angle) whose exponential is R.

    Handles both theta -> 0 and theta -> pi. The near-pi branch cannot use the
    sin formula at all, so it recovers the axis from the diagonal of
    0.5 (R + I) and fixes signs from the off-diagonal terms.
    """
    cos_theta = jnp.clip((jnp.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)

    small = theta < 1e-7
    near_pi = (jnp.pi - theta) < 1e-5

    safe_sin = jnp.where(small | near_pi, 1.0, jnp.sin(theta))
    generic = theta * vee(R - R.T) / (2.0 * safe_sin)
    tiny = 0.5 * vee(R - R.T)

    A = 0.5 * (R + jnp.eye(3))
    axis = jnp.sqrt(jnp.maximum(jnp.diag(A), 0.0))
    axis = jnp.array([
        jnp.copysign(axis[0], R[2, 1] - R[1, 2]),
        jnp.copysign(axis[1], R[0, 2] - R[2, 0]),
        jnp.copysign(axis[2], R[1, 0] - R[0, 1]),
    ])
    axis_n = jnp.linalg.norm(axis)
    safe_axis_n = jnp.where(axis_n < 1e-8, 1.0, axis_n)
    pi_branch = jnp.where(axis_n < 1e-8, jnp.zeros(3), theta * axis / safe_axis_n)

    return jnp.where(small, tiny, jnp.where(near_pi, pi_branch, generic))


def normalize(v, fallback, eps=1e-9):
    """Normalize, returning ``fallback`` for a near-zero vector.

    The sanitization has to happen BEFORE the norm, not just before the
    division. ``norm(v)`` at v = 0 has derivative v/||v|| = 0/0, so guarding
    only the division still yields a NaN gradient at the origin -- the norm
    itself is already poisoned by then. Substituting a safe input up front
    fixes both the primal and the gradient.
    """
    n2 = jnp.dot(v, v)
    ok = n2 > eps ** 2
    safe_v = jnp.where(ok, v, jnp.ones_like(v))
    return jnp.where(ok, safe_v / jnp.linalg.norm(safe_v), fallback)
