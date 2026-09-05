"""Vectorized sampling primitives used by the Monte Carlo tracer.

All functions here operate on batches of ``n`` photons at once (shape
``(n,)`` or ``(n, 3)``), are pure functions of explicit PRNG keys, and are
safe to call under ``jax.jit``/``lax.scan`` (no data-dependent Python control
flow; edge cases are handled with ``jnp.where``/clipping so no NaNs are ever
produced, which also keeps gradients well-behaved).
"""

import jax
import jax.numpy as jnp


def safe_mu_z(mu_z, floor=1e-6):
    """Clamp |mu_z| away from 0 (a photon moving exactly transverse to the
    line of sight would otherwise never change its LOS fraction ``x``, a
    measure-zero edge case that would divide by zero)."""
    sign = jnp.where(mu_z < 0, -1.0, 1.0)
    return jnp.where(jnp.abs(mu_z) < floor, sign * floor, mu_z)


def sample_next_event(u_tau, profile, tau_sca, x_current, mu_z):
    """Sample whether/where a photon's next real scattering event happens.

    A photon at fractional line-of-sight position ``x_current`` (0 = observer,
    1 = source), traveling with direction z-cosine ``mu_z``, accumulates
    optical depth ``d(tau) = tau_sca * dF / mu_z`` along its path (``F`` the
    profile's normalized cumulative optical depth -- see
    :mod:`dsh.dust_model`), a relation that holds regardless of the sign of
    ``mu_z`` or of the local dust density. Given a pre-sampled ``Exp(1)``
    variate ``xi = -log(1 - u_tau)``, this inverts that relation for the
    next-event LOS position, or determines the photon exits the dusty region
    [0, 1] first.

    Returns
    -------
    scatters : bool (n,), True if a real scattering event happens before the
        photon would leave the [0, 1] domain.
    x_next : float (n,), the LOS fraction of the event (if ``scatters``) or
        of the domain boundary the photon exits through (if not).
    delta_s_cm : float (n,), physical arc length traveled this leg [cm]
        (the caller multiplies this by the *current* direction and the
        source-observer distance is already folded in via ``x``... see
        :mod:`dsh.tracer` for how this combines with D_cm).
    """
    xi = -jnp.log1p(-u_tau)
    mu_z_safe = safe_mu_z(mu_z)

    F_current = profile.cumulative_fraction(x_current)
    x_boundary = jnp.where(mu_z < 0, 0.0, 1.0)
    F_boundary = profile.cumulative_fraction(x_boundary)

    tau_available = tau_sca * (F_boundary - F_current) / mu_z_safe
    scatters = xi < tau_available

    F_next_raw = F_current + mu_z_safe * xi / tau_sca
    F_next_safe = jnp.clip(F_next_raw, 1e-6, 1.0 - 1e-6)
    x_scatter = profile.inverse_cumulative_fraction(F_next_safe)

    x_next = jnp.where(scatters, x_scatter, x_boundary)
    delta_x = x_next - x_current
    return scatters, x_next, delta_x


def sample_grain_bin_index(key, tables, n):
    """Categorical sample of which grain-size bin causes each scattering event."""
    return jax.random.choice(key, tables.bin_centers_um.shape[0], shape=(n,), p=tables.bin_weights)


def sample_scatter_angle(key, tables, bin_idx):
    """Inverse-CDF sample of the scattering polar angle, per photon's grain bin."""
    n = bin_idx.shape[0]
    u = jax.random.uniform(key, shape=(n,))
    cdf_rows = tables.cdf[bin_idx]
    theta_rows = tables.theta_grid[bin_idx]
    return jax.vmap(jnp.interp)(u, cdf_rows, theta_rows)


def rotate_direction(dir_vec, theta, phi):
    """Deflect unit vectors ``dir_vec`` by polar angle ``theta`` at azimuth ``phi``.

    ``theta``/``phi`` are measured about the *current* direction (standard
    scattering-angle convention), using an arbitrary but continuous local
    orthonormal frame (Gram-Schmidt against a reference axis chosen per
    photon to avoid near-degeneracy).
    """
    ref = jnp.where(
        jnp.abs(dir_vec[..., 0:1]) < 0.9,
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
    )
    e1 = jnp.cross(dir_vec, ref)
    e1 = e1 / jnp.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = jnp.cross(dir_vec, e1)

    new_dir = jnp.cos(theta)[..., None] * dir_vec + jnp.sin(theta)[..., None] * (
        jnp.cos(phi)[..., None] * e1 + jnp.sin(phi)[..., None] * e2
    )
    return new_dir / jnp.linalg.norm(new_dir, axis=-1, keepdims=True)
