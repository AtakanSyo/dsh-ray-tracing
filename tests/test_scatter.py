import jax
import jax.numpy as jnp
import numpy as np

from dsh import cross_section as cs
from dsh import dust_model, grains, scatter


def test_rotate_direction_preserves_norm_and_deflection_angle():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    n = 200
    # random unit directions
    v = jax.random.normal(k1, (n, 3))
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    theta = jax.random.uniform(k2, (n,), minval=0.0, maxval=jnp.pi / 2)
    phi = jax.random.uniform(k3, (n,), minval=0.0, maxval=2 * jnp.pi)

    new_v = scatter.rotate_direction(v, theta, phi)
    norms = np.asarray(jnp.linalg.norm(new_v, axis=-1))
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    cos_angle = np.asarray(jnp.sum(v * new_v, axis=-1))
    np.testing.assert_allclose(cos_angle, np.asarray(jnp.cos(theta)), atol=1e-4)


def test_rotate_direction_theta_zero_is_identity():
    v = jnp.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    theta = jnp.zeros(2)
    phi = jnp.array([0.3, 1.2])
    new_v = scatter.rotate_direction(v, theta, phi)
    np.testing.assert_allclose(np.asarray(new_v), np.asarray(v), atol=1e-6)


def test_sample_next_event_matches_analytic_scatter_probability():
    # A uniform slab spanning the *entire* domain [0, 1] means a photon
    # launched from the source (x=1, mu_z=-1) has, by construction, the
    # full tau_sca available before it can reach the observer -- so
    # P(scatter) = 1 - exp(-tau_sca) exactly.
    profile = dust_model.uniform_slab(0.0, 1.0)
    tau_sca = 0.7
    n = 200_000
    u = jax.random.uniform(jax.random.PRNGKey(1), (n,))
    x_current = jnp.ones(n)
    mu_z = -jnp.ones(n)
    scatters, x_next, _ = scatter.sample_next_event(u, profile, tau_sca, x_current, mu_z)

    empirical = float(jnp.mean(scatters.astype(jnp.float32)))
    expected = 1.0 - np.exp(-tau_sca)
    assert abs(empirical - expected) < 0.01

    x_next_np = np.asarray(x_next)
    assert np.all(x_next_np >= -1e-6) and np.all(x_next_np <= 1.0 + 1e-6)


def test_sample_scatter_angle_respects_cdf_median():
    g = grains.mrn_distribution()
    tables = cs.build_scattering_tables(g, E_keV=1.0, n_bins=16, n_theta=1001)
    n = 50_000
    key = jax.random.PRNGKey(2)
    k_bin, k_theta = jax.random.split(key)
    bin_idx = scatter.sample_grain_bin_index(k_bin, tables, n)
    theta = scatter.sample_scatter_angle(k_theta, tables, bin_idx)
    theta_np = np.asarray(theta)
    assert np.all(theta_np >= 0.0) and np.all(theta_np <= np.pi)

    # median sampled angle for a single bin should match that bin's CDF-inverted median
    single_bin = jnp.zeros(n, dtype=jnp.int32)
    theta_single = np.asarray(scatter.sample_scatter_angle(k_theta, tables, single_bin))
    table_median = float(jnp.interp(0.5, tables.cdf[0], tables.theta_grid[0]))
    assert abs(np.median(theta_single) - table_median) < 0.1 * (table_median + 1e-12) + 1e-8
