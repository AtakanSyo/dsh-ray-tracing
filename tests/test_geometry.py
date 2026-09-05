import jax.numpy as jnp
import numpy as np

from dsh import geometry
from dsh.constants import C_LIGHT_CM_S


def _brute_force_delay(x, theta, D_cm):
    """Independent re-derivation via the law of cosines, straight from the
    module docstring's geometric setup, to cross-check geometry.time_delay."""
    l = x * D_cm
    b = l * np.tan(theta)
    path = np.sqrt(b**2 + (D_cm - l) ** 2) + np.sqrt(b**2 + l**2)
    return (path - D_cm) / C_LIGHT_CM_S


def test_time_delay_matches_independent_law_of_cosines_derivation():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.05, 0.95, size=50)
    theta = rng.uniform(1e-6, 0.3, size=50)
    D_cm = 3.0e21
    got = np.asarray(geometry.time_delay(jnp.asarray(x), jnp.asarray(theta), D_cm))
    expected = _brute_force_delay(x, theta, D_cm)
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_time_delay_zero_at_zero_angle():
    x = jnp.linspace(0.01, 0.99, 20)
    dt = geometry.time_delay(x, jnp.zeros_like(x), 1e21)
    np.testing.assert_allclose(np.asarray(dt), 0.0, atol=1e-6)


def test_time_delay_matches_small_angle_limit():
    x = 0.4
    D_cm = 5e21
    theta = 1e-6
    exact = float(geometry.time_delay(x, theta, D_cm))
    approx = float(geometry.time_delay_small_angle(x, theta, D_cm))
    np.testing.assert_allclose(exact, approx, rtol=1e-4)


def test_time_delay_increases_with_angle():
    x = 0.5
    D_cm = 1e21
    thetas = np.linspace(1e-5, 0.5, 30)
    dt = np.asarray(geometry.time_delay(x, thetas, D_cm))
    assert np.all(np.diff(dt) > 0)


def test_sky_observables_on_axis_point_gives_zero_angle_and_zero_delay():
    # Fractional (dimensionless) coordinates: z=0.3 on-axis, no accumulated
    # excess path (a photon that only ever traveled exactly along the axis).
    D_cm = 4e21
    r = jnp.array([[0.0, 0.0, 0.3]])
    excess_path_frac = jnp.array([0.0])
    theta_sky, _phi_sky, dt, R = geometry.sky_observables(r, excess_path_frac, D_cm)
    np.testing.assert_allclose(np.asarray(theta_sky), 0.0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(dt), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(R), 0.3, rtol=1e-6)


def test_sky_observables_matches_brute_force_off_axis():
    D_cm = 4e21
    x, y, z = 1.5e-4, -3.0e-5, 0.6
    r = jnp.array([[x, y, z]])
    excess_path_frac = jnp.array([2.0e-9])
    theta_sky, _phi_sky, dt, R = geometry.sky_observables(r, excess_path_frac, D_cm)

    R_expected = np.sqrt(x**2 + y**2 + z**2)
    theta_expected = np.arctan2(np.sqrt(x**2 + y**2), z)
    dt_expected = (2.0e-9 + (R_expected - z)) * D_cm / geometry.C_LIGHT_CM_S

    np.testing.assert_allclose(np.asarray(R), R_expected, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(theta_sky), theta_expected, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(dt), dt_expected, rtol=1e-3)


def test_leg_excess_path_fraction_zero_for_axial_travel():
    dir_vec = jnp.array([[0.0, 0.0, -1.0]])
    excess = geometry.leg_excess_path_fraction(jnp.array([0.5]), dir_vec)
    np.testing.assert_allclose(np.asarray(excess), 0.0, atol=1e-10)


def test_leg_excess_path_fraction_matches_one_minus_cosine_formula():
    # A leg of *arc length* delta_s traveled at deflection angle phi from the
    # axis covers axial (towards-observer) distance delta_s*cos(phi); the
    # excess is the remainder, delta_s*(1 - cos(phi)), checked here against a
    # direct (non-cancellation-safe) float64 reference.
    rng = np.random.default_rng(1)
    phi = rng.uniform(1e-4, 1.0, size=30)
    delta_s = rng.uniform(0.01, 1.0, size=30)
    dir_vec = np.stack([np.sin(phi), np.zeros_like(phi), -np.cos(phi)], axis=-1)
    got = np.asarray(geometry.leg_excess_path_fraction(jnp.asarray(delta_s), jnp.asarray(dir_vec)))
    expected = delta_s * (1.0 - np.cos(phi))
    np.testing.assert_allclose(got, expected, rtol=1e-4)
