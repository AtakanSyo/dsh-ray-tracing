import jax.numpy as jnp
import numpy as np
import pytest

from dsh import cross_section as cs
from dsh import grains


def test_j1_over_y_matches_scipy():
    scipy_special = pytest.importorskip("scipy.special")
    y = np.concatenate([np.array([0.0, 1e-6, 1e-4, 1e-3]), np.linspace(0.01, 50, 200)])
    expected = scipy_special.spherical_jn(1, y) / np.where(y == 0, 1.0, y)
    expected[0] = 1.0 / 3.0  # true limit at y=0
    got = np.asarray(cs._j1_over_y(jnp.asarray(y)))
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-8)


def test_j1_over_y_series_matches_exact_branch_at_boundary():
    # continuity check straddling the small-y series/exact-formula switch at |y|=0.5
    y = jnp.array([0.49, 0.5, 0.51, 0.6])
    vals = np.asarray(cs._j1_over_y(y))
    assert np.all(np.isfinite(vals))
    # float32's ~7 significant digits, combined with the exact formula's
    # residual cancellation right at the switch point, limits agreement to
    # a few parts in 1e3 here -- looser than the series' own truncation
    # error (which is ~1e-7), but that's the float32 exact-branch floor.
    np.testing.assert_allclose(vals[0], vals[1], rtol=2e-3)
    np.testing.assert_allclose(vals[1], vals[2], rtol=2e-3)


def test_dsigma_dOmega_positive_and_forward_peaked():
    theta = jnp.linspace(0.0, jnp.pi, 500)
    dsig = cs.dsigma_dOmega(theta, a_um=0.1, E_keV=1.0)
    dsig = np.asarray(dsig)
    assert np.all(dsig >= 0)
    assert np.all(np.isfinite(dsig))
    assert dsig[0] == dsig.max()  # forward-peaked: theta=0 is the global max


def test_total_cross_section_positive_and_grows_with_grain_size():
    sigma_small = cs.total_cross_section(a_um=0.005, E_keV=1.0)
    sigma_large = cs.total_cross_section(a_um=0.25, E_keV=1.0)
    assert float(sigma_small) > 0
    assert float(sigma_large) > 0
    assert np.isfinite(float(sigma_small))
    assert np.isfinite(float(sigma_large))
    # a factor-50 increase in grain radius should increase the cross section by orders
    # of magnitude (Rayleigh-ish a^6 scaling near the small-grain end), even allowing
    # for O(1) diffraction-ringing wiggles away from strict monotonicity.
    assert float(sigma_large) > 100 * float(sigma_small)


def test_total_cross_section_vmap_over_grain_size():
    import jax

    a_grid = jnp.geomspace(0.005, 0.25, 16)
    vals = jax.vmap(lambda a: cs.total_cross_section(a, 1.0))(a_grid)
    assert vals.shape == (16,)
    assert np.all(np.isfinite(np.asarray(vals)))
    assert np.all(np.asarray(vals) > 0)


def test_build_scattering_tables_normalization():
    g = grains.mrn_distribution()
    tables = cs.build_scattering_tables(g, E_keV=2.0, n_bins=20, n_theta=801)

    assert np.isclose(float(jnp.sum(tables.bin_weights)), 1.0, atol=1e-4)
    assert np.all(np.asarray(tables.bin_weights) >= 0)

    # CDF should end at ~1 and be monotonically non-decreasing per bin
    cdf = np.asarray(tables.cdf)
    np.testing.assert_allclose(cdf[:, -1], 1.0, atol=1e-3)
    assert np.all(np.diff(cdf, axis=-1) >= -1e-6)


def test_mixture_phase_function_normalized_to_unit_solid_angle_integral():
    g = grains.mrn_distribution()
    tables = cs.build_scattering_tables(g, E_keV=1.0, n_bins=24, n_theta=1501)

    # Integrate Phi(theta) over the full sphere on the union of every grain
    # bin's own (correctly-resolved-per-bin) angular grid: each bin's grid
    # already spans exactly the dynamic range that bin's diffraction pattern
    # needs (see _theta_grid_and_pdf), and bins differ enormously in angular
    # scale (sub-arcmin for the largest grains to several degrees for the
    # smallest), so no single fixed nondimensional grid resolves all of them
    # at once.
    theta = jnp.sort(tables.theta_grid.reshape(-1))
    phi = cs.mixture_phase_function(theta, tables)
    integral = cs._cumtrapz_last_axis(phi * 2.0 * jnp.pi * jnp.sin(theta), theta)[-1]
    assert np.isclose(float(integral), 1.0, rtol=0.1)
