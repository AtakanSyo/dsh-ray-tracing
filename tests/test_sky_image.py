import jax
import jax.numpy as jnp
import numpy as np

import dsh


def test_dt_window_sum_full_range_matches_plain_sum():
    bin_spec = dsh.default_bin_spec(n_theta=10, n_dt=8)
    rng = np.random.default_rng(0)
    image = jnp.asarray(rng.uniform(size=bin_spec.shape).astype(np.float32))
    full = dsh.dt_window_sum(image, bin_spec)
    np.testing.assert_allclose(np.asarray(full), np.asarray(jnp.sum(image, axis=1)))


def test_dt_window_sum_narrow_window_selects_subset():
    bin_spec = dsh.default_bin_spec(n_theta=5, n_dt=10, dt_min_s=0, dt_max_s=100)
    image = dsh.binning.zeros(bin_spec).at[2, 5].set(7.0)  # one bin lit up
    dt_centers = 0.5 * (np.asarray(bin_spec.dt_edges[:-1]) + np.asarray(bin_spec.dt_edges[1:]))
    lit_center = dt_centers[5]

    hit = dsh.dt_window_sum(image, bin_spec, dt_min_s=lit_center - 1, dt_max_s=lit_center + 1)
    miss = dsh.dt_window_sum(image, bin_spec, dt_min_s=lit_center + 20, dt_max_s=lit_center + 30)
    assert float(hit[2]) == 7.0
    assert float(jnp.sum(miss)) == 0.0


def test_sky_image_is_radially_symmetric_and_matches_profile_at_axes():
    bin_spec = dsh.default_bin_spec(theta_min_arcsec=1, theta_max_arcsec=1000, n_theta=30, n_dt=1)
    theta_arcsec = np.asarray(dsh.binning.theta_bin_centers(bin_spec)) * dsh.constants.RAD_TO_ARCSEC
    radial = np.exp(-theta_arcsec / 200.0)  # smooth, monotonic decaying profile

    image2d, _x, _y = dsh.sky_image(jnp.asarray(radial), bin_spec, extent_arcsec=800, npix=101)
    assert image2d.shape == (101, 101)
    assert np.all(np.isfinite(image2d))
    assert np.all(image2d >= 0)

    # radial symmetry: value should only depend on distance from center
    cy = cx = 50  # center pixel index (npix=101 -> center at index 50)
    r_right = image2d[cy, cx + 20]
    r_up = image2d[cy - 20, cx]
    r_diag = image2d[cy - 14, cx + 14]  # ~ same radius as the above (14*sqrt(2)~20)
    np.testing.assert_allclose(r_right, r_up, rtol=1e-5)
    np.testing.assert_allclose(r_right, r_diag, rtol=0.05)

    # brightness should decrease outward, matching the monotonic input profile
    assert image2d[cy, cx + 5] > image2d[cy, cx + 20] > image2d[cy, cx + 45]


def test_sky_image_center_pixel_below_innermost_bin_is_zero():
    bin_spec = dsh.default_bin_spec(theta_min_arcsec=10, theta_max_arcsec=1000, n_theta=20, n_dt=1)
    radial = jnp.ones(20)
    image2d, x, y = dsh.sky_image(radial, bin_spec, extent_arcsec=500, npix=51)
    center = 25
    assert x[center] == 0.0 and y[center] == 0.0
    assert float(image2d[center, center]) == 0.0


def test_sky_image_accepts_2d_image_with_dt_window():
    key = jax.random.PRNGKey(0)
    profile = dsh.uniform_slab(0.3, 0.35)
    bin_spec = dsh.default_bin_spec(
        theta_min_arcsec=1, theta_max_arcsec=3600, n_theta=20, dt_min_s=0, dt_max_s=1e7, n_dt=20
    )
    result = dsh.simulate(
        key, n_photons=20_000, D_pc=3000, E_keV=1.0, tau_sca=0.2, profile=profile,
        bin_spec=bin_spec, max_scatter_order=5, n_grain_bins=10, n_theta_table=501,
    )
    image2d, _x, _y = dsh.sky_image(result.image, bin_spec, extent_arcsec=1000, npix=41, dt_min_s=0, dt_max_s=1e6)
    assert image2d.shape == (41, 41)
    assert np.all(np.isfinite(image2d))
