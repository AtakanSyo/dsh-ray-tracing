import jax
import jax.numpy as jnp
import numpy as np

import dsh


def _run(tau_sca, n_photons=40_000, key_seed=0, max_scatter_order=8):
    key = jax.random.PRNGKey(key_seed)
    profile = dsh.uniform_slab(0.3, 0.35)
    bin_spec = dsh.default_bin_spec(
        theta_min_arcsec=1, theta_max_arcsec=3600, n_theta=25, dt_min_s=0, dt_max_s=1e7, n_dt=25
    )
    return dsh.simulate(
        key,
        n_photons=n_photons,
        D_pc=3000,
        E_keV=1.0,
        tau_sca=tau_sca,
        profile=profile,
        bin_spec=bin_spec,
        max_scatter_order=max_scatter_order,
        n_grain_bins=12,
        n_theta_table=801,
    )


def test_simulate_smoke_no_nan_and_sane_shapes():
    result = _run(tau_sca=0.2)
    assert result.image.shape == (25, 25)
    assert np.all(np.isfinite(np.asarray(result.image)))
    assert np.all(np.asarray(result.image) >= 0)
    assert float(jnp.sum(result.image)) > 0
    assert 0.0 <= float(result.fraction_still_active) <= 1.0


def test_simulate_zero_optical_depth_gives_empty_image():
    result = _run(tau_sca=1e-8)
    total = float(jnp.sum(result.image))
    assert total < 1e-3  # essentially nothing scatters


def test_simulate_scattered_flux_scales_linearly_at_low_optical_depth():
    # At low tau_sca, first-order (single) scattering dominates and the
    # per-photon probability of a real scattering event is ~tau_sca, so the
    # total peeled-off weight should scale ~linearly with tau_sca.
    low = _run(tau_sca=0.002, n_photons=300_000, max_scatter_order=3)
    high = _run(tau_sca=0.004, n_photons=300_000, max_scatter_order=3)
    ratio = float(jnp.sum(high.image)) / float(jnp.sum(low.image))
    assert 1.7 < ratio < 2.3


def test_simulate_deterministic_given_same_key():
    r1 = _run(tau_sca=0.15, key_seed=42)
    r2 = _run(tau_sca=0.15, key_seed=42)
    np.testing.assert_allclose(np.asarray(r1.image), np.asarray(r2.image))


def test_max_scatter_order_one_only_runs_first_scatter_step():
    result = _run(tau_sca=0.1, max_scatter_order=1)
    assert np.all(np.isfinite(np.asarray(result.image)))
    assert float(jnp.sum(result.image)) > 0


def test_to_surface_brightness_finite_and_nonnegative():
    result = _run(tau_sca=0.2)
    sb = dsh.binning.to_surface_brightness(result.image, result.bin_spec)
    assert np.all(np.isfinite(np.asarray(sb)))
    assert np.all(np.asarray(sb) >= 0)
