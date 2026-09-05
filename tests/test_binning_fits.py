import jax.numpy as jnp
import numpy as np
import pytest

import dsh

pytest.importorskip("astropy")


def test_save_and_load_fits_roundtrip(tmp_path):
    bin_spec = dsh.default_bin_spec(n_theta=10, n_dt=8)
    rng = np.random.default_rng(0)
    image = jnp.asarray(rng.uniform(size=bin_spec.shape).astype(np.float32))

    path = tmp_path / "halo.fits"
    dsh.save_fits(image, bin_spec, path, metadata={"D_PC": 3000, "TAU_SCA": 0.2})

    loaded_image, loaded_bin_spec = dsh.load_fits(path)
    np.testing.assert_allclose(loaded_image, np.asarray(image), rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(loaded_bin_spec.theta_edges), np.asarray(bin_spec.theta_edges), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(loaded_bin_spec.dt_edges), np.asarray(bin_spec.dt_edges), rtol=1e-5
    )


def test_save_fits_header_metadata(tmp_path):
    from astropy.io import fits

    bin_spec = dsh.default_bin_spec(n_theta=5, n_dt=5)
    image = dsh.binning.zeros(bin_spec)
    path = tmp_path / "halo.fits"
    dsh.save_fits(image, bin_spec, path, metadata={"D_PC": 3000, "E_KEV": 1.0})

    with fits.open(path) as hdul:
        assert hdul[0].header["D_PC"] == 3000
        assert hdul[0].header["E_KEV"] == 1.0
        assert "THETA_EDGES" in hdul
        assert "DT_EDGES" in hdul
