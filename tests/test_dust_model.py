import jax.numpy as jnp
import numpy as np

from dsh import dust_model


def test_uniform_slab_roundtrip():
    profile = dust_model.uniform_slab(0.2, 0.6)
    assert float(profile.cumulative_fraction(0.2)) == 0.0
    assert float(profile.cumulative_fraction(0.6)) == 1.0
    assert float(profile.cumulative_fraction(0.05)) == 0.0
    assert float(profile.cumulative_fraction(0.9)) == 1.0

    u = jnp.linspace(0.01, 0.99, 25)
    x = profile.inverse_cumulative_fraction(u)
    back = profile.cumulative_fraction(x)
    np.testing.assert_allclose(np.asarray(back), np.asarray(u), atol=1e-5)


def test_thin_screen_clips_to_domain():
    profile = dust_model.thin_screen(0.0005, width=1e-3)
    assert float(profile.x_lo) >= 0.0
    profile2 = dust_model.thin_screen(0.9995, width=1e-3)
    assert float(profile2.x_hi) <= 1.0


def test_exponential_disk_roundtrip_and_endpoints():
    profile = dust_model.exponential_disk(scale_x=0.1, x_max=0.5)
    assert np.isclose(float(profile.cumulative_fraction(0.0)), 0.0, atol=1e-6)
    assert np.isclose(float(profile.cumulative_fraction(0.5)), 1.0, atol=1e-6)

    u = jnp.linspace(0.01, 0.99, 25)
    x = profile.inverse_cumulative_fraction(u)
    back = profile.cumulative_fraction(x)
    np.testing.assert_allclose(np.asarray(back), np.asarray(u), atol=1e-4)


def test_profiles_monotonic_cumulative_fraction():
    for profile in [dust_model.uniform_slab(0.1, 0.9), dust_model.exponential_disk(0.2, 1.0)]:
        x = jnp.linspace(0.0, 1.0, 200)
        F = np.asarray(profile.cumulative_fraction(x))
        assert np.all(np.diff(F) >= -1e-8)
