import numpy as np

from dsh import grains

# numpy renamed trapz -> trapezoid in 2.0 (and later dropped trapz); avoid
# depending on either specific name.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def test_dn_da_zero_outside_support():
    g = grains.mrn_distribution()
    assert float(g.dn_da(0.001)) == 0.0
    assert float(g.dn_da(1.0)) == 0.0
    assert float(g.dn_da(0.05)) > 0.0


def test_bin_number_fraction_sums_to_one_and_matches_numeric_integral():
    g = grains.mrn_distribution()
    edges = g.log_spaced_bin_edges(40)
    frac = g.bin_number_fraction(edges)
    assert frac.shape == (40,)
    assert np.isclose(frac.sum(), 1.0)
    assert np.all(frac >= 0)

    # cross-check the analytic per-bin integral against brute-force quadrature
    fine_a = np.geomspace(g.a_min_um, g.a_max_um, 200_000)
    fine_dnda = np.asarray(g.dn_da(fine_a))
    total = _trapz(fine_dnda, fine_a)
    idx = np.searchsorted(edges, fine_a) - 1
    idx = np.clip(idx, 0, 39)
    numeric_frac = np.array([_trapz(fine_dnda[idx == k], fine_a[idx == k]) for k in range(40)])
    numeric_frac /= total
    np.testing.assert_allclose(frac, numeric_frac, atol=5e-3)


def test_bin_centers_within_edges():
    g = grains.mrn_distribution()
    edges = g.log_spaced_bin_edges(10)
    centers = g.bin_centers_geometric(edges)
    assert np.all(centers > edges[:-1])
    assert np.all(centers < edges[1:])
