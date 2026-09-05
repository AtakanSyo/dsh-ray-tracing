"""Small-angle X-ray scattering by dust grains: Rayleigh-Gans cross section.

Physics
-------
X-rays scattering off interstellar dust grains satisfy the two Rayleigh-Gans
(RG) conditions extremely well over the grain sizes/energies relevant to
observed dust-scattering halos: the grain's refractive index is very close to
1 (``|m-1| << 1``), and ``k*a*|m-1| << 1``. Under these conditions the exact
differential scattering cross section is (e.g. Mauche & Gorenstein 1986, ApJ
303, 569; Smith & Dwek 1998, ApJ 503, 831):

    dsigma/dOmega(E, a, theta) = 2 a^2 (2*pi*a/lambda)^4 |m-1|^2
                                  * (j1(y)/y)^2 * (1 + cos^2(theta))

    y = (4*pi*a/lambda) * sin(theta/2)

where ``a`` is the grain radius, ``lambda`` the photon wavelength, ``theta``
the scattering angle, ``m`` the grain material's complex refractive index,
and ``j1`` the spherical Bessel function of the first order. This module
implements that formula exactly (``j1`` written in closed elementary form,
with a Taylor series used near ``y = 0`` to avoid the removable 0/0
singularity) rather than the commonly-used small-angle Gaussian
approximation, so the diffraction structure at larger angles is retained.

The complex refractive index is modeled with the Drude (free-electron)
approximation, valid away from photo-absorption edges:

    |m - 1| ~= n_e * r_e * lambda^2 / (2*pi),      n_e = rho_grain * N_A / mu_e

This is a first-order approximation to the grain material (no edge
structure); ``mu_e`` (mean atomic mass per electron) and ``rho_grain``
(grain material density) are the two knobs, defaulting to values typical of
astronomical silicate/graphite mixes (``rho_grain = 3 g/cm^3``, ``mu_e = 2``).
Everything downstream (total cross section, phase function, characteristic
scattering angle) is derived from this one formula by numerical quadrature
rather than from separately-memorized closed-form totals, so the pieces stay
internally consistent -- swap in tabulated optical constants later by
replacing :func:`refractive_index_drude`.

Grain-size mixing
------------------
A photon's scattering partner (grain size) is not fixed: at every real
scattering event it is effectively drawn from the local grain population.
:func:`build_scattering_tables` discretizes a grain-size distribution (see
:mod:`dsh.grains`) into a modest number of log-spaced bins, and precomputes,
per bin: the relative probability that a scattering event happens off a
grain in that bin (proportional to ``n(a) * sigma_tot(a)``), and a tabulated inverse-CDF
of the angular phase function for efficient GPU sampling. The mixture phase
function (needed for the peel-off estimator in the tracer) is the
bin-weighted sum of the individual grains' normalized phase functions.
"""

from typing import NamedTuple

import jax.numpy as jnp

from .constants import UM_TO_CM, wavelength_cm


def _j1_over_y(y):
    """spherical Bessel j1(y) / y, safe (and accurate) as y -> 0.

    j1(y) = sin(y)/y^2 - cos(y)/y, so j1(y)/y = sin(y)/y^3 - cos(y)/y^2.
    Near y=0 this is a 0/0 form computed as a difference of two large,
    nearly-equal terms (~1/y^2 each) -- in float32 that cancellation is
    already inaccurate to a percent or more by y ~ 1e-2, well before the
    naive "is y suspiciously small" threshold of 1e-3 one might guess. The
    Taylor series (whose next dropped term, ``-y^6/45360``, is <= 3e-7 for
    all ``|y| < 0.5``) is used up to that radius instead.
        j1(y)/y = 1/3 - y^2/30 + y^4/840 - ...
    """
    y = jnp.asarray(y)
    small = jnp.abs(y) < 0.5
    y_safe = jnp.where(small, 1.0, y)
    exact = jnp.sin(y_safe) / y_safe**3 - jnp.cos(y_safe) / y_safe**2
    series = 1.0 / 3.0 - y**2 / 30.0 + y**4 / 840.0
    return jnp.where(small, series, exact)


def refractive_index_drude(E_keV, rho_grain_g_cm3=3.0, mu_e=2.0):
    """Drude (free-electron) approximation to |m - 1| for a grain material.

    Parameters
    ----------
    E_keV : photon energy [keV]
    rho_grain_g_cm3 : grain material bulk density [g/cm^3]
    mu_e : mean atomic mass per electron (A/Z), dimensionless

    Returns
    -------
    |m - 1|, dimensionless.
    """
    from .constants import N_AVOGADRO, R_ELECTRON_CM

    lam_cm = wavelength_cm(E_keV)
    n_e = rho_grain_g_cm3 * N_AVOGADRO / mu_e
    return n_e * R_ELECTRON_CM * lam_cm**2 / (2.0 * jnp.pi)


def dsigma_dOmega(theta_rad, a_um, E_keV, rho_grain_g_cm3=3.0, mu_e=2.0):
    """Exact Rayleigh-Gans differential scattering cross section [cm^2/sr].

    Broadcasts over ``theta_rad`` and ``a_um``; ``E_keV`` is normally a
    scalar (monochromatic source) but may also broadcast.
    """
    a_cm = jnp.asarray(a_um) * UM_TO_CM
    lam_cm = wavelength_cm(E_keV)
    m_minus_1 = refractive_index_drude(E_keV, rho_grain_g_cm3, mu_e)

    x = 2.0 * jnp.pi * a_cm / lam_cm
    y = 2.0 * x * jnp.sin(theta_rad / 2.0)
    j1_ratio = _j1_over_y(y)

    prefactor = 2.0 * a_cm**2 * x**4 * m_minus_1**2
    return prefactor * j1_ratio**2 * (1.0 + jnp.cos(theta_rad) ** 2)


def _theta_grid_and_pdf(a_um, E_keV, n_theta=2001, u_max=300.0, rho_grain_g_cm3=3.0, mu_e=2.0):
    """Adaptive scattering-angle grid + unnormalized angular PDF pieces.

    The natural angular scale of the RG diffraction pattern is
    ``lambda / (2*pi*a)``; the grid is built in that dimensionless unit
    (log-spaced) so it resolves the forward peak regardless of grain size
    or energy, then clipped to ``[0, pi]``.

    Returns ``theta`` (shape ``a_um.shape + (n_theta,)``), ``dsigma_dOmega``
    on that grid (same shape), and the solid-angle element ``dOmega/dtheta =
    2*pi*sin(theta)`` (same shape).
    """
    a_um = jnp.asarray(a_um, dtype=jnp.float32)
    u_nodes = jnp.concatenate([jnp.zeros(1), jnp.geomspace(1e-5, u_max, n_theta - 1)])

    lam_cm = wavelength_cm(E_keV)
    a_cm = a_um * UM_TO_CM
    theta_scale = lam_cm / (2.0 * jnp.pi * a_cm)  # shape a_um.shape

    theta = jnp.clip(theta_scale[..., None] * u_nodes, 0.0, jnp.pi)
    dsig = dsigma_dOmega(theta, a_um[..., None], E_keV, rho_grain_g_cm3, mu_e)
    dOm = 2.0 * jnp.pi * jnp.sin(theta)
    return theta, dsig, dOm


def _cumtrapz_last_axis(y, x):
    """Cumulative trapezoidal integral of y(x) along the last axis, starting at 0.

    A hand-rolled implementation (rather than ``jnp.trapz``/``jnp.trapezoid``)
    since that function's very name has churned across numpy/JAX versions
    (``trapz`` deprecated and removed in favor of ``trapezoid``); this needs
    no version-dependent API at all.
    """
    dx = jnp.diff(x, axis=-1)
    avg = 0.5 * (y[..., 1:] + y[..., :-1])
    inc = avg * dx
    cum = jnp.cumsum(inc, axis=-1)
    return jnp.concatenate([jnp.zeros_like(cum[..., :1]), cum], axis=-1)


def total_cross_section(a_um, E_keV, n_theta=2001, u_max=300.0, rho_grain_g_cm3=3.0, mu_e=2.0):
    """Total scattering cross section sigma_sca(a, E) [cm^2], by quadrature.

    Broadcasts over ``a_um`` (scalar or array).
    """
    theta, dsig, dOm = _theta_grid_and_pdf(a_um, E_keV, n_theta, u_max, rho_grain_g_cm3, mu_e)
    return _cumtrapz_last_axis(dsig * dOm, theta)[..., -1]


class ScatteringTables(NamedTuple):
    """Precomputed per-grain-size-bin scattering tables for a fixed energy.

    A plain :class:`typing.NamedTuple` of arrays is used (rather than a
    custom pytree) so instances pass through ``jax.jit``/``vmap``/``scan``
    transparently.
    """

    bin_centers_um: jnp.ndarray  # (n_bins,)
    bin_weights: jnp.ndarray  # (n_bins,), sums to 1: P(scattering event is off bin i)
    sigma_tot_per_bin: jnp.ndarray  # (n_bins,) [cm^2]
    theta_grid: jnp.ndarray  # (n_bins, n_theta) [rad]
    cdf: jnp.ndarray  # (n_bins, n_theta), in [0, 1]
    dsigma_dOmega_grid: jnp.ndarray  # (n_bins, n_theta) [cm^2/sr], for inspection/plotting
    E_keV: jnp.ndarray  # scalar, the energy this table was built for


def build_scattering_tables(
    grain_distribution,
    E_keV,
    n_bins=32,
    n_theta=2001,
    u_max=300.0,
    rho_grain_g_cm3=3.0,
    mu_e=2.0,
):
    """Discretize a grain size distribution into scattering tables at energy E.

    Parameters
    ----------
    grain_distribution : dsh.grains.PowerLawGrainDistribution
    E_keV : float, photon energy [keV] (monochromatic)
    n_bins : number of log-spaced grain-size bins
    n_theta : number of scattering-angle grid points per bin

    Returns
    -------
    ScatteringTables
    """
    edges_um = grain_distribution.log_spaced_bin_edges(n_bins)
    centers_um = grain_distribution.bin_centers_geometric(edges_um)
    number_frac = grain_distribution.bin_number_fraction(edges_um)

    a_um = jnp.asarray(centers_um, dtype=jnp.float32)
    E_keV = jnp.asarray(E_keV, dtype=jnp.float32)

    theta, dsig, dOm = _theta_grid_and_pdf(a_um, E_keV, n_theta, u_max, rho_grain_g_cm3, mu_e)
    pdf_unnorm = dsig * dOm  # d(sigma)/d(theta), unnormalized per bin

    cdf_unnorm = _cumtrapz_last_axis(pdf_unnorm, theta)
    sigma_tot_per_bin = cdf_unnorm[..., -1]
    cdf = cdf_unnorm / sigma_tot_per_bin[..., None]

    number_frac = jnp.asarray(number_frac, dtype=jnp.float32)
    raw_weight = number_frac * sigma_tot_per_bin
    bin_weights = raw_weight / jnp.sum(raw_weight)

    return ScatteringTables(
        bin_centers_um=a_um,
        bin_weights=bin_weights,
        sigma_tot_per_bin=sigma_tot_per_bin,
        theta_grid=theta,
        cdf=cdf,
        dsigma_dOmega_grid=dsig,
        E_keV=E_keV,
    )


def mixture_phase_function(theta_rad, tables: ScatteringTables, rho_grain_g_cm3=3.0, mu_e=2.0):
    """Grain-mixture-averaged, normalized angular scattering PDF Phi(theta) [1/sr].

    ``Phi`` is the probability density (per steradian) of scattering into
    angle ``theta`` given that a scattering event occurs, averaged over the
    grain population with the same relative weights used for sampling
    (:data:`ScatteringTables.bin_weights`). Used by the tracer's peel-off
    (next-event estimator) step, where ``theta`` is an arbitrary angle (the
    one pointing at the observer), not restricted to a table grid point.
    """
    theta_rad = jnp.asarray(theta_rad)
    a_i = tables.bin_centers_um  # (n_bins,)
    dsig = dsigma_dOmega(
        theta_rad[..., None], a_i, tables.E_keV, rho_grain_g_cm3, mu_e
    )  # (..., n_bins)
    normalized = dsig / tables.sigma_tot_per_bin  # (..., n_bins)
    return jnp.sum(normalized * tables.bin_weights, axis=-1)
