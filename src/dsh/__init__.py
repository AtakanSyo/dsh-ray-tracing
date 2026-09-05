"""dsh: a differentiable, GPU-accelerated Monte Carlo ray tracer for
X-ray dust-scattering halos, built on JAX.

See the project README for the physics background and a quickstart. Public
entry points are re-exported here for convenience:

    import dsh
    tables = dsh.build_scattering_tables(dsh.mrn_distribution(), E_keV=1.0)
    profile = dsh.uniform_slab(0.3, 0.4)
    result = dsh.simulate(jax.random.PRNGKey(0), n_photons=200_000,
                           D_pc=3000, E_keV=1.0, tau_sca=0.1, profile=profile,
                           bin_spec=dsh.default_bin_spec())
"""

from . import (
    binning,
    constants,
    cross_section,
    dust_model,
    geometry,
    grains,
    photon,
    scatter,
    tracer,
)
from .binning import BinSpec, linear_dt_bins, log_theta_bins
from .cross_section import (
    ScatteringTables,
    build_scattering_tables,
    dsigma_dOmega,
    total_cross_section,
)
from .dust_model import exponential_disk, thin_screen, uniform_slab
from .grains import mrn_distribution
from .tracer import TraceResult, simulate

__version__ = "0.1.0"


def default_bin_spec(
    theta_min_arcsec=1.0, theta_max_arcsec=3600.0, n_theta=60, dt_min_s=0.0, dt_max_s=1.0e5, n_dt=60
):
    """A reasonable default (sky angle, time delay) binning for a quick look."""
    return BinSpec(
        theta_edges=log_theta_bins(theta_min_arcsec, theta_max_arcsec, n_theta),
        dt_edges=linear_dt_bins(dt_min_s, dt_max_s, n_dt),
    )


__all__ = [
    "BinSpec",
    "ScatteringTables",
    "TraceResult",
    "binning",
    "build_scattering_tables",
    "constants",
    "cross_section",
    "default_bin_spec",
    "dsigma_dOmega",
    "dust_model",
    "exponential_disk",
    "geometry",
    "grains",
    "linear_dt_bins",
    "log_theta_bins",
    "mrn_distribution",
    "photon",
    "scatter",
    "simulate",
    "thin_screen",
    "total_cross_section",
    "tracer",
    "uniform_slab",
]
