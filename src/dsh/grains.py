"""Interstellar dust grain size distributions.

Only the *shape* of the size distribution matters for this code: it sets the
relative mix of grain sizes a photon can scatter off, and therefore the shape
of the scattering phase function. The absolute dust column / optical depth is
supplied separately by the user (see :mod:`dsh.dust_model`), so no dust-to-gas
mass ratio or absolute grain abundance constant is needed here.

Default parameters follow the classic MRN distribution:
    Mathis, Rumpl & Nordsieck (1977), ApJ 217, 425 --
    dn/da ∝ a^-3.5,  0.005 um <= a <= 0.25 um.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

MRN_A_MIN_UM = 0.005
MRN_A_MAX_UM = 0.25
MRN_Q = 3.5


@dataclass(frozen=True)
class PowerLawGrainDistribution:
    """A power-law grain size distribution dn/da ∝ a^-q on [a_min_um, a_max_um].

    This is a plain (non-pytree) container of static Python floats -- it is
    meant to be consumed at trace/setup time (e.g. by
    :func:`dsh.cross_section.build_scattering_tables`), not passed through
    jitted code itself.
    """

    a_min_um: float = MRN_A_MIN_UM
    a_max_um: float = MRN_A_MAX_UM
    q: float = MRN_Q

    def dn_da(self, a_um):
        """Unnormalized dn/da (relative units) at grain radius(es) ``a_um``."""
        a_um = jnp.asarray(a_um)
        in_range = (a_um >= self.a_min_um) & (a_um <= self.a_max_um)
        return jnp.where(in_range, a_um ** (-self.q), 0.0)

    def bin_number_fraction(self, edges_um):
        """Fraction of all grains (by number) falling in each bin.

        ``edges_um`` is a length-(n_bins + 1) array of bin edges [um].
        Uses the exact analytic integral of the power law over each bin
        rather than a bin-center approximation.

        Returns an array of length n_bins summing to 1.
        """
        edges_um = np.asarray(edges_um, dtype=np.float64)
        q = self.q
        if abs(q - 1.0) < 1e-12:
            antideriv = np.log(edges_um)
        else:
            antideriv = edges_um ** (1.0 - q) / (1.0 - q)
        raw = np.diff(antideriv)
        # Guard against numerical sign flips for decreasing antiderivatives.
        raw = np.abs(raw)
        return raw / raw.sum()

    def log_spaced_bin_edges(self, n_bins):
        """``n_bins + 1`` grain-radius bin edges [um], log-spaced over the support."""
        return np.geomspace(self.a_min_um, self.a_max_um, n_bins + 1)

    def bin_centers_geometric(self, edges_um):
        """Geometric-mean center of each bin, given its edges [um]."""
        edges_um = np.asarray(edges_um, dtype=np.float64)
        return np.sqrt(edges_um[:-1] * edges_um[1:])


def mrn_distribution(a_min_um=MRN_A_MIN_UM, a_max_um=MRN_A_MAX_UM, q=MRN_Q):
    """Convenience constructor for the standard MRN distribution."""
    return PowerLawGrainDistribution(a_min_um=a_min_um, a_max_um=a_max_um, q=q)
