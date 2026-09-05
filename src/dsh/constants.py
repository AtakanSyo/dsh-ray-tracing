"""Physical constants and unit conversions.

Internal convention used throughout ``dsh``:

- length              : centimeters (cm)
- grain radius ``a``   : microns (um), converted to cm at the point of use
- energy ``E``         : keV
- angle               : radians
- time                : seconds
- source distance ``D``: centimeters (helpers below convert from parsec)

Keeping units explicit in variable/argument names (``a_um``, ``E_keV``,
``D_pc``, ``theta_rad``, ...) rather than adopting a full unit-handling
library (e.g. ``astropy.units``) is a deliberate choice: those libraries do
not, in general, survive ``jax.jit``/``vmap`` cleanly. All conversions are
plain floats/arrays.
"""

import numpy as np

# --- fundamental constants (CGS / keV) --------------------------------------

C_LIGHT_CM_S = 2.99792458e10
"""Speed of light [cm/s]."""

R_ELECTRON_CM = 2.8179403262e-13
"""Classical electron radius [cm]."""

N_AVOGADRO = 6.02214076e23
"""Avogadro's number [1/mol]."""

HC_KEV_CM = 1.2398419843e-7
"""Planck constant times speed of light, hc [keV * cm].

Equivalent to the well known hc = 12398.4198 eV * angstrom.
"""

# --- length conversions ------------------------------------------------------

PC_TO_CM = 3.0856775814913673e18
UM_TO_CM = 1.0e-4
ANGSTROM_TO_CM = 1.0e-8

# --- angle conversions --------------------------------------------------------

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)
DEG_TO_RAD = np.pi / 180.0

RAD_TO_ARCSEC = 1.0 / ARCSEC_TO_RAD
RAD_TO_ARCMIN = 1.0 / ARCMIN_TO_RAD
RAD_TO_DEG = 1.0 / DEG_TO_RAD


def wavelength_cm(E_keV):
    """Photon wavelength [cm] for photon energy ``E_keV`` [keV]."""
    return HC_KEV_CM / E_keV
