"""Photon packet state for the Monte Carlo tracer.

Coordinates: the observer sits at the origin, the point source at
``(0, 0, 1)`` -- positions are stored as *fractions of the source-observer
distance D* (dimensionless), not centimeters, because centimeters at
astronomical distances (D ~ 1e21 cm for a kpc-scale source) overflow
float32 the moment they are squared (as any straight-line-distance formula
must); see :mod:`dsh.geometry` for the full numerical rationale. The
fractional line-of-sight coordinate used by :mod:`dsh.dust_model` is simply
``x = pos[..., 2]`` in these units.

``excess_path_frac`` accumulates the photon's real path length in excess of
the axial (z) distance it has covered, in the same D-relative units (see
:func:`dsh.geometry.leg_excess_path_fraction`) -- this is the quantity that,
converted to seconds, gives the dust-echo time delay; it is accumulated
incrementally in stable small positive steps rather than recovered by
subtracting two large numbers.

``active=False`` marks a photon that has left the dusty region [0, 1] in z
and will never scatter again (its contribution to the halo image was
already recorded, at its last real scattering event, by the tracer's
peel-off step -- see :mod:`dsh.tracer`).
"""

from typing import NamedTuple

import jax.numpy as jnp


class PhotonState(NamedTuple):
    pos: jnp.ndarray  # (n, 3), fractions of D_cm
    dir: jnp.ndarray  # (n, 3) unit vectors
    weight: jnp.ndarray  # (n,) dimensionless
    excess_path_frac: jnp.ndarray  # (n,) fractions of D_cm, see module docstring
    active: jnp.ndarray  # (n,) bool
    order: jnp.ndarray  # (n,) int32, number of real scatters so far


def init_photons(n_photons):
    """All ``n_photons`` start at the source, headed straight at the observer.

    The very first leg is, by construction, exactly axial, so it starts
    with zero excess path (see :mod:`dsh.tracer`'s special handling of the
    first real scattering event).
    """
    pos = jnp.zeros((n_photons, 3), dtype=jnp.float32).at[:, 2].set(1.0)
    dir_ = jnp.zeros((n_photons, 3), dtype=jnp.float32).at[:, 2].set(-1.0)
    return PhotonState(
        pos=pos,
        dir=dir_,
        weight=jnp.ones((n_photons,), dtype=jnp.float32),
        excess_path_frac=jnp.zeros((n_photons,), dtype=jnp.float32),
        active=jnp.ones((n_photons,), dtype=bool),
        order=jnp.zeros((n_photons,), dtype=jnp.int32),
    )
