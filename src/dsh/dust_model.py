"""Line-of-sight dust distribution profiles.

A profile describes only the *shape* of the dust column between observer
(``x=0``) and source (``x=1``), as a normalized cumulative distribution
``cumulative_fraction(x) in [0, 1]`` (monotonically increasing, 0 at the
observer, 1 at the source) together with its inverse. The absolute amount of
dust is a separate scalar, the total scattering optical depth ``tau_sca``
(at the source energy), supplied directly by the caller (see
:mod:`dsh.tracer`) -- this sidesteps needing an absolute dust-to-gas mass
ratio or grain abundance calibration, which the observational literature
typically fits per source anyway.

All profiles are plain :class:`typing.NamedTuple` of JAX scalars, so they
are valid pytrees: their parameters (e.g. a screen's distance) can be
batched over with ``vmap`` or differentiated through with ``grad`` for
inference/fitting use cases, and they pass through ``jit`` unmodified.

Each profile exposes:
    cumulative_fraction(x)          -- F(x) in [0, 1], monotonic increasing
    inverse_cumulative_fraction(u)  -- F^-1(u), u in [0, 1]
"""

from typing import NamedTuple

import jax.numpy as jnp


class UniformSlab(NamedTuple):
    """Uniform dust density between fractional distances x_lo and x_hi from the observer."""

    x_lo: jnp.ndarray
    x_hi: jnp.ndarray

    def cumulative_fraction(self, x):
        return jnp.clip((x - self.x_lo) / (self.x_hi - self.x_lo), 0.0, 1.0)

    def inverse_cumulative_fraction(self, u):
        return self.x_lo + u * (self.x_hi - self.x_lo)


def thin_screen(x0, width=1e-3):
    """A UniformSlab of small ``width`` centered on ``x0``, clipped to [0, 1].

    A literally infinitesimal screen is not compatible with volumetric
    free-path sampling in the Monte Carlo tracer; a narrow slab is the
    standard practical stand-in and is accurate as long as ``width`` is
    small compared to any other scale of interest (e.g. the resolution of
    the output time-delay/image bins).
    """
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    half = width / 2.0
    x_lo = jnp.clip(x0 - half, 0.0, 1.0)
    x_hi = jnp.clip(x0 + half, 0.0, 1.0)
    return UniformSlab(x_lo=x_lo, x_hi=x_hi)


class ExponentialDisk(NamedTuple):
    """Dust density falling off as exp(-x/scale_x), from the observer out to x_max.

    Mimics an exponential Galactic dust-layer profile projected along a
    line of sight that starts within the layer (x=0, the observer/Earth)
    and exits it well before the source (x_max caps the support).
    """

    scale_x: jnp.ndarray
    x_max: jnp.ndarray

    def _norm(self):
        # -expm1(-u) == 1 - exp(-u), but accurate for small u (near the
        # observer, x/scale_x -> 0) where 1 - exp(-u) cancels to noise.
        return -jnp.expm1(-self.x_max / self.scale_x)

    def cumulative_fraction(self, x):
        x = jnp.clip(x, 0.0, self.x_max)
        return -jnp.expm1(-x / self.scale_x) / self._norm()

    def inverse_cumulative_fraction(self, u):
        return -self.scale_x * jnp.log1p(-u * self._norm())


def uniform_slab(x_lo, x_hi):
    """Convenience constructor for :class:`UniformSlab`."""
    return UniformSlab(x_lo=jnp.asarray(x_lo, dtype=jnp.float32), x_hi=jnp.asarray(x_hi, dtype=jnp.float32))


def exponential_disk(scale_x, x_max=1.0):
    """Convenience constructor for :class:`ExponentialDisk`."""
    return ExponentialDisk(
        scale_x=jnp.asarray(scale_x, dtype=jnp.float32), x_max=jnp.asarray(x_max, dtype=jnp.float32)
    )
