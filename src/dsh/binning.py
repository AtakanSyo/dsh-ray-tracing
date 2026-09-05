"""2D (sky angle, time delay) histogram accumulation for the halo image."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .constants import ARCSEC_TO_RAD, RAD_TO_ARCSEC


class BinSpec(NamedTuple):
    theta_edges: jnp.ndarray  # (n_theta + 1,) [rad], increasing
    dt_edges: jnp.ndarray  # (n_dt + 1,) [s], increasing

    @property
    def shape(self):
        return (self.theta_edges.shape[0] - 1, self.dt_edges.shape[0] - 1)


def log_theta_bins(theta_min_arcsec, theta_max_arcsec, n_bins):
    """Log-spaced sky-angle bin edges, given in arcsec, returned in radians."""
    edges_arcsec = np.geomspace(theta_min_arcsec, theta_max_arcsec, n_bins + 1)
    return jnp.asarray(edges_arcsec * ARCSEC_TO_RAD, dtype=jnp.float32)


def linear_dt_bins(dt_min_s, dt_max_s, n_bins):
    """Linear time-delay bin edges [s]."""
    return jnp.asarray(np.linspace(dt_min_s, dt_max_s, n_bins + 1), dtype=jnp.float32)


def zeros(bin_spec: BinSpec):
    return jnp.zeros(bin_spec.shape, dtype=jnp.float32)


def theta_bin_centers(bin_spec: BinSpec):
    """Geometric-mean center [rad] of each sky-angle bin (matches log spacing)."""
    e = bin_spec.theta_edges
    return jnp.sqrt(e[:-1] * e[1:])


def theta_bin_solid_angles(bin_spec: BinSpec):
    """Exact solid angle [sr] of the annulus between consecutive theta edges,
    ``2*pi*(cos(theta_lo) - cos(theta_hi))`` -- for converting an
    azimuthally-integrated (theta, dt) histogram into a per-steradian
    intensity via :func:`to_surface_brightness`.

    Real halo bins are arcsec-arcmin scale, where ``cos(theta_lo)`` and
    ``cos(theta_hi)`` both round to exactly 1.0 in float32 (any angle below
    ~4e-4 rad has ``theta^2/2`` under the float32 epsilon), which would make
    a direct ``cos(lo) - cos(hi)`` silently vanish to zero. The
    product-to-sum identity ``cos(a) - cos(b) = 2*sin((a+b)/2)*sin((b-a)/2)``
    avoids that cancellation entirely.
    """
    e = bin_spec.theta_edges
    lo, hi = e[:-1], e[1:]
    return 4.0 * jnp.pi * jnp.sin((lo + hi) / 2.0) * jnp.sin((hi - lo) / 2.0)


def to_surface_brightness(image, bin_spec: BinSpec):
    """Convert a raw (per-photon, per-bin) peel-off image into an approximate
    specific intensity: weight per steradian per second, by dividing each
    ring by its solid angle and each time bin by its width.

    Assumes (and only makes sense for) an azimuthally-symmetric problem,
    which is what every dust profile in :mod:`dsh.dust_model` describes.
    """
    solid_angle = theta_bin_solid_angles(bin_spec)  # (n_theta,)
    dt_width = jnp.diff(bin_spec.dt_edges)  # (n_dt,)
    return image / (solid_angle[:, None] * dt_width[None, :])


def dt_window_sum(image, bin_spec: BinSpec, dt_min_s=None, dt_max_s=None):
    """Collapse the (theta, dt) image over time delay into a 1D radial profile.

    With no bounds, this is the time-integrated (i.e. total) halo profile.
    Pass ``dt_min_s``/``dt_max_s`` to instead select a single "snapshot":
    a narrow window (comparable to one ``bin_spec.dt_edges`` bin width) for
    "what the halo looks like at time delay ~t", or a wider one for "what
    the halo looks like accumulated over a texposure window".
    """
    dt_centers = 0.5 * (bin_spec.dt_edges[:-1] + bin_spec.dt_edges[1:])
    mask = jnp.ones_like(dt_centers, dtype=bool)
    if dt_min_s is not None:
        mask &= dt_centers >= dt_min_s
    if dt_max_s is not None:
        mask &= dt_centers <= dt_max_s
    return jnp.sum(jnp.where(mask[None, :], image, 0.0), axis=1)


def sky_image(image, bin_spec: BinSpec, extent_arcsec, npix=201, dt_min_s=None, dt_max_s=None):
    """Render the halo as an actual (x, y) sky-plane image, in arcsec --
    what an observer's detector would see -- rather than a 1D radial profile.

    Every dust profile in :mod:`dsh.dust_model` depends only on the
    line-of-sight coordinate, so the physical halo is exactly azimuthally
    symmetric about the source direction: there is no information in the
    azimuthal angle to simulate, and this function does not fabricate any
    -- it takes the radial (and, optionally, time-windowed -- see
    :func:`dt_window_sum`) profile already computed by the tracer and
    re-expresses it on a square pixel grid via ``I(x, y) = I(r)`` with
    ``r = sqrt(x^2 + y^2)``, interpolating log-linearly in radius (matching
    the log-spaced angle bins). Pixels outside the tabulated angular range
    (including the central pixel, if it falls inside the innermost bin
    edge -- e.g. within the unresolved central point source) are 0.

    Parameters
    ----------
    image : the (n_theta, n_dt) image (e.g. ``result.image``), or an
        already time-collapsed 1D array of length n_theta (from
        :func:`dt_window_sum` or :func:`to_surface_brightness` summed
        yourself) -- either is accepted.
    bin_spec : the image's binning.
    extent_arcsec : half-width of the square field of view [arcsec]; the
        output spans ``[-extent_arcsec, +extent_arcsec]`` in both x and y.
    npix : output image is ``npix x npix``.
    dt_min_s, dt_max_s : optional time-delay window, see :func:`dt_window_sum`
        -- ignored if ``image`` is already 1D.

    Returns
    -------
    image2d : (npix, npix) array, sky brightness.
    x_arcsec, y_arcsec : (npix,) pixel-center coordinate arrays.
    """
    radial = image if jnp.ndim(image) == 1 else dt_window_sum(image, bin_spec, dt_min_s, dt_max_s)
    theta_centers_arcsec = np.asarray(theta_bin_centers(bin_spec)) * RAD_TO_ARCSEC
    radial = np.asarray(radial)

    x_arcsec = np.linspace(-extent_arcsec, extent_arcsec, npix)
    y_arcsec = np.linspace(-extent_arcsec, extent_arcsec, npix)
    xx, yy = np.meshgrid(x_arcsec, y_arcsec)
    r = np.sqrt(xx**2 + yy**2)

    # log-linear interpolation, matching the log-spaced theta bins; floor r
    # before taking its log (rather than clipping into range) so pixels
    # genuinely below the smallest resolved angle -- including r=0 exactly,
    # at the field center -- correctly fall on the "left" side and read 0,
    # rather than being silently assigned the innermost bin's value.
    log_r = np.log(np.maximum(r, theta_centers_arcsec[0] * 1e-6))
    log_theta = np.log(theta_centers_arcsec)
    image2d = np.interp(log_r, log_theta, radial, left=0.0, right=0.0).astype(np.float32)

    return image2d, x_arcsec, y_arcsec


def save_sky_image_fits(image2d, extent_arcsec, path, metadata=None, overwrite=True):
    """Write a 2D (x, y) sky image from :func:`sky_image` as a FITS file
    with a proper (linear, arcsec-offset) WCS -- unlike :func:`save_fits`,
    an (x, y) sky image has genuinely linear axes, so a standard WCS is the
    right (and DS9/astropy-viewer-friendly) way to describe it. There is no
    real celestial pointing here (this is a simulation centered on the
    source), so the axes are offsets in arcsec from the source, not RA/Dec.

    Requires ``astropy`` (``pip install dsh[fits]``).
    """
    from astropy.io import fits

    npix = image2d.shape[0]
    header = fits.Header()
    header["BUNIT"] = ("weight/photon/bin", "raw peeled-off weight per photon per bin")
    header["WCSAXES"] = 2
    header["CTYPE1"] = "LINEAR"
    header["CTYPE2"] = "LINEAR"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = (npix + 1) / 2.0
    header["CRPIX2"] = (npix + 1) / 2.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 2.0 * extent_arcsec / (npix - 1)
    header["CDELT2"] = 2.0 * extent_arcsec / (npix - 1)
    header["COMMENT"] = "axes are arcsec offsets from the source, not celestial RA/Dec"
    for key, value in (metadata or {}).items():
        header[key] = value

    fits.PrimaryHDU(data=np.asarray(image2d, dtype=np.float32), header=header).writeto(
        path, overwrite=overwrite
    )


def save_fits(image, bin_spec: BinSpec, path, metadata=None, overwrite=True):
    """Write a (sky angle, time delay) halo image to a FITS file.

    Requires ``astropy`` (``pip install dsh[fits]``).

    The image is stored as the primary HDU's data array. Its axes are
    *not* encoded as a standard FITS/WCS coordinate system, because
    :func:`log_theta_bins` makes the sky-angle axis log-spaced and FITS WCS
    has no standard linear-in-log axis type that would represent that
    faithfully; forcing it into ``CDELT``/``CRVAL`` keywords would silently
    mislabel the bin centers. Instead the *exact* bin edges are stored
    losslessly in two accompanying binary table extensions, ``THETA_EDGES``
    (radians) and ``DT_EDGES`` (seconds) -- reconstruct bin centers with
    :func:`theta_bin_centers` or by averaging consecutive ``DT_EDGES``
    entries. :func:`load_fits` reads a file written this way back into an
    ``(image, bin_spec)`` pair.

    Parameters
    ----------
    image : the 2D (n_theta, n_dt) image array (raw peel-off weights from
        :attr:`dsh.tracer.TraceResult.image`, or the output of
        :func:`to_surface_brightness` -- update ``metadata["BUNIT"]``
        accordingly if you pass the latter).
    bin_spec : the image's binning, see :class:`BinSpec`.
    path : output file path (``.fits``).
    metadata : optional dict of extra FITS header keywords, e.g.
        ``{"D_PC": 3000, "TAU_SCA": 0.2, "E_KEV": 1.0, "NPHOTON": 200_000}``.
    overwrite : whether to overwrite an existing file at ``path``.
    """
    from astropy.io import fits

    header = fits.Header()
    header["BUNIT"] = ("weight/photon/bin", "raw peeled-off weight per photon per bin")
    header["AXIS1"] = ("dt [s], see DT_EDGES ext.", "NAXIS1, fastest-varying axis")
    header["AXIS2"] = ("theta [rad], see THETA_EDGES ext.", "NAXIS2, log-spaced")
    for key, value in (metadata or {}).items():
        header[key] = value

    primary = fits.PrimaryHDU(data=np.asarray(image, dtype=np.float32), header=header)
    theta_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="theta_edges_rad", format="D", array=np.asarray(bin_spec.theta_edges, dtype=np.float64))],
        name="THETA_EDGES",
    )
    dt_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="dt_edges_s", format="D", array=np.asarray(bin_spec.dt_edges, dtype=np.float64))],
        name="DT_EDGES",
    )
    fits.HDUList([primary, theta_hdu, dt_hdu]).writeto(path, overwrite=overwrite)


def load_fits(path):
    """Read a FITS file written by :func:`save_fits` back into ``(image, bin_spec)``."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float32)
        theta_edges = np.asarray(hdul["THETA_EDGES"].data["theta_edges_rad"], dtype=np.float32)
        dt_edges = np.asarray(hdul["DT_EDGES"].data["dt_edges_s"], dtype=np.float32)
    bin_spec = BinSpec(theta_edges=jnp.asarray(theta_edges), dt_edges=jnp.asarray(dt_edges))
    return image, bin_spec


def accumulate(image, theta_sky, dt, weights, bin_spec: BinSpec):
    """Scatter-add ``weights`` into ``image`` at (theta_sky, dt), out-of-range safe."""
    te, de = bin_spec.theta_edges, bin_spec.dt_edges
    i = jnp.clip(jnp.searchsorted(te, theta_sky, side="right") - 1, 0, te.shape[0] - 2)
    j = jnp.clip(jnp.searchsorted(de, dt, side="right") - 1, 0, de.shape[0] - 2)
    in_range = (
        (theta_sky >= te[0]) & (theta_sky <= te[-1]) & (dt >= de[0]) & (dt <= de[-1])
    )
    w = jnp.where(in_range, weights, 0.0)
    return image.at[i, j].add(w)
