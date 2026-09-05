"""Render the halo as actual (x, y) sky-plane images -- what an observer's
detector would see -- at different times, and save each as a FITS image.

The dust profile depends only on line-of-sight distance, so the halo is
exactly azimuthally symmetric; dsh.sky_image() re-expresses the tracer's
(angle, time-delay) histogram on an (x, y) pixel grid via that symmetry,
optionally restricted to a time-delay window (a "snapshot"). Because
time delay scales as theta^2 (see dsh/geometry.py), a fixed *time* window
picks out a fixed *ring* of radius -- and that ring visibly expands
outward in later snapshots. This is the same "expanding ring" signature
seen in real observed dust echoes.

Plotting requires matplotlib (``pip install dsh[plot]``); FITS output
requires astropy (``pip install dsh[fits]``).

Run with:
    python examples/sky_image_demo.py
"""

import jax
import matplotlib.pyplot as plt
import numpy as np

import dsh


def main():
    key = jax.random.PRNGKey(0)

    D_pc, E_keV, tau_sca = 3000, 1.0, 0.3
    profile = dsh.uniform_slab(x_lo=0.3, x_hi=0.35)
    bin_spec = dsh.default_bin_spec(
        theta_min_arcsec=1, theta_max_arcsec=3 * 3600, n_theta=50, dt_min_s=0, dt_max_s=5e7, n_dt=50
    )

    result = dsh.simulate(
        key, n_photons=300_000, D_pc=D_pc, E_keV=E_keV, tau_sca=tau_sca,
        profile=profile, bin_spec=bin_spec, max_scatter_order=15,
    )

    extent_arcsec = 3000
    windows = [
        ("time-integrated", None, None, "sky_image_total.fits"),
        ("dt in [0, 1e6] s", 0, 1e6, "sky_image_early.fits"),
        ("dt in [2e7, 3e7] s", 2e7, 3e7, "sky_image_late.fits"),
    ]

    fig, axes = plt.subplots(1, len(windows), figsize=(5 * len(windows), 5))
    for ax, (title, dt_min, dt_max, fits_path) in zip(axes, windows):
        image2d, x, y = dsh.sky_image(
            result.image, bin_spec, extent_arcsec=extent_arcsec, npix=201, dt_min_s=dt_min, dt_max_s=dt_max
        )
        im = ax.imshow(
            np.log10(image2d + 1e-30), origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]], vmin=-6, vmax=4,
        )
        ax.set_title(title)
        ax.set_xlabel("x [arcsec]")
        ax.set_ylabel("y [arcsec]")

        try:
            dsh.save_sky_image_fits(
                image2d, extent_arcsec, fits_path,
                metadata={"D_PC": D_pc, "E_KEV": E_keV, "TAU_SCA": tau_sca},
            )
            print(f"wrote {fits_path}")
        except ImportError:
            print("astropy not installed -- skipping FITS output (pip install dsh[fits])")

    fig.colorbar(im, ax=axes, label="log10(brightness)")
    fig.savefig("sky_image_demo.png", dpi=150)
    print("wrote sky_image_demo.png")


if __name__ == "__main__":
    main()
