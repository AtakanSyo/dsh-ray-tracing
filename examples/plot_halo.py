"""Trace a dust-scattering halo, plot it, and save it as a FITS image.

Plotting requires matplotlib (``pip install dsh[plot]``); the FITS output
requires astropy (``pip install dsh[fits]``) and is skipped with a warning
if astropy isn't installed.

Run with:
    python examples/plot_halo.py [output.png]
"""

import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

import dsh


def main(out_path="halo.png"):
    key = jax.random.PRNGKey(0)

    D_pc = 3000
    E_keV = 1.0
    tau_sca = 0.3
    max_scatter_order = 15

    profile = dsh.uniform_slab(x_lo=0.3, x_hi=0.35)
    bin_spec = dsh.default_bin_spec(
        theta_min_arcsec=1, theta_max_arcsec=3 * 3600, n_theta=50, dt_min_s=0, dt_max_s=5e7, n_dt=50
    )

    n_photons = 300_000
    result = dsh.simulate(
        key,
        n_photons=n_photons,
        D_pc=D_pc,
        E_keV=E_keV,
        tau_sca=tau_sca,
        profile=profile,
        bin_spec=bin_spec,
        max_scatter_order=max_scatter_order,
    )

    fits_path = Path(out_path).with_suffix(".fits")
    try:
        dsh.save_fits(
            result.image,
            bin_spec,
            fits_path,
            metadata={
                "D_PC": D_pc,
                "E_KEV": E_keV,
                "TAU_SCA": tau_sca,
                "NPHOTON": n_photons,
                "MAXORDER": max_scatter_order,
            },
        )
        print(f"wrote {fits_path}")
    except ImportError:
        print("astropy not installed -- skipping FITS output (pip install dsh[fits])")

    image = np.asarray(result.image)
    theta_arcsec = np.asarray(dsh.binning.theta_bin_centers(bin_spec)) / dsh.constants.ARCSEC_TO_RAD
    dt_centers = 0.5 * (np.asarray(bin_spec.dt_edges[:-1]) + np.asarray(bin_spec.dt_edges[1:]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].loglog(theta_arcsec, image.sum(axis=1) + 1e-30)
    axes[0].set_xlabel("sky angle [arcsec]")
    axes[0].set_ylabel("peeled-off weight (per photon)")
    axes[0].set_title("radial profile")

    axes[1].plot(dt_centers, image.sum(axis=0))
    axes[1].set_xlabel("time delay [s]")
    axes[1].set_ylabel("peeled-off weight (per photon)")
    axes[1].set_title("halo light curve")

    im = axes[2].pcolormesh(
        dt_centers, theta_arcsec, np.log10(image + 1e-30), shading="auto"
    )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("time delay [s]")
    axes[2].set_ylabel("sky angle [arcsec]")
    axes[2].set_title("log10(image)")
    fig.colorbar(im, ax=axes[2])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "halo.png")
