"""Trace a dust-scattering halo and plot its radial profile, light curve, and
2D (angle, time) image. Requires matplotlib (``pip install dsh[plot]``).

Run with:
    python examples/plot_halo.py [output.png]
"""

import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import dsh


def main(out_path="halo.png"):
    key = jax.random.PRNGKey(0)

    profile = dsh.uniform_slab(x_lo=0.3, x_hi=0.35)
    bin_spec = dsh.default_bin_spec(
        theta_min_arcsec=1, theta_max_arcsec=3 * 3600, n_theta=50, dt_min_s=0, dt_max_s=5e7, n_dt=50
    )

    result = dsh.simulate(
        key,
        n_photons=300_000,
        D_pc=3000,
        E_keV=1.0,
        tau_sca=0.3,
        profile=profile,
        bin_spec=bin_spec,
        max_scatter_order=15,
    )

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
