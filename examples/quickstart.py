"""Minimal end-to-end example: trace a dust-scattering halo and print a summary.

Run with:
    python examples/quickstart.py
"""

import time

import jax
import jax.numpy as jnp

import dsh


def main():
    key = jax.random.PRNGKey(0)

    # A dust screen sitting 30-35% of the way from the observer to a source
    # 3 kpc away, with a total scattering optical depth of 0.2 at 1 keV.
    profile = dsh.uniform_slab(x_lo=0.30, x_hi=0.35)
    bin_spec = dsh.default_bin_spec(
        theta_min_arcsec=1, theta_max_arcsec=3600, n_theta=40, dt_min_s=0, dt_max_s=2e7, n_dt=40
    )

    t0 = time.time()
    result = dsh.simulate(
        key,
        n_photons=200_000,
        D_pc=3000,
        E_keV=1.0,
        tau_sca=0.2,
        profile=profile,
        bin_spec=bin_spec,
        max_scatter_order=15,
    )
    result.image.block_until_ready()
    print(f"traced 200,000 photons in {time.time() - t0:.2f} s on {jax.devices()[0]}")

    print(f"total peeled-off weight (per photon): {float(jnp.sum(result.image)):.4g}")
    print(f"fraction of photons still bouncing at max_scatter_order: "
          f"{float(result.fraction_still_active):.2%} (should be ~0)")

    radial = jnp.sum(result.image, axis=1)
    theta_arcsec = dsh.binning.theta_bin_centers(bin_spec) / dsh.constants.ARCSEC_TO_RAD
    print("\nradial profile (summed over time delay):")
    for theta, value in zip(theta_arcsec[::4], radial[::4]):
        print(f"  {float(theta):9.1f} arcsec   {float(value):.4g}")


if __name__ == "__main__":
    main()
