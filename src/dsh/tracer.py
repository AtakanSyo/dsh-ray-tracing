"""Monte Carlo multiple-scattering ray tracer for X-ray dust-scattering halos.

Algorithm
---------
Photons are launched from a point source, straight at the observer, and
undergo a real (unweighted) random walk through a plane-parallel dusty
region between the source and a point observer: free paths are sampled from
the local optical depth and, at each real scattering event, a new direction
is drawn from the grain-mixture phase function (see :mod:`dsh.scatter`,
:mod:`dsh.cross_section`).

Because the observer is an idealized point, an unbiased "did a randomly
walked photon happen to arrive exactly at the observer" estimator would have
zero efficiency. Instead this tracer uses the standard **peel-off /
next-event estimator** technique from Monte Carlo radiative transfer
(e.g. Yusef-Zadeh, Morris & White 1984; Whitney 2003, "peeling-off"): at
every real scattering event, a fraction of the photon's weight,

    w * Phi(theta_peel) * exp(-tau_slant)

is deterministically deposited into the output image, where ``theta_peel``
is the angle between the photon's incoming direction and the direction from
that scattering point to the observer, ``Phi`` is the grain-mixture phase
function, and ``tau_slant`` is the exact slant-path optical depth from that
point to the observer.

The first real scattering event needs special handling. Every photon is
launched from the point source exactly along the source-observer axis (a
photon that hasn't scattered yet cannot have a transverse position -- there
is nothing to make it deviate), so its position at the first scattering
event is always exactly on-axis. Peeling off "towards the observer" from an
on-axis point is degenerate: the direction from any on-axis point straight
to the observer is the source direction itself, i.e. zero deflection, for
*every* photon -- which would (wrongly) place all single-scattering flux at
theta=0. The resolution (standard for point-source/point-detector problems,
and equivalent to the single-scattering integral used throughout the X-ray
halo literature, e.g. Smith & Dwek 1998 eq. 2) is to exploit the profile's
translational symmetry in the transverse plane: for the first event only,
the deflection angle required to be seen at a given sky angle theta_img is
theta_img itself, independent of the (irrelevant, on-axis) transverse
position. So :func:`simulate` peels off the first event *once per output
angle bin* (an exact single-scattering calculation, Monte-Carlo-integrated
over the scattering depth only), using the exact geometric time delay from
:mod:`dsh.geometry`. From the second real scattering event onward, the
random walk has genuine (non-degenerate) transverse position and direction,
and the ordinary position-based peel-off described above applies -- this
part is a fixed-length ``jax.lax.scan``, vectorized over all photons via
plain array broadcasting (a photon that has already left the dusty region
is simply frozen/masked out of all further updates).

Known limitations (v1)
-----------------------
- Monochromatic source only; the dust column is plane-parallel with a
  density that depends only on the line-of-sight coordinate (see
  :mod:`dsh.dust_model`).
- Pure scattering (albedo = 1, no photoelectric absorption) -- grain
  extinction from true absorption is not modeled; ``weight`` is carried in
  the photon state for future extension (e.g. absorption weighting or
  Russian roulette).
- The first-scatter peel-off evaluates the phase function/time-delay at
  each output angle bin's center rather than integrating across the bin --
  fine for reasonably fine bins, but a coarse binning will show it.
- The returned image is a raw peel-off weight sum, normalized only by
  ``n_photons``: it captures the correct *shape* of the halo (radial
  profile, time evolution, single- vs multiple-scattering balance).
  :func:`dsh.binning.to_surface_brightness` converts it into an
  (arbitrarily normalized, but properly per-steradian-per-second) specific
  intensity; calibrating that to an absolute physical surface brightness
  additionally requires the source flux/fluence, which is left to the
  caller to multiply in.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import binning, cross_section, geometry, photon, scatter


class TraceResult(NamedTuple):
    image: jnp.ndarray  # (n_theta, n_dt), see BinSpec
    bin_spec: binning.BinSpec
    n_photons: int
    fraction_still_active: jnp.ndarray  # scalar diagnostic, see module docstring


def _first_scatter_image(key, n_photons, D_cm, tau_sca, profile, tables, bin_spec, rho_grain_g_cm3, mu_e):
    """Exact (per output angle bin) single-scattering contribution of the
    first real scattering event of every photon -- see module docstring.

    Returns (image_contribution, post_first_scatter_state).
    """
    k_tau, k_bin, k_theta, k_phi = jax.random.split(key, 4)

    source_state = photon.init_photons(n_photons)
    x_current = source_state.pos[..., 2]  # == 1 (fractional coordinate, see photon.py)
    mu_z = source_state.dir[..., 2]  # == -1
    u_tau = jax.random.uniform(k_tau, shape=(n_photons,))
    scatters1, x1, _delta_x = scatter.sample_next_event(u_tau, profile, tau_sca, x_current, mu_z)

    theta_img = binning.theta_bin_centers(bin_spec)  # (n_theta,)
    n_theta = theta_img.shape[0]

    phase_img = cross_section.mixture_phase_function(theta_img, tables, rho_grain_g_cm3, mu_e)  # (n_theta,)
    F_x1 = profile.cumulative_fraction(x1)  # (n_photons,)
    tau_slant = tau_sca * F_x1[:, None] / jnp.cos(theta_img)[None, :]  # (n_photons, n_theta)
    dt_img = geometry.time_delay(x1[:, None], theta_img[None, :], D_cm)  # (n_photons, n_theta)

    contrib = jnp.where(scatters1[:, None], phase_img[None, :] * jnp.exp(-tau_slant), 0.0)

    de = bin_spec.dt_edges
    dt_idx = jnp.clip(jnp.searchsorted(de, dt_img, side="right") - 1, 0, de.shape[0] - 2)
    dt_in_range = (dt_img >= de[0]) & (dt_img <= de[-1])
    contrib = jnp.where(dt_in_range, contrib, 0.0)
    theta_idx = jnp.broadcast_to(jnp.arange(n_theta), dt_idx.shape)

    image = binning.zeros(bin_spec)
    image = image.at[theta_idx.reshape(-1), dt_idx.reshape(-1)].add(contrib.reshape(-1))

    # Real (unweighted) direction/position update, to seed the ongoing random walk.
    bin_idx = scatter.sample_grain_bin_index(k_bin, tables, n_photons)
    theta1 = scatter.sample_scatter_angle(k_theta, tables, bin_idx)
    phi1 = jax.random.uniform(k_phi, shape=(n_photons,)) * 2.0 * jnp.pi
    dir1 = scatter.rotate_direction(source_state.dir, theta1, phi1)

    pos1 = jnp.zeros((n_photons, 3), dtype=jnp.float32).at[:, 2].set(x1)
    # The first leg is exactly axial (mu_z == -1 exactly), so it contributes
    # zero path-length excess by construction -- see photon.py / geometry.py.
    excess_path_frac1 = jnp.zeros((n_photons,), dtype=jnp.float32)

    state1 = photon.PhotonState(
        pos=pos1,
        dir=dir1,
        weight=source_state.weight,
        excess_path_frac=excess_path_frac1,
        active=scatters1,
        order=scatters1.astype(jnp.int32),
    )
    return image, state1


def _make_step_fn(tables, profile, tau_sca, D_cm, bin_spec, master_key, rho_grain_g_cm3, mu_e):
    """Ordinary position-based peel-off step, valid from the 2nd real scatter onward
    (see module docstring for why the 1st event needs different treatment)."""

    def step(carry, step_idx):
        state, image = carry
        n = state.pos.shape[0]

        key_step = jax.random.fold_in(master_key, step_idx)
        k_tau, k_bin, k_theta, k_phi = jax.random.split(key_step, 4)

        x_current = state.pos[..., 2]  # fractional coordinate, see photon.py
        mu_z = state.dir[..., 2]
        u_tau = jax.random.uniform(k_tau, shape=(n,))
        scatters, _x_next, delta_x = scatter.sample_next_event(u_tau, profile, tau_sca, x_current, mu_z)
        effective_scatters = scatters & state.active

        mu_z_safe = scatter.safe_mu_z(mu_z)
        delta_s = delta_x / mu_z_safe  # fractional arc length of this leg (units of D_cm)

        new_pos = jnp.where(state.active[..., None], state.pos + delta_s[..., None] * state.dir, state.pos)
        leg_excess = geometry.leg_excess_path_fraction(delta_s, state.dir)
        new_excess_path_frac = jnp.where(
            state.active, state.excess_path_frac + leg_excess, state.excess_path_frac
        )

        # --- peel-off: deposit this real scattering event's contribution to the image ---
        theta_sky, _phi_sky, dt, R = geometry.sky_observables(new_pos, new_excess_path_frac, D_cm)
        n_obs = -new_pos / R[..., None]
        cos_theta_peel = jnp.sum(state.dir * n_obs, axis=-1)
        theta_peel = jnp.arccos(jnp.clip(cos_theta_peel, -1.0, 1.0))
        phase = cross_section.mixture_phase_function(theta_peel, tables, rho_grain_g_cm3, mu_e)

        x_r = new_pos[..., 2]
        F_r = profile.cumulative_fraction(x_r)
        cos_theta_sky = jnp.clip(new_pos[..., 2] / R, 1e-6, None)
        tau_slant = tau_sca * F_r / cos_theta_sky
        atten = jnp.exp(-tau_slant)

        contrib = jnp.where(effective_scatters, state.weight * phase * atten, 0.0)
        image = binning.accumulate(image, theta_sky, dt, contrib, bin_spec)

        # --- sample the new outgoing direction for photons that really scattered ---
        bin_idx = scatter.sample_grain_bin_index(k_bin, tables, n)
        theta_scat = scatter.sample_scatter_angle(k_theta, tables, bin_idx)
        phi_scat = jax.random.uniform(k_phi, shape=(n,)) * 2.0 * jnp.pi
        new_dir_scatter = scatter.rotate_direction(state.dir, theta_scat, phi_scat)
        new_dir = jnp.where(effective_scatters[..., None], new_dir_scatter, state.dir)

        new_state = photon.PhotonState(
            pos=new_pos,
            dir=new_dir,
            weight=state.weight,
            excess_path_frac=new_excess_path_frac,
            active=effective_scatters,
            order=state.order + effective_scatters.astype(state.order.dtype),
        )
        return (new_state, image), None

    return step


def simulate(
    key,
    n_photons,
    D_pc,
    E_keV,
    tau_sca,
    profile,
    bin_spec: binning.BinSpec,
    grain_distribution=None,
    max_scatter_order=20,
    n_grain_bins=32,
    n_theta_table=2001,
    rho_grain_g_cm3=3.0,
    mu_e=2.0,
):
    """Run the multiple-scattering Monte Carlo halo simulation.

    Parameters
    ----------
    key : jax.random.PRNGKey
    n_photons : number of photon packets to trace
    D_pc : source-observer distance [pc]
    E_keV : monochromatic photon energy [keV]
    tau_sca : total scattering optical depth of the dust column
    profile : a dust_model profile (e.g. :func:`dsh.dust_model.uniform_slab`)
    bin_spec : output image binning, see :mod:`dsh.binning`
    grain_distribution : a dsh.grains distribution; defaults to standard MRN
    max_scatter_order : max real scatters per photon (>= 1); the first is
        handled by the exact per-bin single-scattering step, the remaining
        ``max_scatter_order - 1`` by the scanned multiple-scattering loop
    n_grain_bins, n_theta_table : grain-size/scattering-table resolution,
        see :func:`dsh.cross_section.build_scattering_tables`
    rho_grain_g_cm3, mu_e : Drude grain-material parameters, see
        :func:`dsh.cross_section.refractive_index_drude`

    Returns
    -------
    TraceResult
    """
    from . import constants
    from . import grains as grains_mod

    if grain_distribution is None:
        grain_distribution = grains_mod.mrn_distribution()
    if max_scatter_order < 1:
        raise ValueError("max_scatter_order must be >= 1")

    D_cm = jnp.asarray(D_pc * constants.PC_TO_CM, dtype=jnp.float32)
    tau_sca = jnp.asarray(tau_sca, dtype=jnp.float32)

    tables = cross_section.build_scattering_tables(
        grain_distribution, E_keV, n_bins=n_grain_bins, n_theta=n_theta_table
    )

    key_first, key_scan = jax.random.split(key)
    image1, state1 = _first_scatter_image(
        key_first, n_photons, D_cm, tau_sca, profile, tables, bin_spec, rho_grain_g_cm3, mu_e
    )

    n_more_steps = max_scatter_order - 1
    if n_more_steps > 0:
        step_fn = _make_step_fn(tables, profile, tau_sca, D_cm, bin_spec, key_scan, rho_grain_g_cm3, mu_e)
        (final_state, final_image), _ = jax.lax.scan(
            step_fn, (state1, image1), jnp.arange(n_more_steps)
        )
    else:
        final_state, final_image = state1, image1

    return TraceResult(
        image=final_image / n_photons,
        bin_spec=bin_spec,
        n_photons=n_photons,
        fraction_still_active=jnp.mean(final_state.active.astype(jnp.float32)),
    )
