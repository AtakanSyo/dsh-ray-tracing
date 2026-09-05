"""Source - dust - observer geometry: exact time delays, computed stably.

Setup: a point source sits on the optical axis at distance ``D`` from a
point observer at the origin. Dust lies somewhere along/near the axis
between them. For a photon that scatters exactly once, at fractional
distance ``x = l/D`` from the *observer* (``x=0`` at the observer, ``x=1``
at the source), into an angle ``theta`` on the sky (as seen by the
observer), the exact path-length excess over the direct line gives a time
delay of

    dt(x, theta) = (D/c) * [ x/cos(theta)
                              + (1-x) * sqrt(1 + (x/(1-x))^2 * tan(theta)^2)
                              - 1 ]

Derivation (self-contained, no small-angle assumption): place the observer
at the origin O, the source at S=(0,0,D). A photon that scatters once, at a
point P offset by transverse distance b from the axis at longitudinal
coordinate l=xD (so P=(b,0,l)), is seen by the observer at sky angle theta
with tan(theta) = b/l. The scattered path length is
    |S-P| + |P-O| = sqrt(b^2+(D-l)^2) + sqrt(b^2+l^2)
Substituting b = l*tan(theta) = xD*tan(theta) and factoring out D gives
exactly the formula above. In the small-angle limit (theta -> 0, appropriate
for real X-ray dust halos where theta ~ arcmin) this reduces to the familiar

    dt(x, theta) ~= (D/(2c)) * theta^2 * x/(1-x)

Numerical note: dt is a *tiny* fractional correction (order theta^2, and
real halo angles are arcsec-arcmin scale) on top of an O(1) geometric
factor -- computing it as a naive "big term minus big term" difference
loses essentially all precision in float32 (it can silently round straight
to exactly 0 for realistic angles). Every function here is instead written
using cancellation-safe identities (``1 - cos(u) = 2*sin^2(u/2)``,
``sqrt(1+u) - 1 = u/(sqrt(1+u)+1)``, and the difference-of-squares form
``a - b = (a^2 - b^2)/(a + b)``) so the small quantity is built up directly
rather than recovered from a catastrophic subtraction. Positions are also
kept as *fractions of D* (dimensionless, order 1) rather than centimeters:
D itself is astronomical (~1e21 cm for a kpc-scale source), and squaring
that in float32 (as any ``sqrt(x^2+y^2+z^2)`` distance formula must)
overflows outright (``(1e21)^2 = 1e42 >> 3.4e38``, the float32 max).
"""

import jax.numpy as jnp

from .constants import C_LIGHT_CM_S


def time_delay(x, theta_rad, D_cm):
    """Exact single-scattering time delay [s], computed cancellation-safely.

    Parameters
    ----------
    x : fractional distance of the scattering point from the observer,
        in (0, 1) -- 0 is at the observer, 1 is at the source.
    theta_rad : observed scattering angle on the sky [rad].
    D_cm : source-observer distance [cm].
    """
    x = jnp.asarray(x)
    theta_rad = jnp.asarray(theta_rad)

    # x/cos(theta) - x == x*(sec(theta) - 1) == x * 2*sin(theta/2)^2/cos(theta)
    sec_m1 = 2.0 * jnp.sin(theta_rad / 2.0) ** 2 / jnp.cos(theta_rad)
    # (1-x)*sqrt(1+u) - (1-x) == (1-x) * (sqrt(1+u) - 1) == (1-x) * u/(sqrt(1+u)+1)
    ratio = x / (1.0 - x)
    u = (ratio * jnp.tan(theta_rad)) ** 2
    sqrt_m1 = u / (jnp.sqrt(1.0 + u) + 1.0)

    excess = x * sec_m1 + (1.0 - x) * sqrt_m1
    return (D_cm / C_LIGHT_CM_S) * excess


def time_delay_small_angle(x, theta_rad, D_cm):
    """Small-angle limit of :func:`time_delay`: dt ~= (D/2c) * theta^2 * x/(1-x)."""
    x = jnp.asarray(x)
    theta_rad = jnp.asarray(theta_rad)
    return (D_cm / (2.0 * C_LIGHT_CM_S)) * theta_rad**2 * (x / (1.0 - x))


def axis_deflection_angle(dir_vec):
    """Angle [rad] between unit vector(s) ``dir_vec`` and the -z axis
    (the direction from the source straight at the observer).

    Uses ``arctan2`` rather than ``arccos(-dir_z)``: the latter has infinite
    derivative sensitivity near dir_z=-1 (i.e. exactly the "barely
    deflected" case that matters most here), amplifying small floating
    point errors into large angle errors.
    """
    transverse = jnp.sqrt(dir_vec[..., 0] ** 2 + dir_vec[..., 1] ** 2)
    return jnp.arctan2(transverse, -dir_vec[..., 2])


def leg_excess_path_fraction(delta_s_frac, dir_vec):
    """Stable per-leg contribution to the accumulated excess path length.

    A leg of (fractional, i.e. in units of D_cm) arc length ``delta_s_frac``
    traveled along ``dir_vec`` (making angle ``phi`` with the -z axis, see
    :func:`axis_deflection_angle`) covers only ``delta_s_frac*cos(phi)`` of
    axial (towards-the-observer) distance; the difference,
    ``delta_s_frac*(1-cos(phi)) = delta_s_frac*2*sin(phi/2)^2``, is this
    leg's contribution to the photon's total path-length excess over a
    direct line -- computed via the stable half-angle identity rather than
    ``1 - cos(phi)`` directly.
    """
    phi = axis_deflection_angle(dir_vec)
    return delta_s_frac * 2.0 * jnp.sin(phi / 2.0) ** 2


def sky_observables(pos_frac, excess_path_frac, D_cm):
    """Convert a photon's (fractional) position and accumulated path-length
    excess into the observable (sky angle, time delay) for a photon that
    travels straight from that position to the observer at the origin.

    Parameters
    ----------
    pos_frac : array (..., 3), position in units of D_cm (dimensionless),
        source-observer axis along +z (source at z=1, observer at z=0).
    excess_path_frac : array (...), accumulated real path length so far, in
        units of D_cm, *in excess of* the axial (z) distance covered (see
        :func:`leg_excess_path_fraction`) -- i.e. 0 for a photon that has
        only ever traveled exactly along the axis.
    D_cm : source-observer distance [cm].

    Returns
    -------
    theta_sky : halo radius on sky [rad].
    phi_sky : azimuthal angle on sky [rad].
    dt : time delay relative to the direct (unscattered) light-travel time
        D/c [s].
    R_frac : straight-line distance from this position to the observer, in
        units of D_cm (dimensionless).
    """
    x, y, z = pos_frac[..., 0], pos_frac[..., 1], pos_frac[..., 2]
    rho = jnp.sqrt(x**2 + y**2)
    R_frac = jnp.sqrt(rho**2 + z**2)
    theta_sky = jnp.arctan2(rho, z)
    phi_sky = jnp.arctan2(y, x)

    # The final leg (this position -> observer) hasn't been "traveled" yet;
    # its own excess over the axial distance z is R_frac - z, computed via
    # the difference-of-squares identity a-b = (a^2-b^2)/(a+b) rather than
    # a direct (cancellation-prone) subtraction.
    final_leg_excess = rho**2 / (R_frac + z)

    dt = (D_cm / C_LIGHT_CM_S) * (excess_path_frac + final_leg_excess)
    return theta_sky, phi_sky, dt, R_frac
