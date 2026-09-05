# dsh-ray-tracing

A differentiable, GPU-accelerated **Monte Carlo ray tracer for X-ray dust-scattering halos**, built on [JAX](https://github.com/google/jax).

When a bright X-ray source (a GRB, an X-ray binary, an AGN flare, ...) shines through interstellar dust, small-angle scattering off the dust grains spreads some of its light into a fading ring around the source that grows and dims over minutes to months — a *dust-scattering halo* (or "dust echo"). This package simulates that process from first principles: real grain physics (Rayleigh-Gans scattering, an MRN size distribution), a real Monte Carlo photon random walk through the dust (including full multiple scattering), and an exact source-dust-observer time-delay geometry — all written in JAX so it runs, unmodified, on CPU or GPU, and every physical parameter is differentiable.

```
python examples/quickstart.py
```

## Why this exists

Most public dust-scattering-halo codes (e.g. [`eblur/dust`](https://github.com/eblur/dust)) compute the **single**-scattering halo semi-analytically, which is the right approximation for the optically-thin sightlines most halos are observed through. This package instead runs a genuine multiple-scattering Monte Carlo photon transport, vectorized entirely as array operations so it scales to GPUs — useful once you care about denser sightlines, want a from-scratch differentiable simulator to fit into a larger JAX-based inference pipeline (`grad`/`vmap` straight through the dust geometry and grain parameters), or just want to see multiple scattering's contribution to the halo directly rather than assume it away.

## Physics

### Scattering cross section

X-ray scattering off a dust grain satisfies the two conditions for the **Rayleigh-Gans (RG) approximation** essentially everywhere in the ISM: the grain's refractive index is very close to 1, and `k·a·|m-1| << 1`. The exact RG differential cross section (Mauche & Gorenstein 1986, *ApJ* 303, 569; Smith & Dwek 1998, *ApJ* 503, 831) is used directly — not the more common small-angle Gaussian approximation to it — so diffraction structure at larger angles is retained:

```
dσ/dΩ(E, a, θ) = 2 a² (2πa/λ)⁴ |m-1|² (j₁(y)/y)² (1 + cos²θ),      y = (4πa/λ) sin(θ/2)
```

`j₁` (spherical Bessel function) is written in closed elementary form (`sin`/`cos`), so this needs no special-function library and is fully `jit`/`vmap`/`grad`-friendly. The complex refractive index uses the **Drude (free-electron) approximation**, `|m-1| ≈ n_e r_e λ²/2π`, parameterized by a grain material density and mean mass per electron (defaults: 3 g/cm³, 2 — typical astronomical silicate/graphite); swap in tabulated optical constants later by replacing `cross_section.refractive_index_drude`.

Everything downstream — the total cross section, the angular phase function, the size-mixture averaging — is derived from that *one* formula by numerical quadrature, rather than from separately-memorized closed-form totals, so the pieces stay internally consistent.

### Grain size distribution

The standard **MRN** distribution (Mathis, Rumpl & Nordsieck 1977): `dn/da ∝ a⁻³·⁵` for `0.005 μm ≤ a ≤ 0.25 μm`. Only the *shape* of the distribution matters here — it sets the mix of grain sizes a photon can scatter off — so no dust-to-gas mass ratio or absolute grain abundance is needed; the dust column's absolute normalization is a separate, directly-specified optical depth (see below).

### Dust geometry

The dust column between the point source and point observer is described by its *normalized cumulative optical depth* along the line of sight, decoupled from its absolute scale (`tau_sca`, supplied by the caller — a parameter you'd fit to data anyway). Built-in profiles: a uniform slab, a thin screen, and an exponential (Galactic-disk-like) layer; anything else is a ~10-line `NamedTuple` with a `cumulative_fraction`/`inverse_cumulative_fraction` pair (see `dsh/dust_model.py`).

### Monte Carlo transport: the peel-off estimator

Photons are launched from the source and undergo a **real, unweighted random walk**: free paths sampled from the local optical depth, new directions sampled from the grain-mixture phase function at each real scattering event. But the observer is a mathematical point — an unbiased "did a randomly-walked photon happen to land exactly on the observer" estimator would have zero efficiency. So this uses the standard Monte Carlo radiative transfer **peel-off / next-event estimator** technique (Yusef-Zadeh, Morris & White 1984; Whitney 2003): at every real scattering event, a fraction of the photon's weight,

```
w · Φ(θ_peel) · exp(-τ_slant)
```

is deterministically deposited into the output image, where `θ_peel` is the angle between the photon's incoming direction and the direction from that scattering point straight to the observer, `Φ` is the grain-mixture phase function, and `τ_slant` is the exact slant-path optical depth back to the observer.

The very first scattering event needs special handling: every photon is launched exactly along the source-observer axis (there's nothing to deflect it before it scatters even once), so its position at the first event is always exactly on-axis, and peeling off "toward the observer" from an on-axis point is degenerate (zero deflection, for every photon, which would wrongly pile all single-scattering flux at θ=0). The fix — standard for point-source/point-detector problems, and equivalent to the single-scattering integral used throughout the literature — exploits the dust profile's translational symmetry: for the first event only, the deflection angle needed to be seen at sky angle `θ_img` is just `θ_img` itself, independent of position. So the first event is peeled off once per output angle bin (an exact, Monte-Carlo-over-scattering-depth-only single-scattering calculation); every subsequent real scattering event uses the ordinary position-based peel-off above, since by then the walk carries genuine (non-degenerate) off-axis information. See the `dsh/tracer.py` module docstring for the full derivation.

The whole loop is a single fixed-length `jax.lax.scan` over scattering orders, vectorized across all photons via plain array broadcasting — the standard way to put a variable-length random walk on a GPU: a photon that has left the dusty region is simply frozen (masked out of all further updates) for the remaining steps.

### Exact geometry, computed stably

The source-dust-observer time delay is derived from scratch (law of cosines, no small-angle assumption) in `dsh/geometry.py`:

```
δt(x, θ) = (D/c) · [ x/cos θ + (1-x)·√(1 + (x/(1-x))²tan²θ) − 1 ]
```

for a scattering point at fractional distance `x` from the observer (0 = observer, 1 = source) seen at sky angle `θ`; its small-angle limit is the familiar `δt ≈ (D/2c)·θ²·x/(1-x)`.

This delay is a *tiny* correction (order `θ²`, and real halo angles are arcsec-arcmin scale) riding on an O(1) geometric factor. Two numerical traps fall directly out of that, and both are handled explicitly rather than left as float32 landmines:

- **Cancellation**: computing `x/cosθ + (1-x)·√(...) − 1` as a literal "big term minus big term" loses essentially all precision in float32 — for realistic angles it can silently round straight to exactly 0. Every delay formula here instead uses cancellation-safe identities (`1-cos(u) = 2sin²(u/2)`, `√(1+u)-1 = u/(√(1+u)+1)`, and the difference-of-squares form `a-b = (a²-b²)/(a+b)`), including for the *running* multi-scattering path-length excess, which is accumulated incrementally in small stable steps rather than recovered by subtracting two large numbers at the end.
- **Overflow**: a source at a kpc-scale distance has `D ~ 10²¹ cm`; squaring that (as any `√(x²+y²+z²)` distance formula must) overflows float32 outright (`(10²¹)² = 10⁴² ≫ 3.4×10³⁸`). Positions are therefore tracked as *fractions of D* (dimensionless, order 1) throughout the transport, and only converted to seconds via a final multiplication by `D/c` — never squared.

## Known limitations (v1)

- Monochromatic source only; the dust column is plane-parallel (density depends only on the line-of-sight coordinate).
- Pure scattering (albedo = 1) — no photoelectric absorption. A `weight` field is carried through the photon state for future extension (absorption weighting, Russian roulette).
- The first-scatter peel-off evaluates the phase function/time-delay at each output angle bin's center rather than integrating across the bin — fine for reasonably fine binning, coarse binning will show it.
- The returned image is in relative units (peeled-off weight per photon, per bin): it has the correct *shape* — radial profile, time evolution, single- vs. multiple-scattering balance — but calibrating it to an absolute physical surface brightness additionally needs the source flux/fluence and a detector-specific geometric factor, left to the caller. `dsh.binning.to_surface_brightness` converts the raw histogram into a properly per-steradian-per-second quantity up to that overall normalization.

## Installation

```bash
git clone https://github.com/AtakanSyo/dsh-ray-tracing.git
cd dsh-ray-tracing
pip install -e ".[dev]"       # CPU JAX + tests + plotting, for development
# or, on a CUDA 12 GPU machine:
pip install -e ".[cuda]"
```

Requires Python ≥ 3.10. See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) if you need a different accelerator/CUDA version than the `cuda` extra assumes.

## Quickstart

```python
import jax
import dsh

profile = dsh.uniform_slab(x_lo=0.30, x_hi=0.35)   # dust between 30-35% of the way to the source
bin_spec = dsh.default_bin_spec()                   # (sky angle, time delay) binning

result = dsh.simulate(
    jax.random.PRNGKey(0),
    n_photons=200_000,
    D_pc=3000,       # source distance
    E_keV=1.0,       # monochromatic photon energy
    tau_sca=0.2,     # total scattering optical depth
    profile=profile,
    bin_spec=bin_spec,
    max_scatter_order=15,
)

print(result.image.shape)              # (n_theta, n_dt)
print(result.fraction_still_active)    # diagnostic: should be ~0 if max_scatter_order is high enough
```

See `examples/quickstart.py` for a runnable version and `examples/plot_halo.py` for a plotting example (radial profile, light curve, and 2D image — needs `pip install -e ".[plot]"`).

Because everything is plain JAX, the whole simulation is `jit`-compilable end to end and differentiable with respect to any continuous input (`D_pc`, `tau_sca`, the profile's parameters, ...) — useful for gradient-based fitting to real halo data.

## Project layout

```
src/dsh/
  constants.py       physical constants & unit conversions
  grains.py           grain size distributions (MRN)
  cross_section.py    Rayleigh-Gans scattering cross section & phase function
  geometry.py          source-dust-observer time delay (exact, numerically stable)
  dust_model.py        line-of-sight dust density profiles
  photon.py             photon packet state (a JAX pytree)
  scatter.py            free-path & scattering-angle sampling primitives
  tracer.py             the Monte Carlo peel-off ray tracer
  binning.py             (angle, time-delay) image histogramming
tests/                 pytest suite (physics sanity checks + numerical regression tests)
examples/               runnable scripts
```

## Testing

```bash
pytest
```

The test suite includes: an independent re-derivation of the exact time-delay formula (checked against this package's own implementation), a cross-check of the spherical Bessel function against `scipy.special.spherical_jn`, normalization checks on the grain-size and phase-function tables, and Monte Carlo statistical checks (e.g. that the single-scattering probability matches the analytic `1 - exp(-τ)` and that scattered flux scales linearly with optical depth in the optically-thin limit).

## References

- Mathis, J. S., Rumpl, W., & Nordsieck, K. H. 1977, *ApJ*, 217, 425 — the MRN grain size distribution.
- Mauche, C. W., & Gorenstein, P. 1986, *ApJ*, 303, 569 — Rayleigh-Gans X-ray dust scattering.
- Smith, R. K., & Dwek, E. 1998, *ApJ*, 503, 831 — the exact RG cross section and single-scattering halo formalism used here.
- Draine, B. T. 2003, *ApJ*, 598, 1017 — dust optical constants and scattering theory.
- Whitney, B. A. 2003, *"Monte Carlo Radiative Transfer"*, in *SINS - Small Ionized and Neutral Structures in the Diffuse Interstellar Medium*, ASP Conf. Ser. 293 — the peel-off/next-event estimator technique.
- Corrales, L. R., et al. — [`eblur/dust`](https://github.com/eblur/dust) / [`eblur/newdust`](https://github.com/eblur/newdust), a semi-analytic single-scattering reference implementation.

## License

MIT — see [LICENSE](LICENSE).
