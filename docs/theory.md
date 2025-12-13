# Theory and Physics Documentation

This document describes the physical models and assumptions used in generating spiral galaxy initial conditions.

## Table of Contents

1. [Overview](#overview)
2. [Dark Matter Halo](#dark-matter-halo)
3. [Stellar Bulge](#stellar-bulge)
4. [Disc Components](#disc-components)
5. [Kinematics](#kinematics)
6. [Spiral Arms](#spiral-arms)
7. [Bar Structure](#bar-structure)
8. [Merger Orbits](#merger-orbits)
9. [Gas Physics](#gas-physics)

---

## Overview

The initial conditions generator creates equilibrium disc galaxies with the following components:

- **Dark matter halo**: NFW profile
- **Stellar bulge**: Hernquist profile
- **Stellar disc**: Exponential profile with optional spiral arms and bar
- **Gas disc**: Exponential profile with optional spiral arms and bar

The kinematics are computed using a "Tier B" approach:
- Rotation curves from combined gravitational potential
- Velocity dispersions from Toomre Q stability criterion
- Asymmetric drift for disc mean velocities
- Jeans equation for spheroidal components (halo, bulge)

---

## Dark Matter Halo

### NFW Profile

The dark matter halo follows a Navarro-Frenk-White (NFW) profile:

```
ρ(r) = ρ₀ / [x(1 + x)²]
```

where `x = r/rₛ` and `rₛ` is the scale radius.

**Parameters:**
- `M₂₀₀`: Virial mass (mass within r₂₀₀, where mean density = 200 × critical density)
- `c₂₀₀`: Concentration parameter = r₂₀₀/rₛ

**Derivation of rₛ and ρ₀:**

The virial radius r₂₀₀ is:
```
r₂₀₀ = (3M₂₀₀ / 4πρ_crit / 200)^(1/3)
```

Scale radius:
```
rₛ = r₂₀₀ / c₂₀₀
```

Characteristic density:
```
δc = (200/3) × c₂₀₀³ / f(c₂₀₀)
f(c) = ln(1 + c) - c/(1 + c)
```

**Enclosed mass:**
```
M(<r) = M₂₀₀ × [ln(1 + x) - x/(1 + x)] / f(c₂₀₀)
```

### Sampling Method

Particles are sampled using inverse CDF method:
1. Generate uniform random u ∈ [0,1]
2. Solve M(<r)/M_total = u for r
3. Sample angles uniformly on sphere

**Truncation**: The halo is truncated at r_max = 10×rₛ to avoid infinite extent.

---

## Stellar Bulge

### Hernquist Profile

The bulge follows a Hernquist (1990) profile:

```
ρ(r) = M_bulge / (2π) × a / [r(r + a)³]
```

where `a` is the scale length.

**Enclosed mass:**
```
M(<r) = M_bulge × r² / (r + a)²
```

**Gravitational potential:**
```
Φ(r) = -G M_bulge / (r + a)
```

### Sampling Method

Inverse CDF for Hernquist profile:
```
r = a × √u / (1 - √u)
```
where u ∈ [0,1] is uniform random.

### Bulge Mass from D/T Ratio

The bulge mass is determined from the disc-to-total ratio:
```
M_bulge = M_star × (1 - D/T)
M_disc_star = M_star × D/T
```

---

## Disc Components

### Exponential Radial Profile

Both stellar and gas discs follow exponential surface density:

```
Σ(R) = Σ₀ exp(-R/R_d)
```

where:
- `Σ₀` = central surface density
- `R_d` = scale length

**Total mass:**
```
M_disc = 2π Σ₀ R_d²
Σ₀ = M_disc / (2π R_d²)
```

### Vertical Structure

The vertical density profile uses a sech² form:

```
ρ(z) = ρ₀ sech²(z / 2z_d) / (4z_d)
```

where `z_d` is the scale height.

**Sampling vertical positions:**

Inverse CDF:
```
z = 2z_d × arctanh(2u - 1)
```

### Radial Sampling

For exponential disc, solve numerically:
```
1 - (1 + R/R_d) exp(-R/R_d) = u
```

---

## Kinematics

### Rotation Curve

The circular velocity is computed from the combined gravitational potential of all components (halo + bulge + stellar disc + gas disc).

For each component, we compute the Miyamoto-Nagai or direct potential, then:

```
v_c²(R) = R × dΦ_total/dR |_{z=0}
```

The derivative is computed numerically using finite differences.

### Toomre Q Stability

For the stellar disc, the radial velocity dispersion is set by the Toomre Q criterion:

```
Q = σ_R × κ / (π G Σ)
```

Solving for σ_R:
```
σ_R = Q × π G Σ / κ
```

where:
- `κ` = epicyclic frequency
- `Σ` = surface density
- Typical values: Q_star ~ 1.5, Q_gas ~ 2.0

**Epicyclic frequency:**

```
κ² = (2v_c/R)² + 2(v_c/R) × dv_c/dR
```

### Velocity Dispersion Components

From epicyclic theory:
```
σ_φ ≈ 0.7 σ_R
σ_z ≈ 0.6 σ_R
```

### Asymmetric Drift

The mean azimuthal velocity of disc stars is reduced relative to circular velocity due to pressure support:

```
v_c² - v̄_φ² = σ_R² [1 - (σ_φ/σ_R)² + R/(2σ_R²) d(σ_R²)/dR]
```

For exponential disc with constant Q:
```
d(σ_R²)/dR ≈ -2σ_R²/R_d
```

### Jeans Equation for Spheroidal Components

For halo and bulge, we use the spherical Jeans equation with isotropic velocities (β = 0):

```
σ_r² = (1/ρ) ∫_r^∞ ρ(r') G M(<r') / r'² dr'
```

**Simplified local approximation:**
```
σ_r² ≈ G M(<r) / (2r)
```

Velocities are sampled from a Gaussian and truncated at the escape velocity.

### Escape Velocity

```
v_esc² = -2 Φ_total(r)
```

where Φ → 0 at infinity.

---

## Spiral Arms

### Logarithmic Spiral Pattern

The spiral arms follow a logarithmic spiral in the disc plane:

```
φ - φ₀ = (1/tan i) ln(R/R₀)
```

where `i` is the pitch angle.

For m arms, the phase is:
```
Ψ = m[φ - (1/tan i) ln(R/R₀)]
```

### Density Perturbation

The spiral arms modulate the disc density:

```
ρ'(R,φ) = ρ₀(R) × [1 + A f(R) cos Ψ]
```

where:
- `A` = arm strength (typical: 0.2-0.4)
- `f(R)` = radial envelope (smooth turn-on/off at inner/outer radii)

**Radial envelope:**
```
f(R) = 0                    for R < R_min
     = (R - R_min)          for R_min < R < R_min + 1 kpc
     = 1                    for R_min + 1 < R < R_max - 2
     = (R_max - R)/2        for R_max - 2 < R < R_max
     = 0                    for R > R_max
```

Typical: R_min = 2 kpc, R_max = 15 kpc

### Streaming Motions

Spiral arms induce non-circular motions aligned with the arm pattern:

```
v_R = v_stream × f(R) sin Ψ × sin i
Δv_φ = -v_stream × f(R) sin Ψ × cos i
```

where:
```
v_stream = f_stream × v_c(R)
```

and `f_stream` is the streaming fraction (typical: 0.05-0.15).

**Physical interpretation**: Gas and stars flow along the spiral arms, with radial and azimuthal components determined by the pitch angle.

---

## Bar Structure

### Elliptical Deformation

The bar creates an elliptical deformation of the disc within a characteristic radius:

```
R_elliptical = √[x² + (y/q)²]
```

where `q < 1` is the axis ratio (typical: 0.2-0.4).

### Density Perturbation

The bar creates an m=2 perturbation:

```
ρ'(R,φ) = ρ₀(R) × [1 + A_bar exp(-(R/R_bar)²) cos(2φ_bar)]
```

where:
- `φ_bar` = azimuthal angle in bar frame
- `R_bar` = bar extent (typical: 2-4 kpc)
- `A_bar` = bar strength (typical: 0.3-0.7)

### Position Perturbation

Particle positions are compressed along the minor axis:

```
y' = y × [1 + A_bar exp(-(R/R_bar)²) (q - 1)]
```

### Streaming Motions

Bar induces streaming along major axis:

```
v_R = v_stream exp(-(R/R_bar)²) sin(2φ_bar)
Δv_φ = v_stream exp(-(R/R_bar)²) cos(2φ_bar)
```

where `v_stream = f_stream × v_c(R)`.

---

## Merger Orbits

### Parabolic Orbit Approximation

Secondary galaxies are placed on parabolic orbits (E = 0) relative to the primary. This approximation is appropriate for first infall.

**Energy conservation:**
```
E = (1/2)v² - GM/r = 0
v² = 2GM/r
```

**Angular momentum:**
```
L = r_peri × v_peri
```

At pericentre:
```
v_peri = √(2GM/r_peri)
L = √(2GM × r_peri)
```

### "Head-onness" Parameter

The pericentre distance `r_peri` controls the orbital angular momentum:
- `r_peri = 0`: Head-on collision (L = 0)
- `r_peri > 0`: Grazing/orbital encounter with L ∝ √r_peri

### Initial Conditions

At initial separation `r_init`:

**Tangential velocity:**
```
v_t = L / r_init = √(2GM × r_peri) / r_init
```

**Radial velocity:**
```
v_r = -√(2GM/r_init - v_t²)
```

(Negative because infalling)

### Disc Orientation

Secondary galaxy discs can be oriented using:
- **Inclination**: Angle from face-on (0° = face-on, 90° = edge-on)
- **Node angle**: Rotation of line of nodes
- **Orbit plane angle**: Tilt of orbital plane from primary disc plane

**Rotation matrices** are applied sequentially to transform from galaxy frame to simulation frame.

### Center-of-Mass Correction

After placing all galaxies, positions and velocities are corrected to the center-of-mass frame:

```
r_COM = Σ(m_i × r_i) / M_total
v_COM = Σ(m_i × v_i) / M_total

r_i' = r_i - r_COM
v_i' = v_i - v_COM
```

---

## Gas Physics

### Temperature and Internal Energy

Gas particles are assigned an initial temperature (default: 10⁴ K, typical of warm ISM).

**Specific internal energy:**
```
u = k_B T / [(γ - 1) μ m_p]
```

where:
- `k_B` = Boltzmann constant
- `γ = 5/3` = adiabatic index
- `μ = 0.6` = mean molecular weight (ionized)
- `m_p` = proton mass

**Velocity dispersion from temperature:**
```
σ = √(k_B T / μ m_p)
```

### Smoothing Lengths

Gas smoothing lengths are estimated assuming uniform distribution:

```
h = (3 N_ngb / 4π n)^(1/3)
```

where:
- `N_ngb` = target neighbor count (default: 58)
- `n = N_gas / V_box` = number density

SWIFT will refine these during the first few timesteps.

---

## Numerical Implementation Details

### Mass Quantization

All particles have identical mass `m_part`. Requested component masses are rounded:

```
N = round(M_requested / m_part)
M_achieved = N × m_part
```

**Mass conservation:**
```
|M_achieved - M_requested| ≤ 0.5 × m_part
```

This ensures exact energy/momentum conservation in N-body code.

### Particle ID Assignment

Particle IDs are assigned sequentially and globally unique:
1. Gas: IDs 1 to N_gas
2. DM: IDs (N_gas + 1) to (N_gas + N_dm)
3. Stars: IDs (N_gas + N_dm + 1) to (N_gas + N_dm + N_stars)

### Box Wrapping

All coordinates are wrapped into the periodic box [0, L_box):

```
x' = mod(x, L_box)
```

Velocities are not wrapped.

### Guardrails Against Transients

Several measures reduce initial transients:

1. **Toomre Q floor**: Q ≥ 1.0 prevents violent disc instabilities
2. **Minimum dispersion**: σ_R ≥ 5 km/s prevents numerical issues
3. **Escape velocity clipping**: Velocities exceeding v_esc are rescaled
4. **Smooth spiral envelope**: Gradual turn-on/off prevents shocks
5. **Gas pressure support**: T = 10⁴ K provides thermal support

---

## Assumptions and Limitations

### Assumptions

1. **Equilibrium**: Galaxies start in (approximate) equilibrium
2. **Axisymmetry**: Base profiles are axisymmetric (before spiral/bar)
3. **Isotropic bulge/halo**: β = 0 velocity anisotropy
4. **Isothermal gas**: Single temperature for all gas
5. **No initial black holes**: BHs seeded by SWIFT during run
6. **Parabolic orbits**: First infall approximation for mergers

### Limitations

1. **No rotation for bulge/halo**: Spheroidal components non-rotating
2. **Simple spiral pattern**: Single pattern speed (not self-consistent)
3. **Static bar**: Bar does not evolve (pattern frozen in ICs)
4. **No thick disc**: Single exponential for stellar disc
5. **Uniform particle mass**: Cannot resolve multiple scales optimally

### Recommended Parameter Ranges

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| M₂₀₀ | 10¹⁰ - 10¹³ M⊙ | Dwarf to massive galaxy |
| c₂₀₀ | 5 - 20 | Lower for massive halos |
| D/T | 0.6 - 0.9 | Disc-to-total ratio |
| R_d | 2 - 5 kpc | Disc scale length |
| z_d | 0.2 - 0.5 kpc | ~0.1 × R_d |
| Q_star | 1.2 - 2.0 | Stability parameter |
| Pitch angle | 10° - 25° | Typical spiral galaxies |
| Arm strength | 0.1 - 0.4 | Moderate to strong |

---

## References

- Navarro, Frenk & White (1997) - NFW halo profile
- Hernquist (1990) - Bulge model
- Binney & Tremaine (2008) - Galactic Dynamics textbook
- Toomre (1964) - Disc stability criterion
- Miyamoto & Nagai (1975) - Disc potential approximation
- Springel et al. (2005) - GADGET SPH formulation
- Schaye et al. (2015) - EAGLE simulations

---

## Implementation Notes

### Code Structure

```
physics/
  profiles.py      - Density profiles (NFW, Hernquist, exponential)
  potentials.py    - Gravitational potentials and rotation curves
  kinematics.py    - Velocity dispersions and asymmetric drift
  perturbations.py - Spiral arms and bar perturbations
  sampling.py      - Particle position/velocity sampling
  orbits.py        - Merger orbit calculations

io/
  swift_writer.py  - HDF5 IC file output
  yaml_writer.py   - SWIFT parameter file generation
```

### Unit System

**Internal units (matching SWIFT galaxy simulations):**
- Length: kpc
- Mass: M⊙
- Velocity: km/s
- Time: (derived) ~978 Myr
- G = 4.302 × 10⁻⁶ kpc (km/s)² / M⊙

**Conversions to CGS:**
- 1 kpc = 3.086 × 10²¹ cm
- 1 M⊙ = 1.989 × 10³³ g
- 1 km/s = 10⁵ cm/s
