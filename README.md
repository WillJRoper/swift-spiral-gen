# SWIFT Spiral ICs

Generate SWIFT initial conditions for spiral disc galaxies with configurable spiral arms, bars, and merger configurations.

## Features

- **Realistic galaxy models**: NFW halos, Hernquist bulges, exponential discs
- **Spiral structure**: Configurable number of arms, pitch angle, and streaming motions
- **Bar support**: Optional bar component with configurable parameters
- **Merger configurations**: Simple "head-onness" parameter for galaxy encounters
- **Tier B kinematics**: Jeans equation + asymmetric drift + Q-based dispersions
- **SWIFT-compliant output**: Proper HDF5 format with all required fields
- **Diagnostic tools**: Visualization script for density and kinematics
- **Movie maker**: Create movies from SWIFT snapshot sequences

## Installation

### From source

```bash
git clone <repository-url>
cd swift-spiral-ics
pip install -e .
```

### Dependencies

- Python >= 3.9
- numpy, scipy, h5py, pyyaml, tqdm
- matplotlib, imageio (for visualization and movies)

## Quick Start

### Generate The MW-Andromeda Example

```bash
swift-spiral-ics examples/mw_m31_merger.yml
```

This writes `mw_m31_merger.hdf5` and `mw_m31_merger.yml` using a low-resolution parabolic encounter with baryonic particle masses of `1e6 Msun` and coarser `1e7 Msun` dark matter particles.

### Visualize initial conditions

```bash
swift-spiral-ics-viz galaxy_ic.hdf5 --out-pdf diagnostics.pdf
```

### Create a movie from snapshots

```bash
swift-spiral-movie "snapshot_*.hdf5" --out-movie evolution.mp4 --fps 15
```

## Generator Interface

The main generator is `swift-spiral-ics CONFIG.yml`.

All model choices live in a YAML file so each setup is reproducible and easy to re-run. The first complete example is `examples/mw_m31_merger.yml`.

### Required Inputs

For a useful run you should normally set `output`, `simulation`, `particle_masses`, and `galaxies`. For multi-galaxy manual placement, set `galaxies[].placement.position_kpc`; for automatic encounters, set `orbit.type: parabolic`.

Everything else has defaults.

### Mass Model

- `galaxies[].masses.dm_msun`: dark matter halo mass for each galaxy
- `galaxies[].masses.stars_msun`: total stellar mass for each galaxy, including both bulge and stellar disc
- `galaxies[].masses.gas_msun`: total gas disc mass for each galaxy
- `galaxies[].masses.bulge_fraction`: stellar bulge fraction defined as `B / (D + B)`

The stellar mass split is:

- bulge mass = `star_mass * bulge_fraction`
- stellar disc mass = `star_mass * (1 - bulge_fraction)`

### Particle Masses

These are global per component, not per galaxy:

- `particle_masses.dm_msun`: dark matter particle mass
- `particle_masses.stars_msun`: stellar particle mass
- `particle_masses.gas_msun`: gas particle mass

Particle counts are derived internally by rounding component mass divided by component particle mass.

Bulge particles use `--star-part-mass-msun` because the bulge is stellar.

### Halo Parameters

- `galaxies[].halo.c200`: NFW concentration for each galaxy

Dark matter is not disk-based. It is always treated as an NFW halo.

### Stellar Structure

- `galaxies[].stellar_disk.scale_length_kpc`: stellar disk scale length
- `galaxies[].stellar_disk.scale_height_kpc`: stellar disk scale height
- `galaxies[].stellar_disk.Q`: stellar Toomre `Q`

### Gas Structure

- `galaxies[].gas_disk.scale_length_kpc`: gas disk scale length
- `galaxies[].gas_disk.scale_height_kpc`: gas disk scale height
- `galaxies[].gas_disk.Q`: gas Toomre `Q`

### Bulge Structure

- `galaxies[].bulge.a_kpc`: Hernquist bulge scale radius
- `galaxies[].bulge.rmax_scale`: bulge truncation radius in units of `a`

### Central Black Holes

- `galaxies[].black_hole.mass_msun`: optional central black-hole mass written as a `PartType5` particle

If omitted or set to `0`, no explicit black hole is written for that galaxy. When present, the black hole is placed at the galaxy centre and receives the same orbital bulk velocity as the galaxy.

### Circumgalactic Medium

- `galaxies[].cgm.enabled`: add an approximate spherical hot CGM component
- `galaxies[].cgm.mass_msun`: total CGM gas mass
- `galaxies[].cgm.r_min_kpc`: inner sampling radius
- `galaxies[].cgm.r_max_kpc`: outer sampling radius
- `galaxies[].cgm.core_radius_kpc`: beta-profile core radius
- `galaxies[].cgm.beta`: beta-profile slope parameter
- `galaxies[].cgm.temperature_K`: gas temperature used for internal energy

The CGM sampler uses a spherical beta-profile-like density law and assigns the galaxy bulk velocity. It is an approximate hot halo, not a hydrostatic equilibrium solution.

### Multi-Galaxy Placement

Positions are literal box coordinates in kpc and must lie between `0` and `simulation.box_kpc`:

- `galaxies[].placement.position_kpc`

Bulk velocities are given in km/s:

- `galaxies[].placement.velocity_kms`

Disk orientations are given by:

- `galaxies[].placement.inclination_deg`
- `galaxies[].placement.node_angle_deg`

`position_kpc` and `velocity_kms` are three-element vectors. If no velocity is provided, the galaxy receives zero bulk velocity.

Alternatively, for two-galaxy encounters, the generator can compute centre-of-mass positions and velocities from an orbit block.

For observed-like MW-M31 analogues, use:

- `orbit.type: relative_velocity`
- `orbit.separation_kpc`: initial galaxy-centre separation
- `orbit.radial_velocity_kms`: relative radial velocity; negative values approach
- `orbit.tangential_velocity_kms`: relative tangential velocity
- `orbit.plane_angle_deg`: optional orbit-plane rotation around the y-axis

For idealized controlled encounters, use:

- `orbit.type: parabolic`
- `orbit.r_init_kpc`: initial galaxy-centre separation
- `orbit.r_peri_kpc`: target parabolic pericentre distance
- `orbit.plane_angle_deg`: optional orbit-plane rotation around the y-axis

When using either orbit mode, do not also provide manual positions or velocities; those COM positions and velocities are computed from the galaxy masses.

### Spiral Structure

- `galaxies[].spiral.n_arms`: number of spiral arms
- `galaxies[].spiral.pitch_deg`: spiral pitch angle
- `galaxies[].spiral.strength`: spiral perturbation strength
- `galaxies[].spiral.stream_frac`: streaming fraction applied in the spiral perturbation

### Bar Structure

- `galaxies[].bar.enabled`: turn on the bar model
- `galaxies[].bar.strength`: bar strength
- `galaxies[].bar.radius_kpc`: bar radius
- `galaxies[].bar.q`: bar flattening
- `galaxies[].bar.angle_deg`: bar angle in degrees

### Time Integration And Outputs

- `simulation.max_timestep_gyr`: maximum SWIFT timestep written to the YAML
- `simulation.dt_min_gyr`: minimum physical timestep written to the YAML
- `simulation.time_end_gyr`: total simulation duration
- `simulation.snapshot_dt_myr`: snapshot cadence

### Grid Solver Controls

- `grid.nR`: radial grid resolution for the C++ solver
- `grid.nz`: vertical grid resolution for the C++ solver
- `grid.eps_kpc`: solver softening length in kpc
- `grid.h_max_cell_fraction`: set `h_max` as a fraction of the SWIFT top-level cell width
- `grid.max_top_level_cells`: set `Scheduler.max_top_level_cells` in the generated SWIFT YAML
- `grid.scheduler_tasks_per_cell`: set `Scheduler.tasks_per_cell` in the generated SWIFT YAML

### Feedback And Runtime Parameters

- `simulation.feedback_scale`: relative scaling of the EAGLE SNII feedback energy fractions

Examples:

- `1.0`: default feedback
- `0.25`: one quarter of the default SNII energy fraction
- `2.0`: double the default SNII energy fraction

### Background Medium

- `background.gas_density_msun_kpc3`: uniform background gas density
- `background.dm_density_msun_kpc3`: uniform background dark matter density
- `background.grid_kpc`: background particle spacing
- `background.radius_kpc`: optional spherical cutoff radius for background particles around the central galaxy

Background behavior:

- `background.grid_kpc: 0`: random uniform background
- `background.grid_kpc > 0`: regular grid background with jitter
- `background.radius_kpc: R`: limit either background mode to a sphere of radius `R` rather than the full box

### Output And Metadata

- `output.ics`: output IC HDF5 path
- `output.params`: output SWIFT YAML path
- `output.run_name`: run name written into the YAML
- `output.snapshot_basename`: snapshot basename written into the YAML
- `output.param_template`: parameter template name
- `simulation.seed`: random seed
- `simulation.box_kpc`: simulation box size

### Visualization (`swift-spiral-ics-viz`)

```bash
swift-spiral-ics-viz IC_FILE [--out-pdf PDF] [--out-dir DIR]
```

Creates:
- Surface density projections (DM, gas, stars)
- Rotation curves
- Velocity dispersion profiles
- 2D velocity field maps

### Movie Maker (`swift-spiral-movie`)

```bash
swift-spiral-movie PATTERN [--out-movie MP4] [--fps FPS] [--show-vel]
```

Options:
- `--fps INT`: Frames per second (default: 10)
- `--bins INT`: Density projection resolution (default: 512)
- `--show-vel`: Overlay velocity vectors

## Physics Implementation

See [docs/theory.md](docs/theory.md) for detailed documentation of the physics models, including:

- NFW halo profiles
- Hernquist bulge model
- Exponential disc structure
- Toomre Q-based stability
- Asymmetric drift
- Spiral arm density and streaming perturbations
- Bar dynamics
- Parabolic merger orbits

## Output Files

### IC HDF5 Format

The generated HDF5 files comply with SWIFT initial conditions format:

- **Header**: BoxSize, NumPart_Total, Flag_Entropy_ICs, etc.
- **Units**: Unit conversion factors
- **PartType0**: Gas particles (Coordinates, Velocities, Masses, ParticleIDs, InternalEnergy, SmoothingLength)
- **PartType1**: Dark matter particles
- **PartType4**: Star particles

### YAML Parameter File

Includes complete SWIFT configuration with:
- Gravity (FMM)
- SPH hydrodynamics
- EAGLE chemistry and cooling
- Star formation (pressure law)
- Stellar feedback
- SPINJETAGN black hole seeding and feedback

## Running SWIFT

After generating ICs:

```bash
swift --hydro --self-gravity --stars --feedback --threads=16 mw_m31_merger.yml
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Tests cover:
- Physics module correctness
- HDF5 file structure compliance
- Mass conservation
- Particle ID uniqueness
- Integration tests

## Examples

Reusable generator configs live in `examples/`:

- `examples/mw_m31_merger.yml`: low-resolution Milky Way-Andromeda-like parabolic merger

## License

MIT License

## Citation

If you use this code in a publication, please cite:

```
[Citation information to be added]
```

## Contributing

Contributions welcome! Please open an issue or pull request.

## Support

For questions or issues, please open a GitHub issue.
