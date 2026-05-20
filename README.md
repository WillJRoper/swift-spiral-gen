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

### Generate a single spiral galaxy

```bash
swift-spiral-ics \
  --out-ics galaxy_ic.hdf5 \
  --out-params galaxy_params.yml \
  --box-kpc 500 \
  --n-galaxies 1 \
  --dm-mass-msun 1e12 \
  --dm-part-mass-msun 1e7 \
  --star-mass-msun 6e10 \
  --bulge-fraction 0.1666666667 \
  --star-part-mass-msun 1.25e7 \
  --gas-mass-msun 1e10 \
  --gas-part-mass-msun 1e7 \
  --c200 10 \
  --max-timestep-gyr 0.8 \
  --stellar-disk-scale-length-kpc 3.5 \
  --stellar-disk-scale-height-kpc 0.35 \
  --gas-disk-scale-length-kpc 7.0 \
  --gas-disk-scale-height-kpc 0.1 \
  --bulge-a-kpc 0.8 \
  --n-arms 2 \
  --pitch-deg 15 \
  --arm-strength 0.25 \
  --arm-stream-frac 0.1
```

### Generate a galaxy merger

```bash
swift-spiral-ics \
  --out-ics merger_ic.hdf5 \
  --out-params merger_params.yml \
  --box-kpc 1000 \
  --n-galaxies 2 \
  --dm-mass-msun 1e12 4e11 \
  --dm-part-mass-msun 1e7 \
  --star-mass-msun 6e10 2e10 \
  --bulge-fraction 0.1666666667 0.25 \
  --star-part-mass-msun 1.25e7 \
  --gas-mass-msun 1e10 3e9 \
  --gas-part-mass-msun 1e7 \
  --c200 10 8 \
  --max-timestep-gyr 0.8 \
  --stellar-disk-scale-length-kpc 3.5 2.5 \
  --stellar-disk-scale-height-kpc 0.35 0.25 \
  --gas-disk-scale-length-kpc 7.0 5.0 \
  --gas-disk-scale-height-kpc 0.1 0.1 \
  --bulge-a-kpc 0.8 0.6 \
  --xs 400 600 \
  --ys 500 500 \
  --zs 500 500 \
  --vxs 50 -50 \
  --vys 0 0 \
  --vzs 0 0 \
  --inclination-deg 0 30
```

### Visualize initial conditions

```bash
swift-spiral-ics-viz galaxy_ic.hdf5 --out-pdf diagnostics.pdf
```

### Create a movie from snapshots

```bash
swift-spiral-movie "snapshot_*.hdf5" --out-movie evolution.mp4 --fps 15
```

## Generator Interface

The main generator is `swift-spiral-ics`.

The current interface is based on three ideas:

- masses are specified by component
- particle masses are specified by component
- positions, velocities, and many structural parameters can be specified per galaxy

For any per-galaxy argument, provide either:

- one value: reuse it for every galaxy
- `N` values: one value for each of `--n-galaxies N`

### Required Inputs

For a useful run you should normally set:

- `--n-galaxies`
- `--dm-mass-msun`
- `--dm-part-mass-msun`
- `--star-mass-msun`
- `--star-part-mass-msun`
- `--gas-mass-msun`
- `--gas-part-mass-msun`

For `--n-galaxies > 1`, you should also set:

- `--xs`
- `--ys`
- `--zs`

If you do not provide positions:

- one galaxy is placed at the box centre with zero bulk velocity
- multiple galaxies raise an error

Everything else has defaults.

### Mass Model

- `--dm-mass-msun`: dark matter halo mass for each galaxy
- `--star-mass-msun`: total stellar mass for each galaxy, including both bulge and stellar disc
- `--gas-mass-msun`: total gas disc mass for each galaxy
- `--bulge-fraction`: stellar bulge fraction defined as `B / (D + B)`

The stellar mass split is:

- bulge mass = `star_mass * bulge_fraction`
- stellar disc mass = `star_mass * (1 - bulge_fraction)`

### Particle Masses

These are global per component, not per galaxy:

- `--dm-part-mass-msun`: dark matter particle mass
- `--star-part-mass-msun`: stellar particle mass
- `--gas-part-mass-msun`: gas particle mass

Particle counts are derived internally by rounding component mass divided by component particle mass.

Bulge particles use `--star-part-mass-msun` because the bulge is stellar.

### Halo Parameters

- `--c200`: NFW concentration for each galaxy

Dark matter is not disk-based. It is always treated as an NFW halo.

### Stellar Structure

- `--stellar-disk-scale-length-kpc`: stellar disk scale length
- `--stellar-disk-scale-height-kpc`: stellar disk scale height
- `--Q-star`: stellar Toomre `Q`

### Gas Structure

- `--gas-disk-scale-length-kpc`: gas disk scale length
- `--gas-disk-scale-height-kpc`: gas disk scale height
- `--Q-gas`: gas Toomre `Q`

### Bulge Structure

- `--bulge-a-kpc`: Hernquist bulge scale radius
- `--bulge-rmax-scale`: bulge truncation radius in units of `a`

### Multi-Galaxy Placement

Positions are literal box coordinates in kpc and must lie between `0` and `--box-kpc`:

- `--xs`
- `--ys`
- `--zs`

Bulk velocities are given in km/s:

- `--vxs`
- `--vys`
- `--vzs`

Disk orientations are given by:

- `--inclination-deg`

If an axis is omitted while positions are otherwise provided, that axis defaults to the box centre for every galaxy.

### Spiral Structure

- `--n-arms`: number of spiral arms
- `--pitch-deg`: spiral pitch angle
- `--arm-strength`: spiral perturbation strength
- `--arm-stream-frac`: streaming fraction applied in the spiral perturbation

### Bar Structure

- `--bar-enabled`: turn on the bar model
- `--bar-strength`: bar strength
- `--bar-radius`: bar radius
- `--bar-q`: bar flattening
- `--bar-angle`: bar angle in degrees

### Time Integration And Outputs

- `--max-timestep-gyr`: maximum SWIFT timestep written to the YAML
- `--dt-min-gyr`: minimum SWIFT timestep written to the YAML
- `--time-end-gyr`: total simulation duration
- `--snapshot-dt-myr`: snapshot cadence

### Grid Solver Controls

- `--nR-grid`: radial grid resolution for the C++ solver
- `--nz-grid`: vertical grid resolution for the C++ solver
- `--eps-grid`: solver softening length in kpc
- `--h-max-cell-fraction`: set `h_max` as a fraction of the SWIFT top-level cell width
- `--max-top-level-cells`: set `Scheduler.max_top_level_cells` in the generated SWIFT YAML
- `--scheduler-tasks-per-cell`: set `Scheduler.tasks_per_cell` in the generated SWIFT YAML

### Feedback And Runtime Parameters

- `--feedback-scale`: relative scaling of the EAGLE SNII feedback energy fractions

Examples:

- `1.0`: default feedback
- `0.25`: one quarter of the default SNII energy fraction
- `2.0`: double the default SNII energy fraction

### Background Medium

- `--bg-gas-density-msun-kpc3`: uniform background gas density
- `--bg-dm-density-msun-kpc3`: uniform background dark matter density
- `--bg-grid-kpc`: background particle spacing
- `--bg-radius-kpc`: optional spherical cutoff radius for background particles around the central galaxy

Background behavior:

- `--bg-grid-kpc 0`: random uniform background
- `--bg-grid-kpc > 0`: regular grid background with jitter
- `--bg-radius-kpc R`: limit either background mode to a sphere of radius `R` rather than the full box

### Output And Metadata

- `--out-ics`: output IC HDF5 path
- `--out-params`: output SWIFT YAML path
- `--run-name`: run name written into the YAML
- `--snapshot-basename`: snapshot basename written into the YAML
- `--param-template`: parameter template name
- `--seed`: random seed
- `--box-kpc`: simulation box size

## Argument Summary

### Per-Galaxy Arguments

- `--dm-mass-msun`
- `--star-mass-msun`
- `--gas-mass-msun`
- `--bulge-fraction`
- `--xs`, `--ys`, `--zs`
- `--vxs`, `--vys`, `--vzs`
- `--inclination-deg`
- `--c200`
- `--bulge-a-kpc`
- `--bulge-rmax-scale`
- `--stellar-disk-scale-length-kpc`
- `--stellar-disk-scale-height-kpc`
- `--gas-disk-scale-length-kpc`
- `--gas-disk-scale-height-kpc`
- `--Q-star`
- `--Q-gas`
- `--n-arms`
- `--pitch-deg`
- `--arm-strength`
- `--arm-stream-frac`
- `--bar-strength`
- `--bar-radius`
- `--bar-q`
- `--bar-angle`

### Global Scalar Arguments

- `--dm-part-mass-msun`
- `--star-part-mass-msun`
- `--gas-part-mass-msun`
- `--box-kpc`
- `--max-timestep-gyr`
- `--dt-min-gyr`
- `--time-end-gyr`
- `--snapshot-dt-myr`
- `--feedback-scale`
- `--nR-grid`
- `--nz-grid`
- `--eps-grid`
- `--bg-gas-density-msun-kpc3`
- `--bg-dm-density-msun-kpc3`
- `--bg-grid-kpc`
- `--seed`
- `--n-galaxies`
- `--bar-enabled`
- `--out-ics`
- `--out-params`
- `--run-name`
- `--snapshot-basename`
- `--param-template`

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
swift --hydro --self-gravity --stars --feedback --threads=16 galaxy_params.yml
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

See the runnable example scripts in `Runs/` for the current recommended usage patterns, including:

- `Runs/cheap_cpp/run.sh`
- `Runs/medres_cpp/run.sh`
- `Runs/medres_movie/run.sh`
- `Runs/medres_500myr/run.sh`
- `Runs/cheap_1gyr_relax_merge/run.sh`
- `Runs/hires_1gyr_relax_merge/run.sh`
- `Runs/cosma_run/run.sh`
- `Runs/test_run/run.sh`

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
