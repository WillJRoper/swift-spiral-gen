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
  --xs -100 100 \
  --ys 0 0 \
  --zs 0 0 \
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

## Command-Line Options

### Main Generator (`swift-spiral-ics`)

**Global parameters:**
- `--out-ics PATH`: Output IC HDF5 file
- `--out-params PATH`: Output YAML parameter file
- `--box-kpc FLOAT`: Simulation box size (kpc)
- `--m-part-msun FLOAT`: Particle mass (Msun)
- `--n-galaxies INT`: Number of galaxies
- `--seed INT`: Random seed for reproducibility

**Per-galaxy parameters** (provide N values for N galaxies):
- `--m200-msun`: Halo M200 mass (Msun)
- `--c200`: Halo concentration
- `--m-star-msun`: Total stellar mass (Msun)
- `--m-gas-msun`: Total gas mass (Msun)
- `--max-timestep-gyr`: Maximum simulation time-step (Gyr)
- `--stellar-disk-scale-length-kpc`: Stellar disk scale length (kpc)
- `--stellar-disk-scale-height-kpc`: Stellar disk scale height (kpc)
- `--gas-disk-scale-length-kpc`: Gas disk scale length (kpc)
- `--gas-disk-scale-height-kpc`: Gas disk scale height (kpc)
- `--bulge-a-kpc`: Bulge Hernquist scale length (kpc)

**Spiral and bar parameters:**
- `--n-arms`: Number of spiral arms
- `--pitch-deg`: Spiral pitch angle (degrees)
- `--arm-strength`: Spiral arm density perturbation amplitude
- `--arm-stream-frac`: Spiral streaming velocity (fraction of v_c)
- `--bar`: Bar enabled (0 or 1)
- `--bar-r-kpc`: Bar extent (kpc)
- `--bar-q`: Bar axis ratio
- `--bar-stream-frac`: Bar streaming velocity fraction

**Merger parameters** (for secondary galaxies):
- `--r-init-kpc`: Initial separation (kpc)
- `--r-peri-kpc`: Pericentre distance - the "head-onness" knob (kpc)
- `--orbit-plane-deg`: Orbit plane inclination (degrees)
- `--inclination-deg`: Secondary disc inclination (degrees)
- `--node-deg`: Ascending node angle (degrees)

**Stability parameters:**
- `--q-star FLOAT`: Toomre Q for stellar disc (default: 1.5)
- `--q-gas FLOAT`: Toomre Q for gas disc (default: 2.0)

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

### Milky Way-like galaxy

```bash
swift-spiral-ics \
  --out-ics mw_ic.hdf5 --out-params mw_params.yml \
  --box-kpc 500 --m-part-msun 1e6 --n-galaxies 1 \
  --m200-msun 1.5e12 --c200 12 \
  --m-star-msun 6e10 --m-gas-msun 1e10 \
  --max-timestep-gyr 0.75 --stellar-disk-scale-length-kpc 3.0 --stellar-disk-scale-height-kpc 0.3 \
  --gas-disk-scale-length-kpc 7.0 --gas-disk-scale-height-kpc 0.15 --bulge-a-kpc 0.7 \
  --n-arms 2 --pitch-deg 12 --arm-strength 0.3
```

### Major merger (1:3 mass ratio)

```bash
swift-spiral-ics \
  --out-ics major_merger_ic.hdf5 --out-params major_merger_params.yml \
  --box-kpc 1000 --m-part-msun 1e6 --n-galaxies 2 \
  --m200-msun 1.2e12 4e11 \
  --c200 10 10 \
  --m-star-msun 5e10 1.5e10 \
  --m-gas-msun 1e10 3e9 \
  --max-timestep-gyr 0.8 \
  --stellar-disk-scale-length-kpc 3.5 2.0 --stellar-disk-scale-height-kpc 0.35 0.2 \
  --gas-disk-scale-length-kpc 7.0 4.0 --gas-disk-scale-height-kpc 0.1 0.1 \
  --bulge-a-kpc 0.8 0.5 \
  --r-init-kpc 300 --r-peri-kpc 10 \
  --inclination-deg 60 --node-deg 90
```

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
