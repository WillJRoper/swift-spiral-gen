#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/.." && pwd)"
SWIFT_BIN="${SWIFT_BIN:-/Users/willroper/Research/SWIFT/swiftsim/swift}"
EAGLE_FLAGS="--hydro --self-gravity --stars --cooling --star-formation --feedback"

cd "$REPO_DIR"

python -m swift_spiral_ics.cli.generate \
  --out-ics Runs/medres_movie/medres_movie.hdf5 \
  --out-params Runs/medres_movie/medres_movie.yml \
  --run-name medres_movie \
  --snapshot-basename Runs/medres_movie/snapshot \
  --n-galaxies 2 \
  --xs -100 100 \
  --ys -10 10 \
  --zs 0 0 \
  --vxs 25 -25 \
  --vys 0 0 \
  --vzs 0 0 \
  --inclination-deg 0 0 \
  --dm-mass-msun 1e12 8e11 \
  --dm-part-mass-msun 2e8 \
  --star-mass-msun 6e10 4.8e10 \
  --bulge-fraction 0.1666666667 0.1666666667 \
  --star-part-mass-msun 1e7 \
  --gas-mass-msun 1e10 8e9 \
  --gas-part-mass-msun 2e6 \
  --bulge-a-kpc 0.8 0.7155417528 \
  --stellar-disk-scale-length-kpc 3.5 3.1304951685 \
  --stellar-disk-scale-height-kpc 0.35 0.3130495168 \
  --gas-disk-scale-length-kpc 7.0 6.2609903370 \
  --gas-disk-scale-height-kpc 0.1 0.0894427191 \
  --box-kpc 1200 \
  --nR-grid 96 \
  --nz-grid 96 \
  --eps-grid 0.8 \
  --max-timestep-gyr 0.0001 \
  --dt-min-gyr 1e-6 \
  --time-end-gyr 0.02 \
  --snapshot-dt-myr 0.2 \
  --arm-strength 0.15 \
  --arm-stream-frac 0.02 \
  --Q-star 2.0 \
  --Q-gas 1.5 \
  --bulge-rmax-scale 50

"$SWIFT_BIN" $EAGLE_FLAGS --threads=4 Runs/medres_movie/medres_movie.yml

python create_movie.py Runs/medres_movie \
  --width-kpc 320 \
  --npix 420 \
  --fps 12 \
  --dpi 150 \
  --bitrate 2400 \
  --gas-vmin 1e5 \
  --gas-vmax 3e8 \
  --stars-vmin 1e5 \
  --stars-vmax 3e9 \
  --title-prefix "Medium two-galaxy test"
