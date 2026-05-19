#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/.." && pwd)"
SWIFT_BIN="${SWIFT_BIN:-/Users/willroper/Research/SWIFT/swiftsim/swift}"
EAGLE_FLAGS="--hydro --self-gravity --stars --cooling --star-formation --feedback"
FEEDBACK_SCALE=0.25

cd "$REPO_DIR"

python -m swift_spiral_ics.cli.generate \
  --out-ics Runs/cosma_run/cosma_run.hdf5 \
  --out-params Runs/cosma_run/cosma_run.yml \
  --run-name cosma_run \
  --snapshot-basename Runs/cosma_run/snapshot \
  --n-galaxies 3 \
  --xs 0 180 -90 \
  --ys 0 60 -30 \
  --zs 0 40 -20 \
  --vxs 30 -70 50 \
  --vys -18.75 -25 65 \
  --vzs -10 -10 30 \
  --inclination-deg 0 55 110 \
  --dm-mass-msun 7.2e11 6.3e11 4.5e11 \
  --gas-mass-msun 3.5e10 3.2e10 2.4e10 \
  --star-mass-msun 2.5e10 1.4e10 6.0e9 \
  --bulge-fraction 0.18 0.12 0.06 \
  --dm-part-mass-msun 7e5 \
  --gas-part-mass-msun 1e5 \
  --star-part-mass-msun 1e5 \
  --bulge-a-kpc 0.8 0.6 0.6 \
  --stellar-disk-scale-length-kpc 3.5 3.2744877137 2.7669929526 \
  --stellar-disk-scale-height-kpc 0.35 0.3274487714 0.2766992953 \
  --gas-disk-scale-length-kpc 7.0 6.5489754274 5.5339859053 \
  --gas-disk-scale-height-kpc 0.1 0.0935567910 0.0790569415 \
  --box-kpc 1600 \
  --nR-grid 160 \
  --nz-grid 160 \
  --eps-grid 0.8 \
  --bg-gas-density-msun-kpc3 100 \
  --bg-grid-kpc 0 \
  --max-timestep-gyr 0.0005 \
  --dt-min-gyr 1e-6 \
  --time-end-gyr 1.0 \
  --snapshot-dt-myr 5.0 \
  --feedback-scale "$FEEDBACK_SCALE" \
  --arm-strength 0.15 \
  --arm-stream-frac 0.02 \
  --Q-star 2.0 \
  --Q-gas 1.5 \
  --bulge-rmax-scale 50

"$SWIFT_BIN" $EAGLE_FLAGS --threads=8 Runs/cosma_run/cosma_run.yml

python create_movie.py Runs/cosma_run \
  --width-kpc 620 \
  --npix 520 \
  --fps 12 \
  --dpi 160 \
  --bitrate 3200 \
  --gas-vmin 5e4 \
  --gas-vmax 2e8 \
  --stars-vmin 5e4 \
  --stars-vmax 2e9 \
  --title-prefix "500k-particle reduced-feedback three-galaxy merger"
