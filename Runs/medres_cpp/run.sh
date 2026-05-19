#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/.." && pwd)"
SWIFT_BIN="${SWIFT_BIN:-/Users/willroper/Research/SWIFT/swiftsim/swift}"
MAKE_MOVIE="${MAKE_MOVIE:-0}"
EAGLE_FLAGS="--hydro --self-gravity --stars --cooling --star-formation --feedback"

cd "$REPO_DIR"

python -m swift_spiral_ics.cli.generate \
  --out-ics Runs/medres_cpp/medres_merger.hdf5 \
  --out-params Runs/medres_cpp/medres_merger.yml \
  --run-name medres_merger \
  --snapshot-basename Runs/medres_cpp/snapshot \
  --n-galaxies 2 \
  --secondary-mass-ratio 0.8 \
  --separation-kpc 200 \
  --impact-kpc 20 \
  --relative-velocity-kms 50 \
  --galaxy1-inclination-deg 0 \
  --galaxy2-inclination-deg 0 \
  --box-kpc 1200 \
  --n-halo 5000 \
  --n-bulge 1000 \
  --n-star 5000 \
  --n-gas 5000 \
  --nR-grid 96 \
  --nz-grid 96 \
  --eps-grid 0.8 \
  --dt 0.001 \
  --dt-min-gyr 1e-6 \
  --time-end-gyr 0.02 \
  --snapshot-dt-myr 10 \
  --arm-strength 0.15 \
  --arm-stream-frac 0.02 \
  --Q-star 2.0 \
  --Q-gas 1.5 \
  --bulge-rmax-scale 50

"$SWIFT_BIN" $EAGLE_FLAGS --threads=4 Runs/medres_cpp/medres_merger.yml

if [[ "$MAKE_MOVIE" == "1" ]]; then
  python create_movie.py Runs/medres_cpp \
    --width-kpc 320 \
    --npix 420 \
    --fps 12 \
    --dpi 150 \
    --bitrate 2400 \
    --gas-vmin 5e4 \
    --gas-vmax 2e8 \
    --stars-vmin 5e4 \
    --stars-vmax 2e9 \
    --title-prefix "Medium two-galaxy test"
fi
