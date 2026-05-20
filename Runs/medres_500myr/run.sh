#!/usr/bin/env bash
set -euo pipefail

SWIFT_BIN="${SWIFT_BIN:-/Users/willroper/Research/SWIFT/swiftsim/swift}"
EAGLE_FLAGS="--hydro --self-gravity --stars --cooling --star-formation --feedback"

swift-spiral-ics \
  --out-ics medres_500myr.hdf5 \
  --out-params medres_500myr.yml \
  --run-name medres_500myr \
  --snapshot-basename snapshot \
  --n-galaxies 2 \
  --xs 675 925 \
  --ys 782.5 817.5 \
  --zs 800 800 \
  --vxs 40 -40 \
  --vys 0 0 \
  --vzs 0 0 \
  --inclination-deg 0 35 \
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
  --box-kpc 1600 \
  --nR-grid 96 \
  --nz-grid 96 \
  --eps-grid 0.8 \
  --h-max-cell-fraction 0.5 \
  --max-timestep-gyr 0.0005 \
  --dt-min-gyr 1e-6 \
  --time-end-gyr 0.5 \
  --snapshot-dt-myr 2.5 \
  --arm-strength 0.15 \
  --arm-stream-frac 0.02 \
  --Q-star 2.0 \
  --Q-gas 1.5 \
  --bulge-rmax-scale 50

"$SWIFT_BIN" $EAGLE_FLAGS --threads=4 medres_500myr.yml

python ../../create_movie.py . \
  --width-kpc 620 \
  --npix 460 \
  --fps 12 \
  --dpi 150 \
  --bitrate 2800 \
  --gas-vmin 5e4 \
  --gas-vmax 2e8 \
  --stars-vmin 5e4 \
  --stars-vmax 2e9 \
  --title-prefix "500 Myr local-frame merger test"
