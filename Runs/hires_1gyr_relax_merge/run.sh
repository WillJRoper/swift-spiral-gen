#!/usr/bin/env bash
set -euo pipefail

SWIFT_BIN="${SWIFT_BIN:-/Users/willroper/Research/SWIFT/swiftsim/swift}"
EAGLE_FLAGS="--hydro --self-gravity --stars --cooling --star-formation --feedback"

swift-spiral-ics \
  --out-ics hires_1gyr_relax_merge.hdf5 \
  --out-params hires_1gyr_relax_merge.yml \
  --run-name hires_1gyr_relax_merge \
  --snapshot-basename snapshot \
  --n-galaxies 2 \
  --xs 600 1000 \
  --ys 791 809 \
  --zs 800 800 \
  --vxs 75 -75 \
  --vys 0 0 \
  --vzs 0 0 \
  --inclination-deg 0 25 \
  --dm-mass-msun 1e12 8e11 \
  --dm-part-mass-msun 6.6666666667e7 \
  --star-mass-msun 6e10 4.8e10 \
  --bulge-fraction 0.1666666667 0.1666666667 \
  --star-part-mass-msun 3.3333333333e6 \
  --gas-mass-msun 1e10 8e9 \
  --gas-part-mass-msun 3.3333333333e5 \
  --bulge-a-kpc 0.8 0.7155417528 \
  --stellar-disk-scale-length-kpc 3.5 3.1304951685 \
  --stellar-disk-scale-height-kpc 0.35 0.3130495168 \
  --gas-disk-scale-length-kpc 7.0 6.2609903370 \
  --gas-disk-scale-height-kpc 0.1 0.0894427191 \
  --box-kpc 1600 \
  --nR-grid 128 \
  --nz-grid 128 \
  --eps-grid 0.8 \
  --h-max-cell-fraction 0.5 \
  --bg-gas-density-msun-kpc3 10 \
  --bg-grid-kpc 0 \
  --bg-radius-kpc 400 \
  --max-timestep-gyr 0.0005 \
  --dt-min-gyr 1e-6 \
  --time-end-gyr 10.0 \
  --snapshot-dt-myr 5.0 \
  --arm-strength 0.3 \
  --arm-stream-frac 0.02 \
  --Q-star 2.0 \
  --Q-gas 1.5 \
  --bulge-rmax-scale 50

"$SWIFT_BIN" $EAGLE_FLAGS --threads=8 hires_1gyr_relax_merge.yml

python ../../create_movie.py . \
  --width-kpc 620 \
  --npix 520 \
  --fps 12 \
  --dpi 160 \
  --bitrate 3200 \
  --gas-vmin 5e4 \
  --gas-vmax 2e8 \
  --stars-vmin 5e4 \
  --stars-vmax 2e9 \
  --title-prefix "100k-particle 1 Gyr relax-and-merge test"
