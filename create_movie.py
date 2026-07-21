#!/usr/bin/env python3
"""Create a two-panel movie from SWIFT snapshots in a directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_INTERNAL_TIME_TO_MYR = 977.784 * 1000.0


def _find_snapshots(snapshot_dir: Path, pattern: str) -> list[Path]:
    return sorted(snapshot_dir.glob(pattern))


def _project_surface_density(
    coords_mpc: np.ndarray,
    masses_internal: np.ndarray,
    box_mpc: float,
    width_kpc: float,
    npix: int,
) -> np.ndarray:
    coords_kpc = (coords_mpc - box_mpc / 2.0) * 1000.0
    half_width = width_kpc / 2.0
    mask = (
        (np.abs(coords_kpc[:, 0]) <= half_width)
        & (np.abs(coords_kpc[:, 1]) <= half_width)
        & (np.abs(coords_kpc[:, 2]) <= half_width)
    )
    hist, _, _ = np.histogram2d(
        coords_kpc[mask, 0],
        coords_kpc[mask, 1],
        bins=npix,
        range=[[-half_width, half_width], [-half_width, half_width]],
        weights=masses_internal[mask] * 1e10,
    )
    pixel_area = (width_kpc / npix) ** 2
    return hist.T / pixel_area


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an XY movie from all matching SWIFT snapshots in a directory."
    )
    parser.add_argument("snapshot_dir", type=Path, help="Directory containing snapshot files.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="snapshot_*.hdf5",
        help="Glob pattern used to find snapshots within the directory.",
    )
    parser.add_argument(
        "--out-movie",
        type=Path,
        default=None,
        help="Output movie path. Defaults to <snapshot_dir>/<snapshot_dir_name>_xy.mp4.",
    )
    parser.add_argument("--width-kpc", type=float, default=620.0, help="Field of view width in kpc.")
    parser.add_argument("--npix", type=int, default=420, help="Histogram resolution per axis.")
    parser.add_argument("--fps", type=int, default=12, help="Movie frame rate.")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI for rendered frames.")
    parser.add_argument("--bitrate", type=int, default=2800, help="ffmpeg target bitrate.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use for frame rendering.",
    )
    parser.add_argument(
        "--gas-vmin", type=float, default=5e4, help="Lower log-scale surface density limit for gas."
    )
    parser.add_argument(
        "--gas-vmax", type=float, default=2e8, help="Upper log-scale surface density limit for gas."
    )
    parser.add_argument(
        "--stars-vmin",
        type=float,
        default=5e4,
        help="Lower log-scale surface density limit for stars.",
    )
    parser.add_argument(
        "--stars-vmax",
        type=float,
        default=2e9,
        help="Upper log-scale surface density limit for stars.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default=None,
        help="Optional title prefix shown ahead of the time label.",
    )
    return parser.parse_args()


def _render_frame(task: tuple) -> str:
    (
        snap,
        frame_path,
        width_kpc,
        npix,
        gas_vmin,
        gas_vmax,
        stars_vmin,
        stars_vmax,
        title_prefix,
        dpi,
        frame_number,
        total_frames,
    ) = task

    extent = [-width_kpc / 2, width_kpc / 2, -width_kpc / 2, width_kpc / 2]

    with h5py.File(snap, "r") as f:
        box = float(np.atleast_1d(f["Header"].attrs["BoxSize"])[0])
        time_myr = float(np.atleast_1d(f["Header"].attrs["Time"])[0]) * _INTERNAL_TIME_TO_MYR
        gas = _project_surface_density(
            f["PartType0"]["Coordinates"][:],
            f["PartType0"]["Masses"][:],
            box,
            width_kpc,
            npix,
        )
        stars = _project_surface_density(
            f["PartType4"]["Coordinates"][:],
            f["PartType4"]["Masses"][:],
            box,
            width_kpc,
            npix,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")

    axes[0].imshow(
        gas,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(vmin=gas_vmin, vmax=gas_vmax),
        interpolation="nearest",
    )
    axes[1].imshow(
        stars,
        origin="lower",
        extent=extent,
        cmap="viridis",
        norm=LogNorm(vmin=stars_vmin, vmax=stars_vmax),
        interpolation="nearest",
    )
    axes[0].set_title("Gas surface density", color="white")
    axes[1].set_title("Stars + bulge surface density", color="white")
    prefix = f"{title_prefix}, " if title_prefix else ""
    fig.text(
        0.5,
        0.975,
        f"{prefix}t = {time_myr:6.1f} Myr, frame {frame_number}/{total_frames}",
        color="white",
        ha="center",
        va="top",
        fontsize=14,
    )

    fig.savefig(frame_path, dpi=dpi)
    plt.close(fig)
    return str(frame_path)


def main() -> int:
    args = _parse_args()
    snapshot_dir = args.snapshot_dir
    if not snapshot_dir.is_dir():
        print(f"Snapshot directory does not exist: {snapshot_dir}", file=sys.stderr)
        return 1

    snapshots = _find_snapshots(snapshot_dir, args.pattern)
    if not snapshots:
        print(
            f"No snapshots matching '{args.pattern}' found in {snapshot_dir}",
            file=sys.stderr,
        )
        return 1

    out_movie = args.out_movie or snapshot_dir / f"{snapshot_dir.name}_xy.mp4"
    workers = max(1, args.workers)

    with tempfile.TemporaryDirectory(prefix="snapshot_movie_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        tasks = []
        for i, snap in enumerate(snapshots, start=1):
            frame_path = tmp_dir / f"frame_{i:05d}.png"
            tasks.append(
                (
                    snap,
                    frame_path,
                    args.width_kpc,
                    args.npix,
                    args.gas_vmin,
                    args.gas_vmax,
                    args.stars_vmin,
                    args.stars_vmax,
                    args.title_prefix,
                    args.dpi,
                    i,
                    len(snapshots),
                )
            )

        if workers == 1:
            for task in tasks:
                _render_frame(task)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                list(executor.map(_render_frame, tasks))

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(tmp_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            f"{args.bitrate}k",
            str(out_movie),
        ]
        subprocess.run(cmd, check=True)

    print(out_movie)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
