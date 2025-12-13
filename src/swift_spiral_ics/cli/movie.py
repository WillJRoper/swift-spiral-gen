"""CLI for creating movies from SWIFT snapshots."""

import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def load_snapshot(filename: str) -> dict:
    """Load snapshot data.

    Args:
        filename: Snapshot HDF5 filename.

    Returns:
        Dict with particle data and metadata.
    """
    data = {}

    with h5py.File(filename, "r") as f:
        # Try to get time
        if "Header" in f:
            time_attr = f["Header"].attrs.get("Time", 0.0)
            if hasattr(time_attr, "__len__") and not isinstance(time_attr, str) and len(time_attr) > 0:
                data["time"] = float(time_attr[0])
            else:
                data["time"] = float(time_attr)

            box_size_attr = f["Header"].attrs.get("BoxSize", 100.0)
            if hasattr(box_size_attr, "__len__") and not isinstance(box_size_attr, str) and len(box_size_attr) > 0:
                data["box_size"] = float(box_size_attr[0])
            else:
                data["box_size"] = float(box_size_attr)
        else:
            data["time"] = 0.0
            data["box_size"] = 100.0

        # Load gas (PartType0) - handle plural names
        if "PartType0" in f:
            coords_key = "Coordinates" if "Coordinates" in f["PartType0"] else "Coordinate"
            mass_key = "Masses" if "Masses" in f["PartType0"] else "Mass"
            vel_key = "Velocities" if "Velocities" in f["PartType0"] else "Velocity"

            data["gas"] = {
                "pos": f[f"PartType0/{coords_key}"][:],
                "mass": f[f"PartType0/{mass_key}"][:],
            }

            if vel_key in f["PartType0"]:
                data["gas"]["vel"] = f[f"PartType0/{vel_key}"][:]

        # Load stars (PartType4)
        if "PartType4" in f:
            coords_key = "Coordinates" if "Coordinates" in f["PartType4"] else "Coordinate"
            mass_key = "Masses" if "Masses" in f["PartType4"] else "Mass"
            vel_key = "Velocities" if "Velocities" in f["PartType4"] else "Velocity"

            data["stars"] = {
                "pos": f[f"PartType4/{coords_key}"][:],
                "mass": f[f"PartType4/{mass_key}"][:],
            }

            if vel_key in f["PartType4"]:
                data["stars"]["vel"] = f[f"PartType4/{vel_key}"][:]

    return data


def render_snapshot(
    data: dict,
    box_size: float,
    bins: int = 512,
    show_vel: bool = False,
    vel_subsample: int = 500,
) -> np.ndarray:
    """Render snapshot to image.

    Args:
        data: Snapshot data dict.
        box_size: Box size (kpc).
        bins: Number of bins for density projection.
        show_vel: If True, overlay velocity vectors.
        vel_subsample: Number of particles to show for velocity field.

    Returns:
        RGB image array.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Combined density map (gas + stars)
    combined_density = np.zeros((bins, bins))

    for comp_name, color_weight in [("gas", 1.0), ("stars", 0.5)]:
        if comp_name in data:
            pos = data[comp_name]["pos"]
            mass = data[comp_name]["mass"]

            if len(pos) > 0:
                x, y = pos[:, 0], pos[:, 1]

                # Create 2D histogram
                H, xedges, yedges = np.histogram2d(
                    x, y, bins=bins, range=[[0, box_size], [0, box_size]], weights=mass
                )

                combined_density += H.T * color_weight

    # Plot density
    extent = [0, box_size, 0, box_size]
    ax.imshow(
        np.log10(combined_density + 1e-5),
        origin="lower",
        extent=extent,
        cmap="inferno",
        aspect="auto",
    )

    # Overlay velocity field if requested
    if show_vel and "gas" in data and "vel" in data["gas"]:
        pos = data["gas"]["pos"]
        vel = data["gas"]["vel"]

        if len(pos) > vel_subsample:
            indices = np.random.choice(len(pos), vel_subsample, replace=False)
            pos = pos[indices]
            vel = vel[indices]

        x, y = pos[:, 0], pos[:, 1]
        vx, vy = vel[:, 0], vel[:, 1]

        ax.quiver(x, y, vx, vy, color="white", alpha=0.3, scale=1000)

    # Add time label
    time_gyr = data.get("time", 0.0)
    ax.text(
        0.05,
        0.95,
        f"t = {time_gyr:.2f} Gyr",
        transform=ax.transAxes,
        fontsize=14,
        color="white",
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.5},
    )

    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,)) # 4 channels for ARGB

    plt.close(fig)

    return img


def create_movie_ffmpeg(frames: list, output_file: str, fps: int = 10) -> None:
    """Create movie using ffmpeg subprocess.

    Args:
        frames: List of frame image arrays.
        output_file: Output movie filename.
        fps: Frames per second.
    """
    # Save frames to temporary directory
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)

    for i, frame in enumerate(tqdm(frames, desc="Writing temp frames")):
        imageio.imwrite(temp_dir / f"frame_{i:05d}.png", frame)

    # Use ffmpeg to create movie
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(temp_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        output_file,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Movie created with ffmpeg: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {e.stderr.decode()}")
        print("Falling back to imageio...")
        imageio.mimsave(output_file, frames, fps=fps)
    except FileNotFoundError:
        print("ffmpeg not found. Using imageio...")
        imageio.mimsave(output_file, frames, fps=fps)

    # Clean up temp frames
    for file in temp_dir.glob("frame_*.png"):
        file.unlink()
    temp_dir.rmdir()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create movie from SWIFT snapshots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "snapshot_pattern", type=str, help="Snapshot file pattern (e.g., 'snapshot_*.hdf5')"
    )
    parser.add_argument("--out-movie", type=str, default="movie.mp4", help="Output movie file")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument(
        "--bins", type=int, default=512, help="Number of bins for density projection"
    )
    parser.add_argument("--show-vel", action="store_true", help="Show velocity field overlay")
    parser.add_argument(
        "--vel-subsample", type=int, default=500, help="Number of particles for velocity field"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SWIFT SNAPSHOT MOVIE MAKER")
    print("=" * 70)

    # Find snapshot files
    snapshot_files = sorted(Path(".").glob(args.snapshot_pattern))

    if len(snapshot_files) == 0:
        print(f"\nError: No files matching pattern '{args.snapshot_pattern}'")
        sys.exit(1)

    print(f"\nFound {len(snapshot_files)} snapshots")
    print(f"First: {snapshot_files[0]}")
    print(f"Last: {snapshot_files[-1]}")

    # Load first snapshot to get box size
    print("\nLoading first snapshot for metadata...")
    data = load_snapshot(str(snapshot_files[0]))
    box_size = data["box_size"]
    print(f"Box size: {box_size:.2f} kpc")

    # Render frames
    print("\nRendering frames...")
    frames = []

    for snap_file in tqdm(snapshot_files, desc="Rendering"):
        data = load_snapshot(str(snap_file))
        img = render_snapshot(data, box_size, args.bins, args.show_vel, args.vel_subsample)
        frames.append(img)

    # Create movie
    print(f"\nCreating movie: {args.out_movie}")
    create_movie_ffmpeg(frames, args.out_movie, args.fps)

    print("\n" + "=" * 70)
    print("MOVIE CREATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {args.out_movie}")
    print(f"Duration: {len(frames) / args.fps:.1f} seconds ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
