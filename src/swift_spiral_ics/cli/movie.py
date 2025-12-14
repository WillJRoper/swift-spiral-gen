"CLI for creating movies from SWIFT snapshots using swiftsimio."

import argparse
import subprocess
import sys
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import swiftsimio
import unyt
from swiftsimio.visualisation.projection import project_pixel_grid
from tqdm import tqdm


def render_snapshot(
    data: swiftsimio.SWIFTDataset,
    bins: int = 512,
    show_vel: bool = False,
    vel_subsample: int = 500,
    render_component: str = "combined",
) -> np.ndarray:
    """Render snapshot to image using swiftsimio.

    Args:
        data: Loaded swiftsimio dataset.
        bins: Number of bins for density projection.
        show_vel: If True, overlay velocity vectors.
        vel_subsample: Number of particles to show for velocity field.
        render_component: Which component to render ('gas', 'stars', 'dm', or 'combined').

    Returns:
        RGB image array (H, W, 3).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    box_size_unyt = data.metadata.boxsize[0].to("kpc")
    box_size = box_size_unyt.value

    # Project density map
    combined_density = np.zeros((bins, bins))
    
    # Define grid with units
    region = [0 * unyt.kpc, box_size_unyt, 0 * unyt.kpc, box_size_unyt] 
    
    if render_component == "gas" or render_component == "combined":
        if hasattr(data, "gas") and len(data.gas.coordinates) > 0:
            gas_map = project_pixel_grid(
                data=data.gas,
                resolution=bins,
                project="masses",
                parallel=True,
                region=region
            )
            combined_density += gas_map.value * 1.0 # Weight 1.0

    if render_component == "stars" or render_component == "combined":
        if hasattr(data, "stars") and len(data.stars.coordinates) > 0:
            star_map = project_pixel_grid(
                data=data.stars,
                resolution=bins,
                project="masses",
                parallel=True,
                region=region
            )
            combined_density += star_map.value * 0.5 # Weight 0.5

    if render_component == "dm" or render_component == "combined":
        if hasattr(data, "dm") and len(data.dm.coordinates) > 0:
            dm_map = project_pixel_grid(
                data=data.dm,
                resolution=bins,
                project="masses",
                parallel=True,
                region=region
            )
            combined_density += dm_map.value * 1.0 # Increased weight for DM to 1.0

    # Plot density
    extent = [0, box_size, 0, box_size]
    
    # Handle log scale safely
    density_log = np.log10(combined_density + 1e-10) # 1e-10 floor
    
    # Use swiftsimio-like colormap or inferno
    ax.imshow(
        density_log,
        origin="lower",
        extent=extent,
        cmap="inferno",
        aspect="equal",
    )

    # Overlay velocity field if requested (only for gas)
    if show_vel and hasattr(data, "gas") and hasattr(data.gas, "velocities"):
        pos = data.gas.coordinates.to("kpc").value
        vel = data.gas.velocities.to("km/s").value

        if len(pos) > vel_subsample:
            indices = np.random.choice(len(pos), vel_subsample, replace=False)
            pos = pos[indices]
            vel = vel[indices]

        x, y = pos[:, 0], pos[:, 1]
        vx, vy = vel[:, 0], vel[:, 1]

        # Only plot vectors inside the box
        mask = (x >= 0) & (x <= box_size) & (y >= 0) & (y <= box_size)
        
        ax.quiver(
            x[mask], y[mask], vx[mask], vy[mask], 
            color="white", alpha=0.3, scale=1000, width=0.002
        )

    # Add time label and component label
    time_gyr = data.metadata.time.to("Gyr").value
    ax.text(
        0.05,
        0.95,
        f"t = {time_gyr:.2f} Gyr ({render_component.upper()})",
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
    try:
        img = np.asarray(fig.canvas.buffer_rgba())
    except AttributeError:
        # Fallback for older mpl
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

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


def robust_glob(pattern_or_files):
    """Handle both glob patterns and file lists."""
    if isinstance(pattern_or_files, list):
        # Shell expanded
        files = []
        for p in pattern_or_files:
            path = Path(p)
            if path.is_file():
                files.append(path)
        return sorted(files)
    elif isinstance(pattern_or_files, str):
        return sorted(Path(".").glob(pattern_or_files))
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Create movie from SWIFT snapshots using swiftsimio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "snapshots", type=str, nargs="+", help="Snapshot files or pattern"
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
    parser.add_argument(
        "--separate-movies", action="store_true", help="Create separate movies for gas, stars, and combined"
    )

    args = parser.parse_args()
    
    print("=" * 70)
    print("SWIFT SNAPSHOT MOVIE MAKER (swiftsimio)")
    print("=" * 70)
    
    # Check if first arg is a glob pattern that didn't expand (e.g. quoted)
    if len(args.snapshots) == 1 and "*" in args.snapshots[0]:
        snapshot_files = sorted(Path(".").glob(args.snapshots[0]))
    else:
        snapshot_files = [Path(f) for f in args.snapshots]
        snapshot_files.sort()

    if len(snapshot_files) == 0:
        print(f"\nError: No files found.")
        sys.exit(1)

    print(f"\nFound {len(snapshot_files)} snapshots")
    print(f"First: {snapshot_files[0]}")
    print(f"Last: {snapshot_files[-1]}")

    components_to_render = ["gas", "stars"]
    if args.separate_movies:
        components_to_render.append("dm") # DM also

    # Render frames
    movies_to_create = {}
    if args.separate_movies:
        for comp in components_to_render:
            movies_to_create[comp] = []
        movies_to_create["combined"] = [] # Still do combined if separate is requested
    else:
        movies_to_create["combined"] = []

    for snap_file in tqdm(snapshot_files, desc="Loading and Rendering Snapshots"):
        data = swiftsimio.load(str(snap_file))
        
        # Debugging: check if DM data is loaded
        if hasattr(data, "dm") and hasattr(data.dm, "coordinates"):
            print(f"  Loaded DM particles: {len(data.dm.coordinates)} (total mass: {np.sum(data.dm.masses).to('Msun'):.2e})")
        else:
            print("  No DM particles found in swiftsimio data object.")
        
        if args.separate_movies:
            for comp in components_to_render:
                img = render_snapshot(data, args.bins, args.show_vel, args.vel_subsample, render_component=comp)
                movies_to_create[comp].append(img)
            # Also render combined
            img_combined = render_snapshot(data, args.bins, args.show_vel, args.vel_subsample, render_component="combined")
            movies_to_create["combined"].append(img_combined)
        else:
            img_combined = render_snapshot(data, args.bins, args.show_vel, args.vel_subsample, render_component="combined")
            movies_to_create["combined"].append(img_combined)

    for comp, frames_list in movies_to_create.items():
        if not frames_list:
            print(f"No frames rendered for {comp}.")
            continue
        
        output_filename = Path(args.out_movie).parent / f"{Path(args.out_movie).stem}_{comp}.mp4"
        print(f"\nCreating movie for {comp}: {output_filename}")
        create_movie_ffmpeg(frames_list, str(output_filename), args.fps)

    print("\n" + "=" * 70)
    print("MOVIE CREATION COMPLETE")
    print("=" * 70)
    print(f"\nDuration: {len(list(movies_to_create.values())[0]) / args.fps:.1f} seconds ({len(list(movies_to_create.values())[0])} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()