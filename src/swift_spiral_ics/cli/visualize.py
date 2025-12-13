"""CLI for visualizing SWIFT initial conditions."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm


def load_ic_data(filename: str) -> dict:
    """Load particle data from SWIFT IC file.

    Args:
        filename: IC HDF5 filename.

    Returns:
        Dict with particle data for each type.
    """
    data = {}

    with h5py.File(filename, "r") as f:
        # Load gas (PartType0)
        if "PartType0" in f:
            data["gas"] = {
                "pos": f["PartType0/Coordinates"][:],
                "vel": f["PartType0/Velocities"][:],
                "mass": f["PartType0/Masses"][:],
            }

        # Load DM (PartType1)
        if "PartType1" in f:
            data["dm"] = {
                "pos": f["PartType1/Coordinates"][:],
                "vel": f["PartType1/Velocities"][:],
                "mass": f["PartType1/Masses"][:],
            }

        # Load stars (PartType4)
        if "PartType4" in f:
            data["stars"] = {
                "pos": f["PartType4/Coordinates"][:],
                "vel": f["PartType4/Velocities"][:],
                "mass": f["PartType4/Masses"][:],
            }

        # Get box size
        data["box_size"] = f["Header"].attrs["BoxSize"]

    return data


def plot_projected_density(
    ax, pos, mass, box_size, title, plane="xy", bins=300, eps=1e-12
):
    """Plot projected surface density with log scaling for a given plane."""
    if len(pos) == 0:
        ax.text(0.5, 0.5, "No particles", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    center = box_size / 2.0
    shifted = pos - center
    if plane == "xy":
        x, y = shifted[:, 0], shifted[:, 1]
        xlabel, ylabel = "x (kpc)", "y (kpc)"
    elif plane == "xz":
        x, y = shifted[:, 0], shifted[:, 2]
        xlabel, ylabel = "x (kpc)", "z (kpc)"
    elif plane == "yz":
        x, y = shifted[:, 1], shifted[:, 2]
        xlabel, ylabel = "y (kpc)", "z (kpc)"
    else:
        raise ValueError(f"Unknown plane '{plane}'")

    hist_range = [[-box_size / 2, box_size / 2], [-box_size / 2, box_size / 2]]
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=hist_range, weights=mass)
    pixel_area = ((hist_range[0][1] - hist_range[0][0]) / bins) ** 2
    sigma = H.T / pixel_area

    vmax = np.percentile(sigma[sigma > 0], 99) if np.any(sigma > 0) else 1.0
    im = ax.imshow(
        sigma,
        origin="lower",
        extent=[*hist_range[0], *hist_range[1]],
        cmap="magma",
        norm=LogNorm(vmin=max(eps, sigma[sigma > 0].min()) if np.any(sigma > 0) else eps, vmax=vmax),
        aspect="equal",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=r"$\Sigma\ {\rm [M_\odot\,kpc^{-2}]}$")


def plot_rotation_curve(ax, pos, vel, mass, box_size, title):
    """Plot rotation curve.

    Args:
        ax: Matplotlib axis.
        pos: Particle positions (N, 3).
        vel: Particle velocities (N, 3).
        mass: Particle masses (N).
        box_size: Box size.
        title: Plot title.
    """
    if len(pos) == 0:
        ax.text(0.5, 0.5, "No particles", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    # Center coordinates
    center = box_size / 2.0
    x = pos[:, 0] - center
    y = pos[:, 1] - center

    vx = vel[:, 0]
    vy = vel[:, 1]

    # Cylindrical coordinates
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Azimuthal velocity
    v_phi = -vx * np.sin(phi) + vy * np.cos(phi)

    # Bin by radius
    R_bins = np.linspace(0, box_size / 2, 30)
    R_centers = 0.5 * (R_bins[1:] + R_bins[:-1])
    v_phi_mean = np.zeros(len(R_centers))
    v_phi_std = np.zeros(len(R_centers))

    for i in range(len(R_centers)):
        mask = (R >= R_bins[i]) & (R < R_bins[i + 1])
        if np.sum(mask) > 0:
            v_phi_mean[i] = np.mean(v_phi[mask])
            v_phi_std[i] = np.std(v_phi[mask])

    ax.plot(R_centers, v_phi_mean, "o-", label=title)
    ax.fill_between(R_centers, v_phi_mean - v_phi_std, v_phi_mean + v_phi_std, alpha=0.3)
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel(r"$v_\phi$ (km/s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_velocity_dispersion(ax, pos, vel, mass, box_size, title):
    """Plot velocity dispersion profiles.

    Args:
        ax: Matplotlib axis.
        pos: Particle positions (N, 3).
        vel: Particle velocities (N, 3).
        mass: Particle masses (N).
        box_size: Box size.
        title: Plot title.
    """
    if len(pos) == 0:
        ax.text(0.5, 0.5, "No particles", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    # Center coordinates
    center = box_size / 2.0
    x = pos[:, 0] - center
    y = pos[:, 1] - center
    # z = pos[:, 2] - center  # Not used

    vx = vel[:, 0]
    vy = vel[:, 1]
    vz = vel[:, 2]

    # Cylindrical coordinates
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Velocity components
    v_R = vx * np.cos(phi) + vy * np.sin(phi)
    v_phi = -vx * np.sin(phi) + vy * np.cos(phi)

    # Bin by radius
    R_bins = np.linspace(0, box_size / 2, 30)
    R_centers = 0.5 * (R_bins[1:] + R_bins[:-1])
    sigma_R = np.zeros(len(R_centers))
    sigma_phi = np.zeros(len(R_centers))
    sigma_z = np.zeros(len(R_centers))

    for i in range(len(R_centers)):
        mask = (R >= R_bins[i]) & (R < R_bins[i + 1])
        if np.sum(mask) > 0:
            sigma_R[i] = np.std(v_R[mask])
            sigma_phi[i] = np.std(v_phi[mask])
            sigma_z[i] = np.std(vz[mask])

    ax.plot(R_centers, sigma_R, "o-", label=r"$\sigma_R$")
    ax.plot(R_centers, sigma_phi, "s-", label=r"$\sigma_\phi$")
    ax.plot(R_centers, sigma_z, "^-", label=r"$\sigma_z$")
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel(r"$\sigma$ (km/s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_velocity_field(ax, pos, vel, mass, box_size, title, subsample=1000):
    """Plot 2D velocity field.

    Args:
        ax: Matplotlib axis.
        pos: Particle positions (N, 3).
        vel: Particle velocities (N, 3).
        mass: Particle masses (N).
        box_size: Box size.
        title: Plot title.
        subsample: Number of particles to subsample for quiver plot.
    """
    if len(pos) == 0:
        ax.text(0.5, 0.5, "No particles", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    # Subsample for clarity
    N = len(pos)
    if N > subsample:
        indices = np.random.choice(N, subsample, replace=False)
        pos = pos[indices]
        vel = vel[indices]

    x, y = pos[:, 0], pos[:, 1]
    vx, vy = vel[:, 0], vel[:, 1]

    # Plot quiver
    ax.quiver(x, y, vx, vy, alpha=0.5)
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_title(title)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect("equal")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize SWIFT initial conditions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ic_file", type=str, help="Input IC HDF5 file")
    parser.add_argument("--out-pdf", type=str, default="ic_diagnostics.pdf", help="Output PDF file")
    parser.add_argument(
        "--out-dir", type=str, default="ic_plots", help="Output directory for individual plots"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SWIFT IC DIAGNOSTICS")
    print("=" * 70)

    # Load data
    print(f"\nLoading IC file: {args.ic_file}")
    data = load_ic_data(args.ic_file)
    box_size = data["box_size"]

    print(f"Box size: {box_size:.2f} kpc")
    for comp_name in ["dm", "gas", "stars"]:
        if comp_name in data:
            N = len(data[comp_name]["pos"])
            M = np.sum(data[comp_name]["mass"])
            print(f"  {comp_name}: N={N}, M={M:.2e} Msun")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Create plots
    print("\nGenerating diagnostic plots...")

    with PdfPages(args.out_pdf) as pdf:
        # Projected surface density: face-on (xy) and edge-on (xz, yz)
        for comp_name in ["dm", "gas", "stars"]:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            if comp_name in data:
                for ax, plane, label in zip(
                    axes,
                    ["xy", "xz", "yz"],
                    ["Face-on (XY)", "Edge-on (XZ)", "Edge-on (YZ)"],
                ):
                    plot_projected_density(
                        ax,
                        data[comp_name]["pos"],
                        data[comp_name]["mass"],
                        box_size,
                        f"{comp_name.upper()} {label}",
                        plane=plane,
                    )
            else:
                for ax in axes:
                    ax.text(
                        0.5,
                        0.5,
                        f"No {comp_name}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{comp_name.upper()}")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.savefig(out_dir / f"{comp_name}_projections.png", dpi=150)
            plt.close()

        # Rotation curves
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for comp_name in ["gas", "stars"]:
            if comp_name in data:
                plot_rotation_curve(
                    ax,
                    data[comp_name]["pos"],
                    data[comp_name]["vel"],
                    data[comp_name]["mass"],
                    box_size,
                    comp_name.upper(),
                )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.savefig(out_dir / "rotation_curve.png", dpi=150)
        plt.close()

        # Velocity dispersions
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, comp_name in enumerate(["gas", "stars"]):
            if comp_name in data and i < 2:
                plot_velocity_dispersion(
                    axes[i],
                    data[comp_name]["pos"],
                    data[comp_name]["vel"],
                    data[comp_name]["mass"],
                    box_size,
                    f"{comp_name.upper()} Velocity Dispersion",
                )
            elif i < 2:
                axes[i].text(
                    0.5,
                    0.5,
                    f"No {comp_name}",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"{comp_name.upper()} Velocity Dispersion")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.savefig(out_dir / "velocity_dispersion.png", dpi=150)
        plt.close()

        # Velocity fields
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, comp_name in enumerate(["gas", "stars"]):
            if comp_name in data and i < 2:
                plot_velocity_field(
                    axes[i],
                    data[comp_name]["pos"],
                    data[comp_name]["vel"],
                    data[comp_name]["mass"],
                    box_size,
                    f"{comp_name.upper()} Velocity Field",
                )
            elif i < 2:
                axes[i].text(
                    0.5,
                    0.5,
                    f"No {comp_name}",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"{comp_name.upper()} Velocity Field")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.savefig(out_dir / "velocity_field.png", dpi=150)
        plt.close()

    print("\nDiagnostics saved:")
    print(f"  PDF: {args.out_pdf}")
    print(f"  PNG plots: {out_dir}/")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
