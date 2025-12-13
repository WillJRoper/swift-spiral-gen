"""Main CLI for generating SWIFT initial conditions."""

import argparse
import sys

import numpy as np

from ..io.swift_writer import print_ic_summary, write_swift_ic
from ..io.yaml_writer import (
    available_param_templates,
    generate_swift_params,
    print_yaml_summary,
    write_yaml_file,
)
from ..physics.orbits import (
    center_of_mass_correction,
    parabolic_orbit_initial_conditions,
    place_galaxy_in_orbit,
)
from ..physics.sampling import (
    sample_bulge_velocities,
    sample_disc_velocities,
    sample_exponential_disc,
    sample_halo_velocities,
    sample_hernquist_bulge,
    sample_nfw_halo,
)
from ..utils.random import get_rng


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SWIFT initial conditions for spiral disc galaxies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Global options
    parser.add_argument("--out-ics", type=str, required=True, help="Output IC HDF5 file path")
    parser.add_argument(
        "--out-params", type=str, required=True, help="Output YAML parameter file path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--box-kpc", type=float, default=500.0, help="Simulation box size (kpc)")
    parser.add_argument("--m-part-msun", type=float, required=True, help="Particle mass (Msun)")
    parser.add_argument("--n-galaxies", type=int, default=1, help="Number of galaxies (>=1)")

    # Simulation time parameters
    parser.add_argument("--time-end-gyr", type=float, default=2.0, help="Simulation end time (Gyr)")
    parser.add_argument(
        "--snapshot-dt-myr", type=float, default=10.0, help="Snapshot spacing (Myr)"
    )
    parser.add_argument(
        "--snapshot-basename", type=str, default="snapshot", help="Snapshot basename for YAML output"
    )
    parser.add_argument(
        "--param-template",
        type=str,
        default="eagle_isolated",
        choices=available_param_templates(),
        help="Packaged SWIFT parameter template to start from",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MetaData run_name override for the parameter file",
    )

    # Per-galaxy halo parameters
    parser.add_argument(
        "--m200-msun", type=float, nargs="+", required=True, help="Halo M200 mass per galaxy (Msun)"
    )
    parser.add_argument(
        "--c200", type=float, nargs="+", required=True, help="Halo concentration per galaxy"
    )

    # Per-galaxy baryon parameters
    parser.add_argument(
        "--m-star-msun",
        type=float,
        nargs="+",
        required=True,
        help="Total stellar mass per galaxy (Msun)",
    )
    parser.add_argument(
        "--m-gas-msun",
        type=float,
        nargs="+",
        required=True,
        help="Total gas mass per galaxy (Msun)",
    )
    parser.add_argument(
        "--dt", type=float, nargs="+", required=True, help="Disc-to-total stellar ratio per galaxy"
    )
    parser.add_argument(
        "--rd-kpc",
        type=float,
        nargs="+",
        required=True,
        help="Stellar disc scale length per galaxy (kpc)",
    )
    parser.add_argument(
        "--zd-kpc",
        type=float,
        nargs="+",
        required=True,
        help="Stellar disc scale height per galaxy (kpc)",
    )
    parser.add_argument(
        "--rg-kpc",
        type=float,
        nargs="+",
        required=True,
        help="Gas disc scale length per galaxy (kpc)",
    )
    parser.add_argument(
        "--zg-kpc",
        type=float,
        nargs="+",
        required=True,
        help="Gas disc scale height per galaxy (kpc)",
    )
    parser.add_argument(
        "--bulge-a-kpc",
        type=float,
        nargs="+",
        required=True,
        help="Bulge scale length per galaxy (kpc)",
    )

    # Spiral/bar parameters
    parser.add_argument(
        "--n-arms", type=int, nargs="+", default=None, help="Number of spiral arms per galaxy"
    )
    parser.add_argument(
        "--pitch-deg",
        type=float,
        nargs="+",
        default=None,
        help="Spiral pitch angle per galaxy (deg)",
    )
    parser.add_argument(
        "--arm-strength", type=float, nargs="+", default=None, help="Spiral arm strength per galaxy"
    )
    parser.add_argument(
        "--arm-stream-frac",
        type=float,
        nargs="+",
        default=None,
        help="Spiral streaming fraction per galaxy",
    )
    parser.add_argument(
        "--bar", type=int, nargs="+", default=None, help="Bar enabled (0/1) per galaxy"
    )
    parser.add_argument(
        "--bar-r-kpc", type=float, nargs="+", default=None, help="Bar radius per galaxy (kpc)"
    )
    parser.add_argument(
        "--bar-q", type=float, nargs="+", default=None, help="Bar axis ratio per galaxy"
    )
    parser.add_argument(
        "--bar-stream-frac",
        type=float,
        nargs="+",
        default=None,
        help="Bar streaming fraction per galaxy",
    )

    # Stability parameters
    parser.add_argument(
        "--q-star", type=float, default=1.5, help="Toomre Q for stellar disc (global)"
    )
    parser.add_argument("--q-gas", type=float, default=2.0, help="Toomre Q for gas disc (global)")

    # Multi-galaxy encounter parameters
    parser.add_argument(
        "--r-init-kpc",
        type=float,
        nargs="+",
        default=None,
        help="Initial separation for secondaries (kpc)",
    )
    parser.add_argument(
        "--r-peri-kpc",
        type=float,
        nargs="+",
        default=None,
        help="Pericentre distance for secondaries (kpc)",
    )
    parser.add_argument(
        "--orbit-plane-deg",
        type=float,
        nargs="+",
        default=None,
        help="Orbit plane angle for secondaries (deg)",
    )
    parser.add_argument(
        "--inclination-deg",
        type=float,
        nargs="+",
        default=None,
        help="Disc inclination for secondaries (deg)",
    )
    parser.add_argument(
        "--node-deg",
        type=float,
        nargs="+",
        default=None,
        help="Ascending node angle for secondaries (deg)",
    )

    return parser.parse_args()


def validate_args(args):
    """Validate command-line arguments."""
    n_gal = args.n_galaxies

    # Check lengths of per-galaxy arrays
    required_per_galaxy = {
        "m200-msun": args.m200_msun,
        "c200": args.c200,
        "m-star-msun": args.m_star_msun,
        "m-gas-msun": args.m_gas_msun,
        "dt": args.dt,
        "rd-kpc": args.rd_kpc,
        "zd-kpc": args.zd_kpc,
        "rg-kpc": args.rg_kpc,
        "zg-kpc": args.zg_kpc,
        "bulge-a-kpc": args.bulge_a_kpc,
    }

    for name, val_list in required_per_galaxy.items():
        if len(val_list) != n_gal:
            print(f"Error: --{name} must have {n_gal} values (one per galaxy), got {len(val_list)}")
            sys.exit(1)

    # Check secondary galaxy orbital parameters
    if n_gal > 1:
        n_secondaries = n_gal - 1

        if args.r_init_kpc is None or len(args.r_init_kpc) != n_secondaries:
            print(f"Error: --r-init-kpc required for {n_secondaries} secondary galaxies")
            sys.exit(1)

        if args.r_peri_kpc is None or len(args.r_peri_kpc) != n_secondaries:
            print(f"Error: --r-peri-kpc required for {n_secondaries} secondary galaxies")
            sys.exit(1)

        # Optional orientation parameters
        if args.inclination_deg is None:
            args.inclination_deg = [0.0] * n_secondaries
        if args.node_deg is None:
            args.node_deg = [0.0] * n_secondaries
        if args.orbit_plane_deg is None:
            args.orbit_plane_deg = [0.0] * n_secondaries

    # Set default spiral/bar parameters if not provided
    if args.n_arms is None:
        args.n_arms = [2] * n_gal
    if args.pitch_deg is None:
        args.pitch_deg = [15.0] * n_gal
    if args.arm_strength is None:
        args.arm_strength = [0.2] * n_gal
    if args.arm_stream_frac is None:
        args.arm_stream_frac = [0.1] * n_gal
    if args.bar is None:
        args.bar = [0] * n_gal
    if args.bar_r_kpc is None:
        args.bar_r_kpc = [3.0] * n_gal
    if args.bar_q is None:
        args.bar_q = [0.3] * n_gal
    if args.bar_stream_frac is None:
        args.bar_stream_frac = [0.1] * n_gal

    # Validate spiral/bar parameter lengths
    if len(args.n_arms) != n_gal:
        args.n_arms = args.n_arms * n_gal if len(args.n_arms) == 1 else args.n_arms[:n_gal]
    if len(args.pitch_deg) != n_gal:
        args.pitch_deg = (
            args.pitch_deg * n_gal if len(args.pitch_deg) == 1 else args.pitch_deg[:n_gal]
        )
    if len(args.arm_strength) != n_gal:
        args.arm_strength = (
            args.arm_strength * n_gal if len(args.arm_strength) == 1 else args.arm_strength[:n_gal]
        )
    if len(args.arm_stream_frac) != n_gal:
        args.arm_stream_frac = (
            args.arm_stream_frac * n_gal
            if len(args.arm_stream_frac) == 1
            else args.arm_stream_frac[:n_gal]
        )
    if len(args.bar) != n_gal:
        args.bar = args.bar * n_gal if len(args.bar) == 1 else args.bar[:n_gal]
    if len(args.bar_r_kpc) != n_gal:
        args.bar_r_kpc = (
            args.bar_r_kpc * n_gal if len(args.bar_r_kpc) == 1 else args.bar_r_kpc[:n_gal]
        )
    if len(args.bar_q) != n_gal:
        args.bar_q = args.bar_q * n_gal if len(args.bar_q) == 1 else args.bar_q[:n_gal]
    if len(args.bar_stream_frac) != n_gal:
        args.bar_stream_frac = (
            args.bar_stream_frac * n_gal
            if len(args.bar_stream_frac) == 1
            else args.bar_stream_frac[:n_gal]
        )


def generate_galaxy(
    galaxy_idx: int,
    args,
    rng: np.random.Generator,
) -> dict:
    """Generate a single galaxy.

    Args:
        galaxy_idx: Index of galaxy (0 = primary).
        args: Command-line arguments.
        rng: Random number generator.

    Returns:
        Dict with 'dm', 'gas', 'stars' keys containing particle data.
    """
    print(f"\nGenerating galaxy {galaxy_idx}...")

    # Extract parameters for this galaxy
    m200 = args.m200_msun[galaxy_idx]
    c200 = args.c200[galaxy_idx]
    m_star = args.m_star_msun[galaxy_idx]
    m_gas = args.m_gas_msun[galaxy_idx]
    dt_ratio = args.dt[galaxy_idx]
    R_d = args.rd_kpc[galaxy_idx]
    z_d = args.zd_kpc[galaxy_idx]
    R_g = args.rg_kpc[galaxy_idx]
    z_g = args.zg_kpc[galaxy_idx]
    a_bulge = args.bulge_a_kpc[galaxy_idx]

    # Spiral/bar parameters
    spiral_params = {
        "n_arms": args.n_arms[galaxy_idx],
        "pitch_deg": args.pitch_deg[galaxy_idx],
        "arm_strength": args.arm_strength[galaxy_idx],
        "stream_frac": args.arm_stream_frac[galaxy_idx],
    }

    bar_enabled = args.bar[galaxy_idx] > 0
    bar_params = {
        "enabled": bar_enabled,
        "radius": args.bar_r_kpc[galaxy_idx],
        "q": args.bar_q[galaxy_idx],
        "stream_frac": args.bar_stream_frac[galaxy_idx],
        "angle": 0.0,
        "strength": 0.5 if bar_enabled else 0.0,
    }

    # Calculate component masses
    m_disc_star = m_star * dt_ratio
    m_bulge = m_star * (1 - dt_ratio)

    # Calculate particle numbers
    N_dm = int(round(m200 / args.m_part_msun))
    N_gas = int(round(m_gas / args.m_part_msun))
    N_disc_star = int(round(m_disc_star / args.m_part_msun))
    N_bulge = int(round(m_bulge / args.m_part_msun)) if m_bulge > 0 else 0

    print(f"  Halo: N={N_dm}, M200={m200:.2e} Msun, c200={c200:.2f}")
    print(f"  Gas disc: N={N_gas}, M={m_gas:.2e} Msun, R_d={R_g:.2f} kpc, z_d={z_g:.2f} kpc")
    print(
        f"  Stellar disc: N={N_disc_star}, M={m_disc_star:.2e} Msun, R_d={R_d:.2f} kpc, z_d={z_d:.2f} kpc"
    )
    if N_bulge > 0:
        print(f"  Bulge: N={N_bulge}, M={m_bulge:.2e} Msun, a={a_bulge:.2f} kpc")
    print(
        f"  Spiral arms: n={spiral_params['n_arms']}, pitch={spiral_params['pitch_deg']:.1f} deg, strength={spiral_params['arm_strength']:.2f}"
    )
    if bar_enabled:
        print(f"  Bar: r={bar_params['radius']:.2f} kpc, q={bar_params['q']:.2f}")

    # Sample particles
    print("  Sampling halo particles...")
    from ..physics.profiles import nfw_params

    r_s, _ = nfw_params(m200, c200)
    r_max = 10 * r_s  # Truncate at 10 * r_s

    x_dm, y_dm, z_dm = sample_nfw_halo(N_dm, m200, c200, r_max, rng)

    print("  Sampling gas disc particles...")
    x_gas, y_gas, z_gas = sample_exponential_disc(
        N_gas, m_gas, R_g, z_g, rng, spiral_params=spiral_params, bar_params=bar_params
    )

    print("  Sampling stellar disc particles...")
    x_disc, y_disc, z_disc = sample_exponential_disc(
        N_disc_star, m_disc_star, R_d, z_d, rng, spiral_params=spiral_params, bar_params=bar_params
    )

    if N_bulge > 0:
        print("  Sampling bulge particles...")
        x_bulge, y_bulge, z_bulge = sample_hernquist_bulge(N_bulge, m_bulge, a_bulge, rng)
    else:
        x_bulge = y_bulge = z_bulge = np.array([])

    # Sample velocities
    print("  Sampling halo velocities...")
    vx_dm, vy_dm, vz_dm = sample_halo_velocities(
        x_dm, y_dm, z_dm, m200, c200, m_bulge, a_bulge, m_disc_star, R_d, z_d, m_gas, R_g, z_g, rng
    )

    print("  Sampling gas disc velocities...")
    vx_gas, vy_gas, vz_gas = sample_disc_velocities(
        x_gas,
        y_gas,
        z_gas,
        m_gas,
        R_g,
        z_g,
        args.q_gas,
        m200,
        c200,
        m_bulge,
        a_bulge,
        m_disc_star,
        R_d,
        z_d,
        m_gas,
        R_g,
        z_g,
        rng,
        spiral_params=spiral_params,
        bar_params=bar_params,
        is_gas=True,
    )

    print("  Sampling stellar disc velocities...")
    vx_disc, vy_disc, vz_disc = sample_disc_velocities(
        x_disc,
        y_disc,
        z_disc,
        m_disc_star,
        R_d,
        z_d,
        args.q_star,
        m200,
        c200,
        m_bulge,
        a_bulge,
        m_disc_star,
        R_d,
        z_d,
        m_gas,
        R_g,
        z_g,
        rng,
        spiral_params=spiral_params,
        bar_params=bar_params,
        is_gas=False,
    )

    if N_bulge > 0:
        print("  Sampling bulge velocities...")
        vx_bulge, vy_bulge, vz_bulge = sample_bulge_velocities(
            x_bulge, y_bulge, z_bulge, m200, c200, m_bulge, a_bulge, m_disc_star, R_d, z_d, rng
        )
    else:
        vx_bulge = vy_bulge = vz_bulge = np.array([])

    # Combine stellar components
    x_star = np.concatenate([x_disc, x_bulge])
    y_star = np.concatenate([y_disc, y_bulge])
    z_star = np.concatenate([z_disc, z_bulge])
    vx_star = np.concatenate([vx_disc, vx_bulge])
    vy_star = np.concatenate([vy_disc, vy_bulge])
    vz_star = np.concatenate([vz_disc, vz_bulge])

    # Package particle data
    galaxy_data = {
        "dm": {
            "pos": np.column_stack([x_dm, y_dm, z_dm]),
            "vel": np.column_stack([vx_dm, vy_dm, vz_dm]),
        },
        "gas": {
            "pos": np.column_stack([x_gas, y_gas, z_gas]),
            "vel": np.column_stack([vx_gas, vy_gas, vz_gas]),
        },
        "stars": {
            "pos": np.column_stack([x_star, y_star, z_star]),
            "vel": np.column_stack([vx_star, vy_star, vz_star]),
        },
    }

    return galaxy_data


def main():
    """Main entry point."""
    print("=" * 70)
    print("SWIFT SPIRAL ICs - Initial Conditions Generator")
    print("=" * 70)

    args = parse_args()
    validate_args(args)

    # Initialize RNG
    rng = get_rng(args.seed)

    # Generate galaxies
    galaxies = []
    for i in range(args.n_galaxies):
        galaxy_data = generate_galaxy(i, args, rng)
        galaxies.append(galaxy_data)

    # Place galaxies in orbits (if multiple galaxies)
    if args.n_galaxies > 1:
        print("\n" + "=" * 70)
        print("CONFIGURING MERGER ORBITS")
        print("=" * 70)

        # Primary galaxy is at origin with zero velocity
        # Calculate total masses
        M_primary = args.m200_msun[0] + args.m_star_msun[0] + args.m_gas_msun[0]

        # Place secondary galaxies
        for i in range(1, args.n_galaxies):
            sec_idx = i - 1
            M_secondary = args.m200_msun[i] + args.m_star_msun[i] + args.m_gas_msun[i]

            r_init = args.r_init_kpc[sec_idx]
            r_peri = args.r_peri_kpc[sec_idx]
            orbit_plane = args.orbit_plane_deg[sec_idx]
            inclination = args.inclination_deg[sec_idx]
            node = args.node_deg[sec_idx]

            print(f"\nSecondary galaxy {i}:")
            print(f"  r_init = {r_init:.2f} kpc, r_peri = {r_peri:.2f} kpc")
            print(f"  Orbit plane angle = {orbit_plane:.1f} deg")
            print(f"  Disc inclination = {inclination:.1f} deg, node = {node:.1f} deg")

            # Calculate orbit
            orbit_pos, orbit_vel = parabolic_orbit_initial_conditions(
                M_primary, M_secondary, r_init, r_peri, orbit_plane
            )

            # Place galaxy
            secondary = galaxies[i]
            for comp_name in ["dm", "gas", "stars"]:
                if len(secondary[comp_name]["pos"]) > 0:
                    pos = secondary[comp_name]["pos"]
                    vel = secondary[comp_name]["vel"]

                    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
                    vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]

                    x_new, y_new, z_new, vx_new, vy_new, vz_new = place_galaxy_in_orbit(
                        x, y, z, vx, vy, vz, orbit_pos, orbit_vel, inclination, node
                    )

                    secondary[comp_name]["pos"] = np.column_stack([x_new, y_new, z_new])
                    secondary[comp_name]["vel"] = np.column_stack([vx_new, vy_new, vz_new])

        # Apply COM correction
        print("\nApplying center-of-mass correction...")
        all_masses = []
        all_positions = []
        all_velocities = []

        for galaxy in galaxies:
            for comp_name in ["dm", "gas", "stars"]:
                if len(galaxy[comp_name]["pos"]) > 0:
                    N = len(galaxy[comp_name]["pos"])
                    all_masses.append(np.full(N, args.m_part_msun))
                    all_positions.append(tuple(galaxy[comp_name]["pos"].T))
                    all_velocities.append(tuple(galaxy[comp_name]["vel"].T))

        pos_corrected, vel_corrected = center_of_mass_correction(
            all_masses, all_positions, all_velocities
        )

        # Reassign corrected values
        idx = 0
        for galaxy in galaxies:
            for comp_name in ["dm", "gas", "stars"]:
                if len(galaxy[comp_name]["pos"]) > 0:
                    x, y, z = pos_corrected[idx]
                    vx, vy, vz = vel_corrected[idx]
                    galaxy[comp_name]["pos"] = np.column_stack([x, y, z])
                    galaxy[comp_name]["vel"] = np.column_stack([vx, vy, vz])
                    idx += 1

    # Combine all galaxies
    print("\n" + "=" * 70)
    print("COMBINING GALAXIES")
    print("=" * 70)

    combined_data = {
        "dm": {"pos": [], "vel": []},
        "gas": {"pos": [], "vel": []},
        "stars": {"pos": [], "vel": []},
    }

    for galaxy in galaxies:
        for comp_name in ["dm", "gas", "stars"]:
            if len(galaxy[comp_name]["pos"]) > 0:
                combined_data[comp_name]["pos"].append(galaxy[comp_name]["pos"])
                combined_data[comp_name]["vel"].append(galaxy[comp_name]["vel"])

    for comp_name in ["dm", "gas", "stars"]:
        if combined_data[comp_name]["pos"]:
            combined_data[comp_name]["pos"] = np.vstack(combined_data[comp_name]["pos"])
            combined_data[comp_name]["vel"] = np.vstack(combined_data[comp_name]["vel"])
        else:
            combined_data[comp_name]["pos"] = np.array([]).reshape(0, 3)
            combined_data[comp_name]["vel"] = np.array([]).reshape(0, 3)

    # Center in box
    print("\nCentering in simulation box...")
    box_center = args.box_kpc / 2.0
    for comp_name in ["dm", "gas", "stars"]:
        if len(combined_data[comp_name]["pos"]) > 0:
            combined_data[comp_name]["pos"] += box_center

    # Write IC file
    print("\n" + "=" * 70)
    print("WRITING OUTPUT FILES")
    print("=" * 70)
    print(f"\nWriting IC file: {args.out_ics}")

    requested_masses = {
        "dm": sum(args.m200_msun),
        "gas": sum(args.m_gas_msun),
        "stars": sum(args.m_star_msun),
    }

    write_swift_ic(args.out_ics, args.box_kpc, combined_data, args.m_part_msun)

    print_ic_summary(combined_data, args.m_part_msun, requested_masses, args.box_kpc)

    # Write YAML parameter file
    print(f"Writing YAML parameter file: {args.out_params}")
    params = generate_swift_params(
        ic_filename=args.out_ics,
        box_size=args.box_kpc,
        time_end_gyr=args.time_end_gyr,
        snapshot_dt_myr=args.snapshot_dt_myr,
        output_basename=args.snapshot_basename,
        run_name=args.run_name,
        param_template=args.param_template,
    )
    write_yaml_file(args.out_params, params)
    print_yaml_summary(args.out_params, args.time_end_gyr, args.snapshot_dt_myr)

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print("\nTo run SWIFT:")
    print(f"  swift --hydro --self-gravity --stars --feedback --threads=<N> {args.out_params}")


if __name__ == "__main__":
    main()
