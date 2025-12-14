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
from ..physics.grid_solver import GalaxyGridSolver # Import new solver
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
    
    # Grid solver parameters
    parser.add_argument("--R-grid-kpc", type=float, default=50.0, help="Radial extent of potential grid (kpc)")
    parser.add_argument("--z-grid-kpc", type=float, default=20.0, help="Vertical extent of potential grid (kpc)")
    parser.add_argument("--nR-grid", type=int, default=64, help="Number of radial grid bins")
    parser.add_argument("--nz-grid", type=int, default=64, help="Number of vertical grid bins")
    parser.add_argument("--eps-grid", type=float, default=0.1, help="Softening length for grid potential (kpc)")

    # Simulation time parameters
    parser.add_argument("--time-end-gyr", type=float, default=2.0, help="Simulation end time (Gyr)")
    parser.add_argument(
        "--snapshot-dt-myr", type=float, default=0.0001, help="Snapshot spacing (Myr)"
    )
    parser.add_argument(
        "--snapshot-basename", type=str, default="snapshot", help="Snapshot basename for YAML output"
    )
    parser.add_argument(
        "--dt-min-gyr",
        type=float,
        default=1e-15,
        help="Minimum physical timestep (Gyr). This will override dt_min in the parameter file.",
    )
    parser.add_argument(
        "--param-template",
        type=str,
        default=available_param_templates()[0],
        choices=available_param_templates(),
        help="Packaged SWIFT parameter template to start from",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MetaData run_name override for the parameter file",
    )
    parser.add_argument(
        "--bg-gas-density-msun-kpc3",
        type=float,
        default=0.0,
        help="Uniform background gas density (Msun / kpc^3); 0 disables background gas",
    )
    parser.add_argument(
        "--bg-dm-density-msun-kpc3",
        type=float,
        default=0.0,
        help="Uniform background dark matter density (Msun / kpc^3); 0 disables background DM",
    )
    parser.add_argument(
        "--bg-grid-kpc",
        type=float,
        default=0.0,
        help="Optional regular grid spacing (kpc) for background gas+DM; 0 disables the grid",
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


def generate_galaxy_positions(
    galaxy_idx: int,
    args,
    rng: np.random.Generator,
) -> dict:
    """Generate positions and masses for a single galaxy.

    Args:
        galaxy_idx: Index of galaxy (0 = primary).
        args: Command-line arguments.
        rng: Random number generator.

    Returns:
        Dict with 'dm', 'gas', 'stars' keys, each containing
        'pos' (N,3) and 'mass' (N) arrays.
    """
    print(f"\nGenerating positions for galaxy {galaxy_idx}...")

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
    print("  Sampling halo positions...")
    r_s, _ = nfw_params(m200, c200)
    r_max = 10 * r_s  # Truncate at 10 * r_s

    x_dm, y_dm, z_dm = sample_nfw_halo(N_dm, m200, c200, r_max, rng)

    print("  Sampling gas disc positions...")
    x_gas, y_gas, z_gas = sample_exponential_disc(
        N_gas, m_gas, R_g, z_g, rng, spiral_params=spiral_params, bar_params=bar_params
    )

    print("  Sampling stellar disc positions...")
    x_disc, y_disc, z_disc = sample_exponential_disc(
        N_disc_star, m_disc_star, R_d, z_d, rng, spiral_params=spiral_params, bar_params=bar_params
    )

    if N_bulge > 0:
        print("  Sampling bulge positions...")
        x_bulge, y_bulge, z_bulge = sample_hernquist_bulge(N_bulge, m_bulge, a_bulge, rng)
    else:
        x_bulge = y_bulge = z_bulge = np.array([])

    # Combine stellar components
    x_star = np.concatenate([x_disc, x_bulge])
    y_star = np.concatenate([y_disc, y_bulge])
    z_star = np.concatenate([z_disc, z_bulge])
    
    # Package particle data (positions and masses only)
    galaxy_pos_mass_data = {
        "dm": {
            "pos": np.column_stack([x_dm, y_dm, z_dm]),
            "mass": np.full(N_dm, args.m_part_msun),
        },
        "gas": {
            "pos": np.column_stack([x_gas, y_gas, z_gas]),
            "mass": np.full(N_gas, args.m_part_msun),
        },
        "stars": {
            "pos": np.column_stack([x_star, y_star, z_star]),
            "mass": np.full(len(x_star), args.m_part_msun),
        },
    }

    return galaxy_pos_mass_data


def add_uniform_background(
    combined_data: dict,
    box_size: float,
    m_part: float,
    rho_gas: float,
    rho_dm: float,
    grid_spacing: float,
    rng: np.random.Generator,
) -> dict:
    """Add uniform background gas and DM to the combined particle set."""
    volume = box_size**3
    updated = dict(combined_data)

    # 1. Random Background (using fixed m_part)
    # Only if density is set but grid is NOT (or we add on top?)
    # Current logic: add random if rho > 0.
    # New logic: If grid_spacing > 0, we assume the user wants the density to be satisfied by the grid.
    # So we should ONLY do random sampling if grid_spacing == 0.
    
    use_grid = grid_spacing > 0

    if not use_grid:
        if rho_dm > 0:
            n_dm = int(round(rho_dm * volume / m_part))
            if n_dm > 0:
                pos_dm = rng.uniform(0, box_size, (n_dm, 3))
                vel_dm = np.zeros((n_dm, 3), dtype=float)
                mass_dm = np.full(n_dm, m_part)
                
                if updated["dm"]["pos"].size == 0:
                    updated["dm"]["pos"] = pos_dm
                    updated["dm"]["vel"] = vel_dm
                    updated["dm"]["mass"] = mass_dm
                else:
                    updated["dm"]["pos"] = np.vstack([updated["dm"]["pos"], pos_dm])
                    updated["dm"]["vel"] = np.vstack([updated["dm"]["vel"], vel_dm])
                    updated["dm"]["mass"] = np.concatenate([updated["dm"]["mass"], mass_dm])
                print(f"  Added uniform DM background (random): N={n_dm}, rho={rho_dm:.3e} Msun/kpc^3")

        if rho_gas > 0:
            n_gas = int(round(rho_gas * volume / m_part))
            if n_gas > 0:
                pos_gas = rng.uniform(0, box_size, (n_gas, 3))
                vel_gas = np.zeros((n_gas, 3), dtype=float)
                mass_gas = np.full(n_gas, m_part)
                
                if updated["gas"]["pos"].size == 0:
                    updated["gas"]["pos"] = pos_gas
                    updated["gas"]["vel"] = vel_gas
                    updated["gas"]["mass"] = mass_gas
                else:
                    updated["gas"]["pos"] = np.vstack([updated["gas"]["pos"], pos_gas])
                    updated["gas"]["vel"] = np.vstack([updated["gas"]["vel"], vel_gas])
                    updated["gas"]["mass"] = np.concatenate([updated["gas"]["mass"], mass_gas])
                print(f"  Added uniform gas background (random): N={n_gas}, rho={rho_gas:.3e} Msun/kpc^3")

    # 2. Grid Background (variable mass)
    else:
        coords_1d = np.arange(0, box_size, grid_spacing)
        if coords_1d.size > 0:
            gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
            grid_positions = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
            n_grid = len(grid_positions)
            
            # Add small random jitter
            jitter_gas = rng.normal(scale=0.1 * grid_spacing, size=grid_positions.shape)
            jitter_dm = rng.normal(scale=0.1 * grid_spacing, size=grid_positions.shape)
            gas_grid = np.mod(grid_positions + jitter_gas, box_size)
            dm_grid = np.mod(grid_positions + jitter_dm, box_size)
            
            # Calculate mass per particle to match target density
            # M_tot = rho * V
            # m_bg = M_tot / N_grid
            
            if rho_gas > 0:
                m_gas_bg = (rho_gas * volume) / n_grid
                vel_gas = np.zeros((n_grid, 3), dtype=float)
                mass_gas = np.full(n_grid, m_gas_bg)
                
                if updated["gas"]["pos"].size == 0:
                    updated["gas"]["pos"] = gas_grid.copy()
                    updated["gas"]["vel"] = vel_gas
                    updated["gas"]["mass"] = mass_gas
                else:
                    updated["gas"]["pos"] = np.vstack([updated["gas"]["pos"], gas_grid])
                    updated["gas"]["vel"] = np.vstack([updated["gas"]["vel"], vel_gas])
                    updated["gas"]["mass"] = np.concatenate([updated["gas"]["mass"], mass_gas])
                print(f"  Added grid gas background: spacing={grid_spacing} kpc, N={n_grid}, m={m_gas_bg:.2e} Msun")

            if rho_dm > 0:
                m_dm_bg = (rho_dm * volume) / n_grid
                vel_dm = np.zeros((n_grid, 3), dtype=float)
                mass_dm = np.full(n_grid, m_dm_bg)
                
                if updated["dm"]["pos"].size == 0:
                    updated["dm"]["pos"] = dm_grid.copy()
                    updated["dm"]["vel"] = vel_dm
                    updated["dm"]["mass"] = mass_dm
                else:
                    updated["dm"]["pos"] = np.vstack([updated["dm"]["pos"], dm_grid])
                    updated["dm"]["vel"] = np.vstack([updated["dm"]["vel"], vel_dm])
                    updated["dm"]["mass"] = np.concatenate([updated["dm"]["mass"], mass_dm])
                print(f"  Added grid DM background: spacing={grid_spacing} kpc, N={n_grid}, m={m_dm_bg:.2e} Msun")

    return updated


def compute_cosmic_background_densities(
    h: float = 0.6777, omega_cdm: float = 0.2587481, omega_b: float = 0.0482519
) -> tuple[float, float]:
    """Return cosmic-mean DM and gas densities in Msun/kpc^3 using the template cosmology."""
    G_kpc = 4.30091e-6  # (km/s)^2 kpc / Msun
    H0_kpc = 0.1 * h  # 100 km/s/Mpc -> 0.1 km/s/kpc
    rho_crit = 3 * H0_kpc**2 / (8 * np.pi * G_kpc)  # Msun/kpc^3
    rho_dm = rho_crit * omega_cdm
    rho_gas = rho_crit * omega_b
    return rho_dm, rho_gas


def main():
    """Main entry point."""
    print("=" * 70)
    print("SWIFT SPIRAL ICs - Initial Conditions Generator")
    print("=" * 70)

    args = parse_args()
    validate_args(args)

    # Initialize RNG
    rng = get_rng(args.seed)

    # 1. Generate positions and masses for all galaxies
    galaxies_pos_mass = []
    for i in range(args.n_galaxies):
        galaxy_pos_mass_data = generate_galaxy_positions(i, args, rng)
        galaxies_pos_mass.append(galaxy_pos_mass_data)

    # 2. Apply merger orbits and COM correction (positions only for now)
    if args.n_galaxies > 1:
        print("\n" + "=" * 70)
        print("CONFIGURING MERGER ORBITS (positions only)")
        print("=" * 70)

        # Primary galaxy is at origin with zero velocity
        # Calculate total masses (using args values directly)
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
            orbit_pos, orbit_vel_dummy = parabolic_orbit_initial_conditions( # orbit_vel not used now
                M_primary, M_secondary, r_init, r_peri, orbit_plane
            )

            # Place galaxy
            secondary_galaxy = galaxies_pos_mass[i]
            for comp_name in ["dm", "gas", "stars"]:
                if len(secondary_galaxy[comp_name]["pos"]) > 0:
                    pos = secondary_galaxy[comp_name]["pos"]

                    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
                    # Dummy velocities for rotation, will be re-sampled later
                    vx_dummy = np.zeros_like(x) 
                    vy_dummy = np.zeros_like(y)
                    vz_dummy = np.zeros_like(z)

                    x_new, y_new, z_new, _, _, _ = place_galaxy_in_orbit(
                        x, y, z, vx_dummy, vy_dummy, vz_dummy, orbit_pos, orbit_vel_dummy, inclination, node
                    )

                    secondary_galaxy[comp_name]["pos"] = np.column_stack([x_new, y_new, z_new])

        # Apply COM correction
        print("\nApplying center-of-mass correction to positions...")
        all_masses_for_com = []
        all_positions_for_com = []
        
        for galaxy in galaxies_pos_mass:
            for comp_name in ["dm", "gas", "stars"]:
                if len(galaxy[comp_name]["pos"]) > 0:
                    all_masses_for_com.append(galaxy[comp_name]["mass"])
                    all_positions_for_com.append(tuple(galaxy[comp_name]["pos"].T))

        pos_corrected, _ = center_of_mass_correction( # Dummy velocities removed
            all_masses_for_com, all_positions_for_com, [None]*len(all_positions_for_com)
        )

        # Reassign corrected values
        idx = 0
        for galaxy in galaxies_pos_mass:
            for comp_name in ["dm", "gas", "stars"]:
                if len(galaxy[comp_name]["pos"]) > 0:
                    x, y, z = pos_corrected[idx]
                    galaxy[comp_name]["pos"] = np.column_stack([x, y, z])
                    idx += 1

    # 3. Combine all galaxy particles into a single structure
    # This includes background particles if requested
    all_pos_dict = {"dm": [], "gas": [], "stars": []}
    all_mass_dict = {"dm": [], "gas": [], "stars": []}
    
    for galaxy in galaxies_pos_mass:
        for comp_name in ["dm", "gas", "stars"]:
            if len(galaxy[comp_name]["pos"]) > 0:
                all_pos_dict[comp_name].append(galaxy[comp_name]["pos"])
                all_mass_dict[comp_name].append(galaxy[comp_name]["mass"])

    # Process background particles if requested
    print("\n" + "=" * 70)
    print("GENERATING BACKGROUND PARTICLES")
    print("=" * 70)

    # The add_uniform_background function returns updated dictionaries.
    # It must be called with initial empty lists if no galaxy particles of that type.
    # It also manages mass assignment for background particles.
    
    # Flatten initial galaxy data for background function input format
    initial_combined_data = {
        "dm": {"pos": np.array([]).reshape(0,3), "mass": np.array([])},
        "gas": {"pos": np.array([]).reshape(0,3), "mass": np.array([])},
        "stars": {"pos": np.array([]).reshape(0,3), "mass": np.array([])},
    }
    for comp_name in ["dm", "gas", "stars"]:
        if all_pos_dict[comp_name]:
            initial_combined_data[comp_name]["pos"] = np.vstack(all_pos_dict[comp_name])
            initial_combined_data[comp_name]["mass"] = np.concatenate(all_mass_dict[comp_name])
            
    final_pos_mass_data = add_uniform_background(
        initial_combined_data,
        box_size=args.box_kpc,
        m_part=args.m_part_msun, # For galaxy particles only if no grid, ignored otherwise
        rho_gas=args.bg_gas_density_msun_kpc3,
        rho_dm=args.bg_dm_density_msun_kpc3,
        grid_spacing=args.bg_grid_kpc,
        rng=rng,
    )
    
    all_pos_final = {k: v["pos"] for k,v in final_pos_mass_data.items()}
    all_mass_final = {k: v["mass"] for k,v in final_pos_mass_data.items()}

    # Center all particles in box (after background added)
    print("\nCentering all particles in simulation box...")
    box_center = args.box_kpc / 2.0
    for comp_name in ["dm", "gas", "stars"]:
        if all_pos_final[comp_name].size > 0:
            all_pos_final[comp_name] += box_center

    # 4. Initialize and run Grid Solver (Poisson part)
    print("\n" + "=" * 70)
    print("COMPUTING SELF-CONSISTENT POTENTIAL ON GRID")
    print("=" * 70)
    
    R_grid = np.linspace(0, args.R_grid_kpc, args.nR_grid) # Radial grid
    z_grid = np.linspace(-args.z_grid_kpc, args.z_grid_kpc, args.nz_grid) # Vertical grid
    
    grid_solver = GalaxyGridSolver(
        R_grid, z_grid, args.eps_grid,
        m200=args.m200_msun[0], c200=args.c200[0], # Assuming single primary galaxy
        m_bulge=args.m_star_msun[0] * (1 - args.dt[0]), # M_bulge for primary
        a_bulge=args.bulge_a_kpc[0], # a_bulge for primary
        M_disc_star=args.m_star_msun[0] * args.dt[0], # M_disc_star for primary
        R_d_star=args.rd_kpc[0], z_d_star=args.zd_kpc[0], # Disc params for primary
        M_disc_gas=args.m_gas_msun[0], R_d_gas=args.rg_kpc[0], z_d_gas=args.zg_kpc[0] # Gas disc params for primary
    )
    grid_solver.bin_particles_to_grid(all_pos_final, all_mass_final)
    grid_solver.compute_potential_grid()

    # 5. Generate velocities using grid solver's potential
    print("\n" + "=" * 70)
    print("GENERATING VELOCITIES FROM GRID POTENTIAL")
    print("=" * 70)

    # These will be the components of the final combined_data
    dm_vels = np.array([]).reshape(0,3)
    gas_vels = np.array([]).reshape(0,3)
    stars_vels = np.array([]).reshape(0,3)
    
    # Loop over original galaxies again to process velocities
    offset_dm = 0
    offset_gas = 0
    offset_stars = 0

    for i, galaxy_pos_mass_data in enumerate(galaxies_pos_mass):
        # Extract galaxy parameters for current galaxy
        m200 = args.m200_msun[i]
        c200 = args.c200[i]
        m_star = args.m_star_msun[i]
        m_gas = args.m_gas_msun[i]
        dt_ratio = args.dt[i]
        R_d = args.rd_kpc[i]
        z_d = args.zd_kpc[i]
        R_g = args.rg_kpc[i]
        z_g = args.zg_kpc[i]
        a_bulge = args.bulge_a_kpc[i]

        m_disc_star = m_star * dt_ratio
        m_bulge = m_star * (1 - dt_ratio)
        N_bulge = int(round(m_bulge / args.m_part_msun))

        # Handle DM
        if galaxy_pos_mass_data["dm"]["pos"].size > 0:
            pos_dm = galaxy_pos_mass_data["dm"]["pos"]
            vx_dm, vy_dm, vz_dm = sample_halo_velocities(
                pos_dm[:,0], pos_dm[:,1], pos_dm[:,2], rng, grid_solver
            )
            dm_vels = np.vstack([dm_vels, np.column_stack([vx_dm, vy_dm, vz_dm])])

        # Handle Gas
        if galaxy_pos_mass_data["gas"]["pos"].size > 0:
            pos_gas = galaxy_pos_mass_data["gas"]["pos"]
            # Pass original component mass and scale length for sigma_surf calc
            vx_gas, vy_gas, vz_gas = sample_disc_velocities(
                pos_gas[:,0], pos_gas[:,1], pos_gas[:,2],
                M_disc=m_gas, R_d=R_g, z_d=z_g,
                Q_target=args.q_gas,
                rng=rng, grid_solver=grid_solver,
                spiral_params={ # Pass spiral/bar params as they are per galaxy
                    "n_arms": args.n_arms[i], "pitch_deg": args.pitch_deg[i],
                    "arm_strength": args.arm_strength[i], "stream_frac": args.arm_stream_frac[i]},
                bar_params={
                    "enabled": args.bar[i]>0, "radius": args.bar_r_kpc[i], "q": args.bar_q[i],
                    "stream_frac": args.bar_stream_frac[i]},
                is_gas=True
            )
            gas_vels = np.vstack([gas_vels, np.column_stack([vx_gas, vy_gas, vz_gas])])
        
        # Handle Stars (disc + bulge)
        if galaxy_pos_mass_data["stars"]["pos"].size > 0:
            pos_star_all = galaxy_pos_mass_data["stars"]["pos"]
            N_star_total = len(pos_star_all)

            # Split into disc and bulge (original counts from generate_galaxy_positions)
            N_disc_star_orig = int(round(m_disc_star / args.m_part_msun))
            pos_disc_star = pos_star_all[:N_disc_star_orig]
            pos_bulge_star = pos_star_all[N_disc_star_orig:]
            
            vx_disc_star, vy_disc_star, vz_disc_star = sample_disc_velocities(
                pos_disc_star[:,0], pos_disc_star[:,1], pos_disc_star[:,2],
                M_disc=m_disc_star, R_d=R_d, z_d=z_d,
                Q_target=args.q_star,
                rng=rng, grid_solver=grid_solver,
                spiral_params={ # Pass spiral/bar params as they are per galaxy
                    "n_arms": args.n_arms[i], "pitch_deg": args.pitch_deg[i],
                    "arm_strength": args.arm_strength[i], "stream_frac": args.arm_stream_frac[i]},
                bar_params={
                    "enabled": args.bar[i]>0, "radius": args.bar_r_kpc[i], "q": args.bar_q[i],
                    "stream_frac": args.bar_stream_frac[i]},
                is_gas=False
            )
            
            if N_bulge > 0:
                vx_bulge, vy_bulge, vz_bulge = sample_bulge_velocities(
                    pos_bulge_star[:,0], pos_bulge_star[:,1], pos_bulge_star[:,2],
                    rng=rng, grid_solver=grid_solver
                )
            else:
                vx_bulge = vy_bulge = vz_bulge = np.array([]).reshape(0)

            vx_star = np.concatenate([vx_disc_star, vx_bulge])
            vy_star = np.concatenate([vy_disc_star, vy_bulge])
            vz_star = np.concatenate([vz_disc_star, vz_bulge])
            
            stars_vels = np.vstack([stars_vels, np.column_stack([vx_star, vy_star, vz_star])])

    # Combine all particles (positions, velocities, masses)
    combined_data_final = {
        "dm": {"pos": final_pos_mass_data["dm"]["pos"], "vel": dm_vels, "mass": final_pos_mass_data["dm"]["mass"]},
        "gas": {"pos": final_pos_mass_data["gas"]["pos"], "vel": gas_vels, "mass": final_pos_mass_data["gas"]["mass"]},
        "stars": {"pos": final_pos_mass_data["stars"]["pos"], "vel": stars_vels, "mass": final_pos_mass_data["stars"]["mass"]},
    }

    # 6. Write IC file
    print("\n" + "=" * 70)
    print("WRITING OUTPUT FILES")
    print("=" * 70)
    print(f"\nWriting IC file: {args.out_ics}")

    requested_masses = {
        "dm": sum(args.m200_msun),
        "gas": sum(args.m_gas_msun),
        "stars": sum(args.m_star_msun),
    }

    write_swift_ic(args.out_ics, args.box_kpc, combined_data_final)

    print_ic_summary(combined_data_final, requested_masses, args.box_kpc)

    # Write YAML parameter file
    print(f"Writing YAML parameter file: {args.out_params}")
    params = generate_swift_params(
        ic_filename=args.out_ics,
        box_size=args.box_kpc,
        time_end_gyr=args.time_end_gyr,
        snapshot_dt_myr=args.snapshot_dt_myr,
        dt_min_gyr=args.dt_min_gyr,
        output_basename=args.snapshot_basename,
        run_name=args.run_name,
        param_template=args.param_template,
    )
    write_yaml_file(args.out_params, params)
    print_yaml_summary(args.out_params, args.time_end_gyr, args.snapshot_dt_myr, args.dt_min_gyr)

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print("\nTo run SWIFT:")
    print(f"  swift --hydro --self-gravity --stars --feedback --threads=<N> {args.out_params}")


if __name__ == "__main__":
    main()
