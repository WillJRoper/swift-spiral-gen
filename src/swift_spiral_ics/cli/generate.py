"""SWIFT Initial Conditions Generator.
Command line interface for generating initial conditions for SWIFT simulations.
"""

import argparse
import sys
import numpy as np
import yaml
from unyt import G
from tqdm import tqdm

from ..io.swift_writer import write_swift_ic
from ..io.yaml_writer import generate_swift_params
from ..physics.grid_solver import GalaxyGridSolver # Import new solver
from ..physics.profiles import nfw_params
from ..physics.sampling import (
    sample_bulge_velocities,
    sample_disc_velocities,
    sample_exponential_disc,
    sample_halo_velocities,
    sample_hernquist_bulge,
    sample_nfw_halo,
)
from ..utils.random import get_rng


def generate_galaxy_positions(galaxy_id: int, args: argparse.Namespace, rng: np.random.Generator) -> dict:
    """Generate positions and masses for a single galaxy component."""

    M_star = args.m_star_msun # Arguments are in Msun
    M_gas = args.m_gas_msun

    # Ensure consistent number of particles
    N_total = args.n_halo + args.n_bulge + args.n_star + args.n_gas
    
    # 1. Sample halo positions
    m200_msun = args.m200_msun
    c200 = args.c200
    r_s, _ = nfw_params(m200_msun, c200)
    
    # Calculate truncation radius
    r_max_halo = r_s * 10 # 10 R_s is a common truncation (approx R_200 * 2)
    
    pos_halo = np.zeros((args.n_halo, 3))
    mass_halo = np.full(args.n_halo, m200_msun / args.n_halo)
    
    if args.n_halo > 0:
        x_halo, y_halo, z_halo = sample_nfw_halo(
            args.n_halo, m200_msun, c200, r_max_halo, rng
        )
        pos_halo = np.column_stack([x_halo, y_halo, z_halo])

    # 2. Sample bulge positions
    m_bulge_msun = args.m_bulge_msun
    bulge_a_kpc = args.bulge_a_kpc
    
    pos_bulge = np.zeros((args.n_bulge, 3))
    mass_bulge = np.full(args.n_bulge, m_bulge_msun / args.n_bulge)
    
    if args.n_bulge > 0:
        x_bulge, y_bulge, z_bulge = sample_hernquist_bulge(
            args.n_bulge, m_bulge_msun, bulge_a_kpc, rng
        )
        pos_bulge = np.column_stack([x_bulge, y_bulge, z_bulge])

    # 3. Sample stellar disc positions
    R_d_star_kpc = args.rd_kpc
    z_d_star_kpc = args.zd_kpc
    
    pos_star = np.zeros((args.n_star, 3))
    mass_star = np.full(args.n_star, M_star / args.n_star)

    if args.n_star > 0:
        x_star, y_star, z_star = sample_exponential_disc(
            args.n_star, M_star, R_d_star_kpc, z_d_star_kpc, rng,
            spiral_params={
                "arm_strength": args.arm_strength,
                "n_arms": args.n_arms,
                "pitch_deg": args.pitch_deg
            } if args.arm_strength > 0 else None,
            bar_params={
                "enabled": args.bar_enabled,
                "strength": args.bar_strength,
                "radius": args.bar_radius,
                "q": args.bar_q,
                "angle": args.bar_angle
            } if args.bar_enabled else None,
        )
        pos_star = np.column_stack([x_star, y_star, z_star])

    # 4. Sample gas disc positions
    R_d_gas_kpc = args.rg_kpc
    z_d_gas_kpc = args.zg_kpc
    
    pos_gas = np.zeros((args.n_gas, 3))
    mass_gas = np.full(args.n_gas, M_gas / args.n_gas)

    if args.n_gas > 0:
        x_gas, y_gas, z_gas = sample_exponential_disc(
            args.n_gas, M_gas, R_d_gas_kpc, z_d_gas_kpc, rng,
            spiral_params={
                "arm_strength": args.arm_strength,
                "n_arms": args.n_arms,
                "pitch_deg": args.pitch_deg
            } if args.arm_strength > 0 else None,
            bar_params={
                "enabled": args.bar_enabled,
                "strength": args.bar_strength,
                "radius": args.bar_radius,
                "q": args.bar_q,
                "angle": args.bar_angle
            } if args.bar_enabled else None,
        )
        pos_gas = np.column_stack([x_gas, y_gas, z_gas])

    # Combine positions and masses for a single galaxy
    galaxy_data = {
        "dm": {"pos": pos_halo, "mass": mass_halo},
        "gas": {"pos": pos_gas, "mass": mass_gas},
        "stars": {"pos": pos_star, "mass": mass_star},
        "bulge": {"pos": pos_bulge, "mass": mass_bulge},
    }

    # Jitter identical positions
    galaxy_data["dm"]["pos"] = _jitter_duplicates(galaxy_data["dm"]["pos"], rng, id_str="DM")
    galaxy_data["gas"]["pos"] = _jitter_duplicates(galaxy_data["gas"]["pos"], rng, id_str="Gas")
    galaxy_data["stars"]["pos"] = _jitter_duplicates(galaxy_data["stars"]["pos"], rng, id_str="Stars")
    galaxy_data["bulge"]["pos"] = _jitter_duplicates(galaxy_data["bulge"]["pos"], rng, id_str="Bulge")

    return galaxy_data


def add_uniform_background(
    initial_data: dict, # Renamed from combined_data for clarity
    box_size: float,
    m_part: float,
    rho_gas: float,
    rho_dm: float,
    grid_spacing: float,
    rng: np.random.Generator,
) -> dict:
    """Add uniform background gas and DM to the combined particle set.

    Args:
        initial_data: Dict with existing galaxy particles, pos and mass only.
        box_size: Simulation box size (kpc).
        m_part: Particle mass (Msun) for galaxy particles.
        rho_gas: Background gas density (Msun / kpc^3).
        rho_dm: Background DM density (Msun / kpc^3).
        grid_spacing: Grid spacing for background particles (kpc).
        rng: Random number generator.

    Returns:
        Dict with updated particle data (pos, vel, mass) including background.
    """
    volume = box_size**3
    
    # Initialize updated dictionary with existing galaxy data, and ensure 'vel' key exists
    updated = {
        "dm": {"pos": initial_data["dm"]["pos"], "mass": initial_data["dm"]["mass"], "vel": np.array([]).reshape(0,3)},
        "gas": {"pos": initial_data["gas"]["pos"], "mass": initial_data["gas"]["mass"], "vel": np.array([]).reshape(0,3)},
        "stars": {"pos": initial_data["stars"]["pos"], "mass": initial_data["stars"]["mass"], "vel": np.array([]).reshape(0,3)},
        "bulge": {"pos": initial_data["bulge"]["pos"], "mass": initial_data["bulge"]["mass"], "vel": np.array([]).reshape(0,3)},
    }
    
    use_grid = grid_spacing > 0
    half_box = box_size / 2.0

    if not use_grid:
        # Add random background particles
        if rho_dm > 0:
            n_dm = int(round(rho_dm * volume / m_part)) # Uses m_part for random bg
            if n_dm > 0:
                pos_dm = rng.uniform(-half_box, half_box, (n_dm, 3))
                vel_dm = np.zeros((n_dm, 3), dtype=float)
                mass_dm = np.full(n_dm, m_part)
                
                updated["dm"]["pos"] = np.vstack([updated["dm"]["pos"], pos_dm])
                updated["dm"]["vel"] = np.vstack([updated["dm"]["vel"], vel_dm])
                updated["dm"]["mass"] = np.concatenate([updated["dm"]["mass"], mass_dm])
                print(f"  Added uniform DM background (random): N={n_dm}, rho={rho_dm:.3e} Msun/kpc^3")

        if rho_gas > 0:
            n_gas = int(round(rho_gas * volume / m_part)) # Uses m_part for random bg
            if n_gas > 0:
                pos_gas = rng.uniform(-half_box, half_box, (n_gas, 3))
                vel_gas = np.zeros((n_gas, 3), dtype=float)
                mass_gas = np.full(n_gas, m_part)
                
                updated["gas"]["pos"] = np.vstack([updated["gas"]["pos"], pos_gas])
                updated["gas"]["vel"] = np.vstack([updated["gas"]["vel"], vel_gas])
                updated["gas"]["mass"] = np.concatenate([updated["gas"]["mass"], mass_gas])
                print(f"  Added uniform gas background (random): N={n_gas}, rho={rho_gas:.3e} Msun/kpc^3")

    else:
        # Add grid background particles (centered on origin)
        coords_1d = np.arange(-half_box, half_box, grid_spacing)
        if coords_1d.size > 0:
            gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
            grid_positions = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
            n_grid = len(grid_positions)
            
            # Add small random jitter
            jitter_gas = rng.normal(scale=0.1 * grid_spacing, size=grid_positions.shape)
            jitter_dm = rng.normal(scale=0.1 * grid_spacing, size=grid_positions.shape)
            gas_grid = grid_positions + jitter_gas
            dm_grid = grid_positions + jitter_dm
            
            # Calculate mass per particle to match target density
            if rho_gas > 0:
                m_gas_bg = (rho_gas * volume) / n_grid
                vel_gas = np.zeros((n_grid, 3), dtype=float)
                mass_gas = np.full(n_grid, m_gas_bg)
                
                updated["gas"]["pos"] = np.vstack([updated["gas"]["pos"], gas_grid])
                updated["gas"]["vel"] = np.vstack([updated["gas"]["vel"], vel_gas])
                updated["gas"]["mass"] = np.concatenate([updated["gas"]["mass"], mass_gas])
                print(f"  Added grid gas background: spacing={grid_spacing} kpc, N={n_grid}, m={m_gas_bg:.2e} Msun")

            if rho_dm > 0:
                m_dm_bg = (rho_dm * volume) / n_grid
                vel_dm = np.zeros((n_grid, 3), dtype=float)
                mass_dm = np.full(n_grid, m_dm_bg)
                
                updated["dm"]["pos"] = np.vstack([updated["dm"]["pos"], dm_grid])
                updated["dm"]["vel"] = np.vstack([updated["dm"]["vel"], vel_dm])
                updated["dm"]["mass"] = np.concatenate([updated["dm"]["mass"], mass_dm])
                print(f"  Added grid DM background: spacing={grid_spacing} kpc, N={n_grid}, m={m_dm_bg:.2e} Msun")

    return updated


def _jitter_duplicates(pos: np.ndarray, rng: np.random.Generator, id_str: str) -> np.ndarray:
    """Jitter particles that have identical positions."""
    if pos.size == 0:
        return pos

    unique_rows, counts = np.unique(pos, axis=0, return_counts=True)
    duplicates_exist = np.any(counts > 1)

    if duplicates_exist:
        # For simplicity, jitter all particles if duplicates exist
        # A more robust solution would only jitter the duplicate ones
        jitter_scale = 1e-4 # kpc, small jitter
        pos += rng.normal(loc=0.0, scale=jitter_scale, size=pos.shape)
        # print(f"  Jittered {id_str} particles due to duplicates.") # Too verbose
    return pos


def main():
    parser = argparse.ArgumentParser(
        description="SWIFT Initial Conditions Generator."
    )

    # Output arguments
    parser.add_argument(
        "--out-ics", type=str, default="galaxy_ic.hdf5", help="Output ICs file name."
    )
    parser.add_argument(
        "--out-params",
        type=str,
        default="galaxy_params.yml",
        help="Output parameter file name.",
    )

    # General galaxy properties
    parser.add_argument(
        "--box-kpc", type=float, default=100.0, help="Simulation box size in kpc."
    )
    parser.add_argument(
        "--m-part-msun", type=float, default=1e7, help="Nominal particle mass in M_sun (10^10 M_sun units)."
    )
    parser.add_argument(
        "--n-galaxies", type=int, default=1, help="Number of galaxies to generate."
    )
    
    # Halo properties
    parser.add_argument(
        "--n-halo", type=int, default=100000, help="Number of halo particles."
    )
    parser.add_argument(
        "--m200-msun", type=float, default=1e12, help="M200 of the halo in M_sun (10^10 M_sun units)."
    )
    parser.add_argument(
        "--c200", type=float, default=10.0, help="NFW concentration parameter."
    )

    # Bulge properties
    parser.add_argument(
        "--n-bulge", type=int, default=1000, help="Number of bulge particles."
    )
    parser.add_argument(
        "--m-bulge-msun", type=float, default=1e10, help="Bulge mass in M_sun (10^10 M_sun units)."
    )
    parser.add_argument(
        "--bulge-a-kpc", type=float, default=0.8, help="Hernquist bulge scale length in kpc."
    )

    # Stellar disc properties
    parser.add_argument(
        "--n-star", type=int, default=4000, help="Number of stellar particles."
    )
    parser.add_argument(
        "--m-star-msun", type=float, default=5e10, help="Stellar disc mass in M_sun (10^10 M_sun units)."
    )
    parser.add_argument(
        "--rd-kpc", type=float, default=3.5, help="Stellar disc scale length in kpc."
    )
    parser.add_argument(
        "--zd-kpc", type=float, default=0.35, help="Stellar disc scale height in kpc."
    )
    parser.add_argument(
        "--Q-star", type=float, default=1.5, help="Toomre Q parameter for stellar disc."
    )

    # Gas disc properties
    parser.add_argument(
        "--n-gas", type=int, default=1000, help="Number of gas particles."
    )
    parser.add_argument(
        "--m-gas-msun", type=float, default=1e10, help="Gas disc mass in M_sun (10^10 M_sun units)."
    )
    parser.add_argument(
        "--rg-kpc", type=float, default=7.0, help="Gas disc scale length in kpc."
    )
    parser.add_argument(
        "--zg-kpc", type=float, default=0.1, help="Gas disc scale height in kpc."
    )
    parser.add_argument(
        "--Q-gas", type=float, default=1.0, help="Toomre Q parameter for gas disc."
    )

    # Spiral arm properties
    parser.add_argument(
        "--n-arms", type=int, default=2, help="Number of spiral arms."
    )
    parser.add_argument(
        "--pitch-deg", type=float, default=15.0, help="Pitch angle of spiral arms in degrees."
    )
    parser.add_argument(
        "--arm-strength", type=float, default=0.3, help="Strength of spiral arms (0-1)."
    )
    parser.add_argument(
        "--arm-stream-frac", type=float, default=0.1, help="Streaming velocity fraction for spiral arms."
    )

    # Bar properties
    parser.add_argument(
        "--bar-enabled", action="store_true", help="Enable a galactic bar."
    )
    parser.add_argument(
        "--bar-strength", type=float, default=0.1, help="Strength of the bar."
    )
    parser.add_argument(
        "--bar-radius", type=float, default=3.0, help="Radius of the bar in kpc."
    )
    parser.add_argument(
        "--bar-q", type=float, default=0.3, help="Flattening parameter q for the bar."
    )
    parser.add_argument(
        "--bar-angle", type=float, default=0.0, help="Angle of the bar in degrees."
    )

    # Simulation properties
    parser.add_argument(
        "--dt", type=float, default=0.8, help="Base time-step in Gyr."
    )
    parser.add_argument(
        "--dt-min-gyr", type=float, default=1e-5, help="Minimum physical time-step in Gyr."
    )
    parser.add_argument(
        "--time-end-gyr", type=float, default=10.0, help="Total simulation time in Gyr."
    )
    parser.add_argument(
        "--snapshot-dt-myr", type=float, default=10.0, help="Snapshot output interval in Myr."
    )

    # Grid solver properties
    parser.add_argument(
        "--nR-grid", type=int, default=64, help="Number of radial grid cells for potential solver."
    )
    parser.add_argument(
        "--nz-grid", type=int, default=64, help="Number of vertical grid cells for potential solver."
    )
    parser.add_argument(
        "--eps-grid", type=float, default=0.1, help="Softening length for grid potential solver in kpc."
    )

    # Background properties
    parser.add_argument(
        "--bg-gas-density-msun-kpc3", type=float, default=0.0, help="Uniform background gas density (Msun/kpc^3)."
    )
    parser.add_argument(
        "--bg-dm-density-msun-kpc3", type=float, default=0.0, help="Uniform background DM density (Msun/kpc^3)."
    )
    parser.add_argument(
        "--bg-grid-kpc", type=float, default=0.0, help="Grid spacing for background particles (0 for random)."
    )

    # Misc
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Name for the SWIFT run (used in param file)."
    )
    parser.add_argument(
        "--param-template", type=str, default="eagle_ref_cosmo", help="Name of the parameter file template."
    )
    parser.add_argument(
        "--snapshot-basename", type=str, default="snapshot", help="Basename for SWIFT snapshots."
    )


    args = parser.parse_args()

    # --- Initialize RNG ---
    rng = get_rng(args.seed)

    # --- Generate galaxy positions and masses ---
    print("======================================================================")
    print("SWIFT SPIRAL ICs - Initial Conditions Generator")
    print("======================================================================")

    all_galaxies_pos_mass = []
    for i in range(args.n_galaxies):
        print(f"Generating positions for galaxy {i}...")
        galaxy_data = generate_galaxy_positions(i, args, rng)
        all_galaxies_pos_mass.append(galaxy_data)
        
    # --- Combine all galaxies and add background ---
    initial_combined_data = {
        "dm": {"pos": np.vstack([g["dm"]["pos"] for g in all_galaxies_pos_mass]),
               "mass": np.concatenate([g["dm"]["mass"] for g in all_galaxies_pos_mass])},
        "gas": {"pos": np.vstack([g["gas"]["pos"] for g in all_galaxies_pos_mass]),
                "mass": np.concatenate([g["gas"]["mass"] for g in all_galaxies_pos_mass])},
        "stars": {"pos": np.vstack([g["stars"]["pos"] for g in all_galaxies_pos_mass]),
                  "mass": np.concatenate([g["stars"]["mass"] for g in all_galaxies_pos_mass])},
        "bulge": {"pos": np.vstack([g["bulge"]["pos"] for g in all_galaxies_pos_mass]),
                  "mass": np.concatenate([g["bulge"]["mass"] for g in all_galaxies_pos_mass])},
    }

    # Add uniform background particles if specified
    if args.bg_gas_density_msun_kpc3 > 0 or args.bg_dm_density_msun_kpc3 > 0:
        initial_combined_data = add_uniform_background(
            initial_combined_data,
            args.box_kpc,
            args.m_part_msun, # Already in Msun
            args.bg_gas_density_msun_kpc3,
            args.bg_dm_density_msun_kpc3,
            args.bg_grid_kpc,
            rng,
        )

    # Shift all particles to box center AFTER velocity assignment (Wait, logic in turn 24 step 3 moved it to END)
    # But I see in previous `write_file` that I moved it to END.
    # So here I just need to update GridSolver init.

    # --- Setup Grid Solver ---
    # Create grid for potential solver
    R_grid_solver = np.linspace(0, args.box_kpc / 2.0, args.nR_grid) # R goes from 0 to Box/2
    z_grid_solver = np.linspace(-args.box_kpc / 2.0, args.box_kpc / 2.0, args.nz_grid) # z goes from -Box/2 to Box/2
    
    # Initialize GalaxyGridSolver with galaxy parameters
    grid_solver = GalaxyGridSolver(
        R_grid_solver, z_grid_solver, args.eps_grid,
        m200=args.m200_msun, c200=args.c200,
        m_bulge=args.m_bulge_msun, a_bulge=args.bulge_a_kpc,
        M_disc_star=args.m_star_msun, R_d_star=args.rd_kpc, z_d_star=args.zd_kpc,
        M_disc_gas=args.m_gas_msun, R_d_gas=args.rg_kpc, z_d_gas=args.zg_kpc
    )

    # Bin particles to grid and compute potential
    print("Computing self-consistent potential on grid...")
    all_pos_for_grid = {ptype: initial_combined_data[ptype]["pos"] for ptype in initial_combined_data}
    all_mass_for_grid = {ptype: initial_combined_data[ptype]["mass"] for ptype in initial_combined_data}
    grid_solver.bin_particles_to_grid(all_pos_for_grid, all_mass_for_grid)
    grid_solver.compute_potential_grid()
    print("Potential computation complete.")

    # --- Assign velocities ---
    print("Assigning velocities...")
    for ptype in tqdm(initial_combined_data, desc="Assigning velocities"):
        pos = initial_combined_data[ptype]["pos"]
        mass = initial_combined_data[ptype]["mass"]
        
        if pos.size == 0:
            initial_combined_data[ptype]["vel"] = np.empty((0,3), dtype=float)
            continue

        if ptype == "dm":
            vx, vy, vz = sample_halo_velocities(
                pos[:,0], pos[:,1], pos[:,2], mass, rng, grid_solver
            )
        elif ptype == "bulge":
            vx, vy, vz = sample_bulge_velocities(
                pos[:,0], pos[:,1], pos[:,2], mass, rng, grid_solver
            )
        elif ptype == "stars":
            vx, vy, vz = sample_disc_velocities(
                pos[:,0], pos[:,1], pos[:,2], mass, args.m_star_msun, args.rd_kpc, args.zd_kpc, args.Q_star,
                rng, grid_solver, spiral_params=None, bar_params=None, is_gas=False
            )
        elif ptype == "gas":
            vx, vy, vz = sample_disc_velocities(
                pos[:,0], pos[:,1], pos[:,2], mass, args.m_gas_msun, args.rg_kpc, args.zg_kpc, args.Q_gas,
                rng, grid_solver, spiral_params=None, bar_params=None, is_gas=True
            )
        else:
            vx, vy, vz = np.zeros_like(pos), np.zeros_like(pos), np.zeros_like(pos) # Default to zero velocity

        initial_combined_data[ptype]["vel"] = np.column_stack([vx, vy, vz])
    print("Velocities assigned.")

    # --- Shift all particles to box center AFTER velocity assignment ---
    # This ensures grid solver and velocities were calculated on centered data.
    box_center = args.box_kpc / 2.0
    for ptype in initial_combined_data:
        if initial_combined_data[ptype]["pos"].size > 0:
            initial_combined_data[ptype]["pos"] += box_center
            initial_combined_data[ptype]["pos"] = np.mod(initial_combined_data[ptype]["pos"], args.box_kpc) # Wrap around

    # --- Write ICs and parameter file ---
    print(f"Writing ICs to {args.out_ics}...")
    write_swift_ic(
        args.out_ics,
        args.box_kpc,
        initial_combined_data,
    )
    print(f"ICs written to {args.out_ics}.")

    print(f"Generating parameter file {args.out_params}...")
    params = generate_swift_params(
        ic_filename=args.out_ics,
        box_size=args.box_kpc,
        time_end_gyr=args.time_end_gyr,
        snapshot_dt_myr=args.snapshot_dt_myr,
        dt_min_gyr=args.dt_min_gyr,
        softening_kpc=args.eps_grid,
        output_basename=args.snapshot_basename,
        run_name=args.run_name,
        param_template=args.param_template,
    )
    with open(args.out_params, "w") as f:
        f.write(params)
    print(f"Parameter file written to {args.out_params}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
