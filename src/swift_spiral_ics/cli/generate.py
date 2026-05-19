"""SWIFT Initial Conditions Generator.
Command line interface for generating initial conditions for SWIFT simulations.
"""

import argparse
import sys

import numpy as np

from ..io.swift_writer import write_swift_ic
from ..io.yaml_writer import generate_swift_params
from ..physics.grid_solver import GalaxyGridSolver
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

G_KPC_KMS2_MSUN = 4.302e-6


def generate_galaxy_particles(
    galaxy_id: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    mass_scale: float = 1.0,
    size_scale: float = 1.0,
) -> dict:
    """Generate positions, velocities, and masses for a single isolated spiral."""

    m200_msun = args.m200_msun * mass_scale
    M_star = args.m_star_msun * mass_scale
    M_gas = args.m_gas_msun * mass_scale
    m_bulge_msun = args.m_bulge_msun * mass_scale
    rd_star_kpc = args.rd_kpc * size_scale
    zd_star_kpc = args.zd_kpc * size_scale
    rd_gas_kpc = args.rg_kpc * size_scale
    zd_gas_kpc = args.zg_kpc * size_scale
    bulge_a_kpc = args.bulge_a_kpc * size_scale

    # 1. Sample halo positions
    c200 = args.c200
    r_s, _ = nfw_params(m200_msun, c200)

    # Calculate truncation radius
    r_max_halo = r_s * 10

    pos_halo = np.zeros((args.n_halo, 3))
    mass_halo = _component_masses(m200_msun, args.n_halo)

    if args.n_halo > 0:
        x_halo, y_halo, z_halo = sample_nfw_halo(
            args.n_halo, m200_msun, c200, r_max_halo, rng
        )
        pos_halo = np.column_stack([x_halo, y_halo, z_halo])

    # 2. Sample bulge positions
    pos_bulge = np.zeros((args.n_bulge, 3))
    mass_bulge = _component_masses(m_bulge_msun, args.n_bulge)

    if args.n_bulge > 0:
        r_max_bulge = args.bulge_rmax_scale * bulge_a_kpc
        x_bulge, y_bulge, z_bulge = sample_hernquist_bulge(
            args.n_bulge, m_bulge_msun, bulge_a_kpc, rng, r_max=r_max_bulge
        )
        pos_bulge = np.column_stack([x_bulge, y_bulge, z_bulge])

    # 3. Sample stellar disc positions
    pos_star = np.zeros((args.n_star, 3))
    mass_star = _component_masses(M_star, args.n_star)

    if args.n_star > 0:
        x_star, y_star, z_star = sample_exponential_disc(
            args.n_star, M_star, rd_star_kpc, zd_star_kpc, rng,
            spiral_params={
                "arm_strength": args.arm_strength,
                "n_arms": args.n_arms,
                "pitch_deg": args.pitch_deg,
            } if args.arm_strength > 0 else None,
            bar_params={
                "enabled": args.bar_enabled,
                "strength": args.bar_strength,
                "radius": args.bar_radius,
                "q": args.bar_q,
                "angle": args.bar_angle,
            } if args.bar_enabled else None,
        )
        pos_star = np.column_stack([x_star, y_star, z_star])

    # 4. Sample gas disc positions
    pos_gas = np.zeros((args.n_gas, 3))
    mass_gas = _component_masses(M_gas, args.n_gas)

    if args.n_gas > 0:
        x_gas, y_gas, z_gas = sample_exponential_disc(
            args.n_gas, M_gas, rd_gas_kpc, zd_gas_kpc, rng,
            spiral_params={
                "arm_strength": args.arm_strength,
                "n_arms": args.n_arms,
                "pitch_deg": args.pitch_deg,
            } if args.arm_strength > 0 else None,
            bar_params={
                "enabled": args.bar_enabled,
                "strength": args.bar_strength,
                "radius": args.bar_radius,
                "q": args.bar_q,
                "angle": args.bar_angle,
            } if args.bar_enabled else None,
        )
        pos_gas = np.column_stack([x_gas, y_gas, z_gas])

    print(f"Computing isolated C++ grid potential for galaxy {galaxy_id}...")
    R_grid_solver = np.linspace(0.0, args.box_kpc / 2.0, args.nR_grid)
    z_grid_solver = np.linspace(-args.box_kpc / 2.0, args.box_kpc / 2.0, args.nz_grid)
    grid_solver = GalaxyGridSolver(
        R_grid_solver,
        z_grid_solver,
        args.eps_grid,
        m200=m200_msun,
        c200=c200,
        m_bulge=m_bulge_msun,
        a_bulge=bulge_a_kpc,
        M_disc_star=M_star,
        R_d_star=rd_star_kpc,
        z_d_star=zd_star_kpc,
        M_disc_gas=M_gas,
        R_d_gas=rd_gas_kpc,
        z_d_gas=zd_gas_kpc,
    )
    grid_solver.bin_particles_to_grid(
        {
            "dm": pos_halo,
            "gas": pos_gas,
            "stars": pos_star,
            "bulge": pos_bulge,
        },
        {
            "dm": mass_halo,
            "gas": mass_gas,
            "stars": mass_star,
            "bulge": mass_bulge,
        },
    )
    grid_solver.compute_potential_grid()

    vel_halo = _sample_cylindrical_halo_velocities(pos_halo, mass_halo, rng, grid_solver)
    vel_bulge = _sample_cylindrical_bulge_velocities(pos_bulge, mass_bulge, rng, grid_solver)
    vel_star = _sample_cylindrical_disc_velocities(
        pos_star,
        mass_star,
        M_star,
        rd_star_kpc,
        zd_star_kpc,
        args.Q_star,
        rng,
        grid_solver,
        is_gas=False,
        spiral_params={
            "arm_strength": args.arm_strength,
            "stream_frac": args.arm_stream_frac,
            "n_arms": args.n_arms,
            "pitch_deg": args.pitch_deg,
        } if args.arm_strength > 0 else None,
        bar_params={
            "enabled": args.bar_enabled,
            "stream_frac": args.bar_strength,
            "radius": args.bar_radius,
            "angle": args.bar_angle,
        } if args.bar_enabled else None,
    )
    vel_gas = _sample_cylindrical_disc_velocities(
        pos_gas,
        mass_gas,
        M_gas,
        rd_gas_kpc,
        zd_gas_kpc,
        args.Q_gas,
        rng,
        grid_solver,
        is_gas=True,
        spiral_params={
            "arm_strength": args.arm_strength,
            "stream_frac": args.arm_stream_frac,
            "n_arms": args.n_arms,
            "pitch_deg": args.pitch_deg,
        } if args.arm_strength > 0 else None,
        bar_params={
            "enabled": args.bar_enabled,
            "stream_frac": args.bar_strength,
            "radius": args.bar_radius,
            "angle": args.bar_angle,
        } if args.bar_enabled else None,
    )

    # Combine positions and masses for a single galaxy
    galaxy_data = {
        "dm": {"pos": pos_halo, "vel": vel_halo, "mass": mass_halo},
        "gas": {"pos": pos_gas, "vel": vel_gas, "mass": mass_gas},
        "stars": {"pos": pos_star, "vel": vel_star, "mass": mass_star},
        "bulge": {"pos": pos_bulge, "vel": vel_bulge, "mass": mass_bulge},
    }

    # Jitter identical positions
    galaxy_data["dm"]["pos"] = _jitter_duplicates(galaxy_data["dm"]["pos"], rng, id_str="DM")
    galaxy_data["gas"]["pos"] = _jitter_duplicates(galaxy_data["gas"]["pos"], rng, id_str="Gas")
    galaxy_data["stars"]["pos"] = _jitter_duplicates(galaxy_data["stars"]["pos"], rng, id_str="Stars")
    galaxy_data["bulge"]["pos"] = _jitter_duplicates(galaxy_data["bulge"]["pos"], rng, id_str="Bulge")

    return galaxy_data


def _component_masses(total_mass: float, n_particles: int) -> np.ndarray:
    if n_particles <= 0:
        return np.empty(0, dtype=float)
    return np.full(n_particles, total_mass / n_particles, dtype=float)


def _sample_cylindrical_halo_velocities(
    pos: np.ndarray,
    mass: np.ndarray,
    rng: np.random.Generator,
    grid_solver: GalaxyGridSolver,
) -> np.ndarray:
    if len(pos) == 0:
        return np.empty((0, 3), dtype=float)
    vx, vy, vz = sample_halo_velocities(pos[:, 0], pos[:, 1], pos[:, 2], mass, rng, grid_solver)
    return np.column_stack([vx, vy, vz])


def _sample_cylindrical_bulge_velocities(
    pos: np.ndarray,
    mass: np.ndarray,
    rng: np.random.Generator,
    grid_solver: GalaxyGridSolver,
) -> np.ndarray:
    if len(pos) == 0:
        return np.empty((0, 3), dtype=float)
    vx, vy, vz = sample_bulge_velocities(pos[:, 0], pos[:, 1], pos[:, 2], mass, rng, grid_solver)
    return np.column_stack([vx, vy, vz])


def _sample_cylindrical_disc_velocities(
    pos: np.ndarray,
    mass: np.ndarray,
    total_mass: float,
    scale_radius: float,
    scale_height: float,
    q_target: float,
    rng: np.random.Generator,
    grid_solver: GalaxyGridSolver,
    is_gas: bool,
    spiral_params: dict | None,
    bar_params: dict | None,
) -> np.ndarray:
    if len(pos) == 0:
        return np.empty((0, 3), dtype=float)
    vx, vy, vz = sample_disc_velocities(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        mass,
        total_mass,
        scale_radius,
        scale_height,
        q_target,
        rng,
        grid_solver,
        spiral_params=spiral_params,
        bar_params=bar_params,
        is_gas=is_gas,
    )
    return np.column_stack([vx, vy, vz])


def _rotate_x(values: np.ndarray, angle_deg: float) -> np.ndarray:
    if len(values) == 0 or angle_deg == 0.0:
        return values
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    out = values.copy()
    y = values[:, 1]
    z = values[:, 2]
    out[:, 1] = c * y - s * z
    out[:, 2] = s * y + c * z
    return out


def _place_galaxy(galaxy_data: dict, offset: np.ndarray, bulk_velocity: np.ndarray, inclination: float):
    for component in galaxy_data.values():
        component["pos"] = _rotate_x(component["pos"], inclination) + offset
        component["vel"] = _rotate_x(component["vel"], inclination) + bulk_velocity


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
        "dm": {"pos": initial_data["dm"]["pos"], "mass": initial_data["dm"]["mass"], "vel": initial_data["dm"]["vel"]},
        "gas": {"pos": initial_data["gas"]["pos"], "mass": initial_data["gas"]["mass"], "vel": initial_data["gas"]["vel"]},
        "stars": {"pos": initial_data["stars"]["pos"], "mass": initial_data["stars"]["mass"], "vel": initial_data["stars"]["vel"]},
        "bulge": {"pos": initial_data["bulge"]["pos"], "mass": initial_data["bulge"]["mass"], "vel": initial_data["bulge"]["vel"]},
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
    parser.add_argument(
        "--secondary-mass-ratio",
        type=float,
        default=0.7,
        help="Mass ratio M2/M1 used for the second galaxy in a merger.",
    )
    parser.add_argument(
        "--secondary-size-ratio",
        type=float,
        default=None,
        help="Size ratio R2/R1. Defaults to sqrt(secondary mass ratio).",
    )
    parser.add_argument(
        "--separation-kpc",
        type=float,
        default=None,
        help="Initial centre-to-centre separation for a two-galaxy merger.",
    )
    parser.add_argument(
        "--impact-kpc",
        type=float,
        default=20.0,
        help="Transverse impact parameter for a two-galaxy merger.",
    )
    parser.add_argument(
        "--relative-velocity-kms",
        type=float,
        default=150.0,
        help="Initial approach speed between the two galaxy centres.",
    )
    parser.add_argument(
        "--galaxy1-inclination-deg",
        type=float,
        default=0.0,
        help="Inclination of the first disc around the x-axis.",
    )
    parser.add_argument(
        "--galaxy2-inclination-deg",
        type=float,
        default=0.0,
        help="Inclination of the second disc around the x-axis.",
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
    parser.add_argument(
        "--bulge-rmax-scale",
        type=float,
        default=50.0,
        help="Truncate Hernquist bulge sampling at this many scale lengths.",
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
        "--nR-grid", type=int, default=256, help="Number of radial grid cells for potential solver."
    )
    parser.add_argument(
        "--nz-grid", type=int, default=256, help="Number of vertical grid cells for potential solver."
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
    if args.n_galaxies < 1:
        raise ValueError("--n-galaxies must be at least 1")
    if args.n_galaxies > 2:
        raise ValueError("This CLI currently supports one isolated galaxy or one two-galaxy merger")

    separation = args.separation_kpc or args.box_kpc / 3.0
    secondary_size_ratio = args.secondary_size_ratio or np.sqrt(args.secondary_mass_ratio)

    for i in range(args.n_galaxies):
        print(f"Generating positions for galaxy {i}...")
        mass_scale = 1.0 if i == 0 else args.secondary_mass_ratio
        size_scale = 1.0 if i == 0 else secondary_size_ratio
        galaxy_data = generate_galaxy_particles(i, args, rng, mass_scale, size_scale)

        if args.n_galaxies == 2:
            if i == 0:
                offset = np.array([-0.5 * separation, -0.5 * args.impact_kpc, 0.0])
                bulk_velocity = np.array([0.5 * args.relative_velocity_kms, 0.0, 0.0])
                inclination = args.galaxy1_inclination_deg
            else:
                offset = np.array([0.5 * separation, 0.5 * args.impact_kpc, 0.0])
                bulk_velocity = np.array([-0.5 * args.relative_velocity_kms, 0.0, 0.0])
                inclination = args.galaxy2_inclination_deg
            _place_galaxy(galaxy_data, offset, bulk_velocity, inclination)

        all_galaxies_pos_mass.append(galaxy_data)

    # --- Combine all galaxies and add background ---
    initial_combined_data = {
        "dm": {"pos": np.vstack([g["dm"]["pos"] for g in all_galaxies_pos_mass]),
               "vel": np.vstack([g["dm"]["vel"] for g in all_galaxies_pos_mass]),
               "mass": np.concatenate([g["dm"]["mass"] for g in all_galaxies_pos_mass])},
        "gas": {"pos": np.vstack([g["gas"]["pos"] for g in all_galaxies_pos_mass]),
                "vel": np.vstack([g["gas"]["vel"] for g in all_galaxies_pos_mass]),
                "mass": np.concatenate([g["gas"]["mass"] for g in all_galaxies_pos_mass])},
        "stars": {"pos": np.vstack([g["stars"]["pos"] for g in all_galaxies_pos_mass]),
                  "vel": np.vstack([g["stars"]["vel"] for g in all_galaxies_pos_mass]),
                  "mass": np.concatenate([g["stars"]["mass"] for g in all_galaxies_pos_mass])},
        "bulge": {"pos": np.vstack([g["bulge"]["pos"] for g in all_galaxies_pos_mass]),
                  "vel": np.vstack([g["bulge"]["vel"] for g in all_galaxies_pos_mass]),
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

    if initial_combined_data["bulge"]["pos"].size > 0:
        initial_combined_data["stars"]["pos"] = np.vstack(
            [initial_combined_data["stars"]["pos"], initial_combined_data["bulge"]["pos"]]
        )
        initial_combined_data["stars"]["vel"] = np.vstack(
            [initial_combined_data["stars"]["vel"], initial_combined_data["bulge"]["vel"]]
        )
        initial_combined_data["stars"]["mass"] = np.concatenate(
            [initial_combined_data["stars"]["mass"], initial_combined_data["bulge"]["mass"]]
        )
    del initial_combined_data["bulge"]

    # --- Shift all particles to box center AFTER velocity assignment ---
    # This ensures grid solver and velocities were calculated on centered data.
    box_center = args.box_kpc / 2.0
    for ptype in initial_combined_data:
        if initial_combined_data[ptype]["pos"].size > 0:
            initial_combined_data[ptype]["pos"] += box_center
            initial_combined_data[ptype]["pos"] = np.mod(initial_combined_data[ptype]["pos"], args.box_kpc)

    # --- Write ICs and parameter file ---
    print(f"Writing ICs to {args.out_ics}...")
    write_swift_ic(
        args.out_ics,
        args.box_kpc,
        initial_combined_data,
    )
    print(f"ICs written to {args.out_ics}.")

    # Determine minimum gas particle mass for splitting threshold
    min_gas_mass = None
    if initial_combined_data["gas"]["mass"].size > 0:
        min_gas_mass = np.min(initial_combined_data["gas"]["mass"])

    print(f"Generating parameter file {args.out_params}...")
    params = generate_swift_params(
        ic_filename=args.out_ics,
        box_size=args.box_kpc,
        time_end_gyr=args.time_end_gyr,
        snapshot_dt_myr=args.snapshot_dt_myr,
        dt_min_gyr=args.dt_min_gyr,
        dt_max_gyr=args.dt,
        softening_kpc=args.eps_grid,
        output_basename=args.snapshot_basename,
        run_name=args.run_name,
        param_template=args.param_template,
        min_gas_mass_msun=min_gas_mass,
    )
    with open(args.out_params, "w") as f:
        f.write(params)
    print(f"Parameter file written to {args.out_params}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
