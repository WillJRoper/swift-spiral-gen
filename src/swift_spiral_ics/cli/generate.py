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
_DEFAULT_COMPONENT_MASSES_MSUN = {
    "dm": 1.0e12,
    "stars": 6.0e10,
    "gas": 1.0e10,
}
_DEFAULT_COMPONENT_PARTICLE_MASSES_MSUN = {
    "dm": 1.0e7,
    "stars": 1.25e7,
    "gas": 1.0e7,
}
_DEFAULT_BULGE_FRACTION = 1.0 / 6.0


def _allocate_total_particles(total_mass: float, particle_mass: float, name: str) -> int:
    if particle_mass <= 0:
        raise ValueError(f"{name} must be positive")
    return max(1, int(round(total_mass / particle_mass)))


def _get_galaxy_value(values: list[float] | list[int], galaxy_id: int) -> float | int:
    return values[galaxy_id]


def _resolve_per_galaxy_values(
    values: list[float] | list[int],
    n_galaxies: int,
    name: str,
) -> list[float] | list[int]:
    if len(values) == n_galaxies:
        return values
    if len(values) == 1:
        return [values[0]] * n_galaxies
    raise ValueError(f"{name} expects 1 or {n_galaxies} values, got {len(values)}")


def _normalise_per_galaxy_args(args: argparse.Namespace) -> None:
    if args.n_galaxies < 1:
        raise ValueError("--n-galaxies must be at least 1")

    if args.dm_mass_msun is None:
        args.dm_mass_msun = [_DEFAULT_COMPONENT_MASSES_MSUN["dm"]]
    if args.star_mass_msun is None:
        args.star_mass_msun = [_DEFAULT_COMPONENT_MASSES_MSUN["stars"]]
    if args.gas_mass_msun is None:
        args.gas_mass_msun = [_DEFAULT_COMPONENT_MASSES_MSUN["gas"]]
    if args.bulge_fraction is None:
        args.bulge_fraction = [_DEFAULT_BULGE_FRACTION]

    if args.dm_part_mass_msun is None:
        args.dm_part_mass_msun = _DEFAULT_COMPONENT_PARTICLE_MASSES_MSUN["dm"]
    if args.star_part_mass_msun is None:
        args.star_part_mass_msun = _DEFAULT_COMPONENT_PARTICLE_MASSES_MSUN["stars"]
    if args.gas_part_mass_msun is None:
        args.gas_part_mass_msun = _DEFAULT_COMPONENT_PARTICLE_MASSES_MSUN["gas"]

    args.m200_msun = _resolve_per_galaxy_values(args.dm_mass_msun, args.n_galaxies, "--dm-mass-msun")
    total_stellar_masses = _resolve_per_galaxy_values(
        args.star_mass_msun, args.n_galaxies, "--star-mass-msun"
    )
    bulge_fractions = _resolve_per_galaxy_values(
        args.bulge_fraction, args.n_galaxies, "--bulge-fraction"
    )
    for bulge_fraction in bulge_fractions:
        if not 0.0 <= bulge_fraction < 1.0:
            raise ValueError("--bulge-fraction must satisfy 0 <= B/T < 1")

    args.m_bulge_msun = [
        total_stellar_masses[i] * bulge_fractions[i] for i in range(args.n_galaxies)
    ]
    args.m_star_msun = [
        total_stellar_masses[i] * (1.0 - bulge_fractions[i]) for i in range(args.n_galaxies)
    ]
    args.m_gas_msun = _resolve_per_galaxy_values(args.gas_mass_msun, args.n_galaxies, "--gas-mass-msun")

    args.n_halo = [
        _allocate_total_particles(args.m200_msun[i], args.dm_part_mass_msun, "--dm-part-mass-msun")
        for i in range(args.n_galaxies)
    ]
    args.n_bulge = [
        _allocate_total_particles(args.m_bulge_msun[i], args.star_part_mass_msun, "--star-part-mass-msun")
        for i in range(args.n_galaxies)
    ]
    args.n_star = [
        _allocate_total_particles(args.m_star_msun[i], args.star_part_mass_msun, "--star-part-mass-msun")
        for i in range(args.n_galaxies)
    ]
    args.n_gas = [
        _allocate_total_particles(args.m_gas_msun[i], args.gas_part_mass_msun, "--gas-part-mass-msun")
        for i in range(args.n_galaxies)
    ]

    args.c200 = _resolve_per_galaxy_values(args.c200, args.n_galaxies, "--c200")
    args.bulge_a_kpc = _resolve_per_galaxy_values(args.bulge_a_kpc, args.n_galaxies, "--bulge-a-kpc")
    args.bulge_rmax_scale = _resolve_per_galaxy_values(
        args.bulge_rmax_scale, args.n_galaxies, "--bulge-rmax-scale"
    )
    args.stellar_disk_scale_length_kpc = _resolve_per_galaxy_values(
        args.stellar_disk_scale_length_kpc, args.n_galaxies, "--stellar-disk-scale-length-kpc"
    )
    args.stellar_disk_scale_height_kpc = _resolve_per_galaxy_values(
        args.stellar_disk_scale_height_kpc, args.n_galaxies, "--stellar-disk-scale-height-kpc"
    )
    args.Q_star = _resolve_per_galaxy_values(args.Q_star, args.n_galaxies, "--Q-star")
    args.gas_disk_scale_length_kpc = _resolve_per_galaxy_values(
        args.gas_disk_scale_length_kpc, args.n_galaxies, "--gas-disk-scale-length-kpc"
    )
    args.gas_disk_scale_height_kpc = _resolve_per_galaxy_values(
        args.gas_disk_scale_height_kpc, args.n_galaxies, "--gas-disk-scale-height-kpc"
    )
    args.Q_gas = _resolve_per_galaxy_values(args.Q_gas, args.n_galaxies, "--Q-gas")
    args.n_arms = _resolve_per_galaxy_values(args.n_arms, args.n_galaxies, "--n-arms")
    args.pitch_deg = _resolve_per_galaxy_values(args.pitch_deg, args.n_galaxies, "--pitch-deg")
    args.arm_strength = _resolve_per_galaxy_values(
        args.arm_strength, args.n_galaxies, "--arm-strength"
    )
    args.arm_stream_frac = _resolve_per_galaxy_values(
        args.arm_stream_frac, args.n_galaxies, "--arm-stream-frac"
    )
    args.bar_strength = _resolve_per_galaxy_values(
        args.bar_strength, args.n_galaxies, "--bar-strength"
    )
    args.bar_radius = _resolve_per_galaxy_values(
        args.bar_radius, args.n_galaxies, "--bar-radius"
    )
    args.bar_q = _resolve_per_galaxy_values(args.bar_q, args.n_galaxies, "--bar-q")
    args.bar_angle = _resolve_per_galaxy_values(args.bar_angle, args.n_galaxies, "--bar-angle")

    if args.inclination_deg is not None:
        args.inclination_deg = _resolve_per_galaxy_values(
            args.inclination_deg, args.n_galaxies, "--inclination-deg"
        )
    else:
        args.inclination_deg = [0.0] * args.n_galaxies


def _resolve_axis_values(
    values: list[float] | None,
    n_galaxies: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    if len(values) != n_galaxies:
        raise ValueError(f"{name} expects {n_galaxies} values, got {len(values)}")
    return np.asarray(values, dtype=float)


def _resolve_galaxy_placement(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    xs = _resolve_axis_values(args.xs, args.n_galaxies, "--xs")
    ys = _resolve_axis_values(args.ys, args.n_galaxies, "--ys")
    zs = _resolve_axis_values(args.zs, args.n_galaxies, "--zs")
    positions = None
    if xs is not None or ys is not None or zs is not None:
        box_center = args.box_kpc / 2.0
        positions = np.column_stack([
            np.full(args.n_galaxies, box_center, dtype=float) if xs is None else xs,
            np.full(args.n_galaxies, box_center, dtype=float) if ys is None else ys,
            np.full(args.n_galaxies, box_center, dtype=float) if zs is None else zs,
        ])

        if np.any(positions < 0.0) or np.any(positions > args.box_kpc):
            raise ValueError(
                "Galaxy positions from --xs, --ys, and --zs must lie within 0 and --box-kpc"
            )

        positions = positions - box_center

    vxs = _resolve_axis_values(args.vxs, args.n_galaxies, "--vxs")
    vys = _resolve_axis_values(args.vys, args.n_galaxies, "--vys")
    vzs = _resolve_axis_values(args.vzs, args.n_galaxies, "--vzs")
    velocities = None if vxs is None and vys is None and vzs is None else np.column_stack([
        np.zeros(args.n_galaxies, dtype=float) if vxs is None else vxs,
        np.zeros(args.n_galaxies, dtype=float) if vys is None else vys,
        np.zeros(args.n_galaxies, dtype=float) if vzs is None else vzs,
    ])

    if positions is not None:
        if velocities is None:
            velocities = np.zeros((args.n_galaxies, 3), dtype=float)
        return positions, velocities

    if args.n_galaxies == 1:
        return np.zeros((1, 3), dtype=float), np.zeros((1, 3), dtype=float)

    raise ValueError("Provide per-galaxy positions with --xs, --ys, and --zs for any multi-galaxy configuration")


def generate_galaxy_particles(
    galaxy_id: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict:
    """Generate positions, velocities, and masses for a single isolated spiral."""

    m200_msun = _get_galaxy_value(args.m200_msun, galaxy_id)
    M_star = _get_galaxy_value(args.m_star_msun, galaxy_id)
    M_gas = _get_galaxy_value(args.m_gas_msun, galaxy_id)
    m_bulge_msun = _get_galaxy_value(args.m_bulge_msun, galaxy_id)
    rd_star_kpc = _get_galaxy_value(args.stellar_disk_scale_length_kpc, galaxy_id)
    zd_star_kpc = _get_galaxy_value(args.stellar_disk_scale_height_kpc, galaxy_id)
    rd_gas_kpc = _get_galaxy_value(args.gas_disk_scale_length_kpc, galaxy_id)
    zd_gas_kpc = _get_galaxy_value(args.gas_disk_scale_height_kpc, galaxy_id)
    bulge_a_kpc = _get_galaxy_value(args.bulge_a_kpc, galaxy_id)

    # 1. Sample halo positions
    c200 = _get_galaxy_value(args.c200, galaxy_id)
    r_s, _ = nfw_params(m200_msun, c200)

    # Calculate truncation radius
    r_max_halo = r_s * 10

    n_halo = _get_galaxy_value(args.n_halo, galaxy_id)
    pos_halo = np.zeros((n_halo, 3))
    mass_halo = _component_masses(m200_msun, n_halo)

    if n_halo > 0:
        x_halo, y_halo, z_halo = sample_nfw_halo(
            n_halo, m200_msun, c200, r_max_halo, rng
        )
        pos_halo = np.column_stack([x_halo, y_halo, z_halo])

    # 2. Sample bulge positions
    n_bulge = _get_galaxy_value(args.n_bulge, galaxy_id)
    pos_bulge = np.zeros((n_bulge, 3))
    mass_bulge = _component_masses(m_bulge_msun, n_bulge)

    if n_bulge > 0:
        r_max_bulge = _get_galaxy_value(args.bulge_rmax_scale, galaxy_id) * bulge_a_kpc
        x_bulge, y_bulge, z_bulge = sample_hernquist_bulge(
            n_bulge, m_bulge_msun, bulge_a_kpc, rng, r_max=r_max_bulge
        )
        pos_bulge = np.column_stack([x_bulge, y_bulge, z_bulge])

    # 3. Sample stellar disc positions
    n_star = _get_galaxy_value(args.n_star, galaxy_id)
    pos_star = np.zeros((n_star, 3))
    mass_star = _component_masses(M_star, n_star)

    arm_strength = _get_galaxy_value(args.arm_strength, galaxy_id)
    n_arms = _get_galaxy_value(args.n_arms, galaxy_id)
    pitch_deg = _get_galaxy_value(args.pitch_deg, galaxy_id)
    bar_strength = _get_galaxy_value(args.bar_strength, galaxy_id)
    bar_radius = _get_galaxy_value(args.bar_radius, galaxy_id)
    bar_q = _get_galaxy_value(args.bar_q, galaxy_id)
    bar_angle = _get_galaxy_value(args.bar_angle, galaxy_id)

    if n_star > 0:
        x_star, y_star, z_star = sample_exponential_disc(
            n_star, M_star, rd_star_kpc, zd_star_kpc, rng,
            spiral_params={
                "arm_strength": arm_strength,
                "n_arms": n_arms,
                "pitch_deg": pitch_deg,
            } if arm_strength > 0 else None,
            bar_params={
                "enabled": args.bar_enabled,
                "strength": bar_strength,
                "radius": bar_radius,
                "q": bar_q,
                "angle": bar_angle,
            } if args.bar_enabled else None,
        )
        pos_star = np.column_stack([x_star, y_star, z_star])

    # 4. Sample gas disc positions
    n_gas = _get_galaxy_value(args.n_gas, galaxy_id)
    pos_gas = np.zeros((n_gas, 3))
    mass_gas = _component_masses(M_gas, n_gas)

    if n_gas > 0:
        x_gas, y_gas, z_gas = sample_exponential_disc(
            n_gas, M_gas, rd_gas_kpc, zd_gas_kpc, rng,
            spiral_params={
                "arm_strength": arm_strength,
                "n_arms": n_arms,
                "pitch_deg": pitch_deg,
            } if arm_strength > 0 else None,
            bar_params={
                "enabled": args.bar_enabled,
                "strength": bar_strength,
                "radius": bar_radius,
                "q": bar_q,
                "angle": bar_angle,
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
        _get_galaxy_value(args.Q_star, galaxy_id),
        rng,
        grid_solver,
        is_gas=False,
        spiral_params={
            "arm_strength": arm_strength,
            "stream_frac": _get_galaxy_value(args.arm_stream_frac, galaxy_id),
            "n_arms": n_arms,
            "pitch_deg": pitch_deg,
        } if arm_strength > 0 else None,
        bar_params={
            "enabled": args.bar_enabled,
            "stream_frac": bar_strength,
            "radius": bar_radius,
            "angle": bar_angle,
        } if args.bar_enabled else None,
    )
    vel_gas = _sample_cylindrical_disc_velocities(
        pos_gas,
        mass_gas,
        M_gas,
        rd_gas_kpc,
        zd_gas_kpc,
        _get_galaxy_value(args.Q_gas, galaxy_id),
        rng,
        grid_solver,
        is_gas=True,
        spiral_params={
            "arm_strength": arm_strength,
            "stream_frac": _get_galaxy_value(args.arm_stream_frac, galaxy_id),
            "n_arms": n_arms,
            "pitch_deg": pitch_deg,
        } if arm_strength > 0 else None,
        bar_params={
            "enabled": args.bar_enabled,
            "stream_frac": bar_strength,
            "radius": bar_radius,
            "angle": bar_angle,
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
        "--dm-mass-msun",
        nargs="+",
        type=float,
        default=None,
        help="Dark matter halo mass of each galaxy in M_sun.",
    )
    parser.add_argument(
        "--dm-part-mass-msun",
        type=float,
        default=None,
        help="Dark matter particle mass in M_sun.",
    )
    parser.add_argument(
        "--bulge-fraction",
        nargs="+",
        type=float,
        default=None,
        help="Bulge-to-total stellar mass fraction B / (D + B) for each galaxy.",
    )
    parser.add_argument(
        "--star-mass-msun",
        nargs="+",
        type=float,
        default=None,
        help="Stellar disc mass of each galaxy in M_sun.",
    )
    parser.add_argument(
        "--star-part-mass-msun",
        type=float,
        default=None,
        help="Stellar particle mass in M_sun.",
    )
    parser.add_argument(
        "--gas-mass-msun",
        nargs="+",
        type=float,
        default=None,
        help="Gas disc mass of each galaxy in M_sun.",
    )
    parser.add_argument(
        "--gas-part-mass-msun",
        type=float,
        default=None,
        help="Gas particle mass in M_sun.",
    )
    parser.add_argument(
        "--n-galaxies", type=int, default=1, help="Number of galaxies to generate."
    )
    parser.add_argument(
        "--inclination-deg",
        nargs="+",
        type=float,
        default=None,
        help="Per-galaxy disc inclinations in degrees.",
    )
    parser.add_argument(
        "--xs",
        nargs="+",
        type=float,
        default=None,
        help="Per-galaxy x coordinates in the box in kpc.",
    )
    parser.add_argument(
        "--ys",
        nargs="+",
        type=float,
        default=None,
        help="Per-galaxy y coordinates in the box in kpc.",
    )
    parser.add_argument(
        "--zs",
        nargs="+",
        type=float,
        default=None,
        help="Per-galaxy z coordinates in the box in kpc.",
    )
    parser.add_argument(
        "--vxs",
        nargs="+",
        type=float,
        default=None,
        help="Per-galaxy bulk x velocities in km/s.",
    )
    parser.add_argument(
        "--vys",
        nargs="+",
        type=float,
        default=None,
        help="Per-galaxy bulk y velocities in km/s.",
    )
    parser.add_argument(
        "--vzs",
        nargs="+",
        type=float,
        default=None,
        help="Per-galaxy bulk z velocities in km/s.",
    )

    # Halo properties
    parser.add_argument(
        "--c200", nargs="+", type=float, default=[10.0], help="NFW concentration per galaxy."
    )

    # Bulge properties
    parser.add_argument(
        "--bulge-a-kpc", nargs="+", type=float, default=[0.8], help="Hernquist bulge scale length per galaxy in kpc."
    )
    parser.add_argument(
        "--bulge-rmax-scale",
        nargs="+",
        type=float,
        default=[50.0],
        help="Truncate Hernquist bulge sampling at this many scale lengths per galaxy.",
    )

    # Stellar disc properties
    parser.add_argument(
        "--stellar-disk-scale-length-kpc",
        nargs="+",
        type=float,
        default=[3.5],
        help="Stellar disk scale length per galaxy in kpc.",
    )
    parser.add_argument(
        "--stellar-disk-scale-height-kpc",
        nargs="+",
        type=float,
        default=[0.35],
        help="Stellar disk scale height per galaxy in kpc.",
    )
    parser.add_argument(
        "--Q-star", nargs="+", type=float, default=[1.5], help="Stellar-disc Toomre Q per galaxy."
    )

    # Gas disc properties
    parser.add_argument(
        "--gas-disk-scale-length-kpc",
        nargs="+",
        type=float,
        default=[7.0],
        help="Gas disk scale length per galaxy in kpc.",
    )
    parser.add_argument(
        "--gas-disk-scale-height-kpc",
        nargs="+",
        type=float,
        default=[0.1],
        help="Gas disk scale height per galaxy in kpc.",
    )
    parser.add_argument(
        "--Q-gas", nargs="+", type=float, default=[1.0], help="Gas-disc Toomre Q per galaxy."
    )

    # Spiral arm properties
    parser.add_argument(
        "--n-arms", nargs="+", type=int, default=[2], help="Number of spiral arms per galaxy."
    )
    parser.add_argument(
        "--pitch-deg", nargs="+", type=float, default=[15.0], help="Spiral-arm pitch angle per galaxy in degrees."
    )
    parser.add_argument(
        "--arm-strength", nargs="+", type=float, default=[0.3], help="Spiral-arm strength per galaxy (0-1)."
    )
    parser.add_argument(
        "--arm-stream-frac", nargs="+", type=float, default=[0.1], help="Spiral-arm streaming fraction per galaxy."
    )

    # Bar properties
    parser.add_argument(
        "--bar-enabled", action="store_true", help="Enable a galactic bar."
    )
    parser.add_argument(
        "--bar-strength", nargs="+", type=float, default=[0.1], help="Bar strength per galaxy."
    )
    parser.add_argument(
        "--bar-radius", nargs="+", type=float, default=[3.0], help="Bar radius per galaxy in kpc."
    )
    parser.add_argument(
        "--bar-q", nargs="+", type=float, default=[0.3], help="Bar flattening q per galaxy."
    )
    parser.add_argument(
        "--bar-angle", nargs="+", type=float, default=[0.0], help="Bar angle per galaxy in degrees."
    )

    # Simulation properties
    parser.add_argument(
        "--max-timestep-gyr", type=float, default=0.8, help="Maximum simulation time-step in Gyr."
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
    parser.add_argument(
        "--feedback-scale",
        type=float,
        default=1.0,
        help="Relative SNII feedback-energy scaling applied to the EAGLE feedback fractions.",
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
    parser.add_argument(
        "--h-max-cell-fraction",
        type=float,
        default=0.5,
        help="Set h_max to this fraction of the top-level cell width in the generated SWIFT YAML.",
    )
    parser.add_argument(
        "--scheduler-tasks-per-cell",
        type=int,
        default=100,
        help="Set Scheduler.tasks_per_cell in the generated SWIFT YAML.",
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
    _normalise_per_galaxy_args(args)
    galaxy_offsets, galaxy_bulk_velocities = _resolve_galaxy_placement(args)

    # --- Initialize RNG ---
    rng = get_rng(args.seed)

    # --- Generate galaxy positions and masses ---
    print("======================================================================")
    print("SWIFT SPIRAL ICs - Initial Conditions Generator")
    print("======================================================================")

    all_galaxies_pos_mass = []

    for i in range(args.n_galaxies):
        print(f"Generating positions for galaxy {i}...")
        galaxy_data = generate_galaxy_particles(i, args, rng)
        _place_galaxy(galaxy_data, galaxy_offsets[i], galaxy_bulk_velocities[i], args.inclination_deg[i])

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
        component_particle_masses = []
        for galaxy_data in all_galaxies_pos_mass:
            for component in ("dm", "gas", "stars", "bulge"):
                masses = galaxy_data[component]["mass"]
                if masses.size > 0:
                    component_particle_masses.append(float(masses[0]))
        background_particle_mass = min(component_particle_masses) if component_particle_masses else 1e7
        initial_combined_data = add_uniform_background(
            initial_combined_data,
            args.box_kpc,
            background_particle_mass,
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
        dt_max_gyr=args.max_timestep_gyr,
        softening_kpc=args.eps_grid,
        output_basename=args.snapshot_basename,
        run_name=args.run_name,
        param_template=args.param_template,
        min_gas_mass_msun=min_gas_mass,
        feedback_scale=args.feedback_scale,
        h_max_cell_fraction=args.h_max_cell_fraction,
        scheduler_tasks_per_cell=args.scheduler_tasks_per_cell,
    )
    with open(args.out_params, "w") as f:
        f.write(params)
    print(f"Parameter file written to {args.out_params}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
