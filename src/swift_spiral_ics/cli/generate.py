"""SWIFT Initial Conditions Generator.
Command line interface for generating initial conditions for SWIFT simulations.
"""

import argparse
import sys

import numpy as np
import yaml

from ..io.swift_writer import compute_internal_energy, write_swift_ic
from ..io.yaml_writer import generate_swift_params
from ..physics.grid_solver import GalaxyGridSolver
from ..physics.kinematics import escape_velocity_from_grid
from ..physics.orbits import parabolic_orbit_initial_conditions
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
    if total_mass <= 0.0:
        return 0
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
    args.black_hole_mass_msun = _resolve_per_galaxy_values(
        args.black_hole_mass_msun,
        args.n_galaxies,
        "black_hole.mass_msun",
    )
    args.cgm_enabled = _resolve_per_galaxy_values(args.cgm_enabled, args.n_galaxies, "cgm.enabled")
    args.cgm_mass_msun = _resolve_per_galaxy_values(args.cgm_mass_msun, args.n_galaxies, "cgm.mass_msun")
    args.cgm_r_min_kpc = _resolve_per_galaxy_values(args.cgm_r_min_kpc, args.n_galaxies, "cgm.r_min_kpc")
    args.cgm_r_max_kpc = _resolve_per_galaxy_values(args.cgm_r_max_kpc, args.n_galaxies, "cgm.r_max_kpc")
    args.cgm_core_radius_kpc = _resolve_per_galaxy_values(
        args.cgm_core_radius_kpc, args.n_galaxies, "cgm.core_radius_kpc"
    )
    args.cgm_beta = _resolve_per_galaxy_values(args.cgm_beta, args.n_galaxies, "cgm.beta")
    args.cgm_temperature_floor_K = _resolve_per_galaxy_values(
        args.cgm_temperature_floor_K, args.n_galaxies, "cgm.temperature_floor_K"
    )
    args.cgm_temperature_ceiling_K = _resolve_per_galaxy_values(
        args.cgm_temperature_ceiling_K, args.n_galaxies, "cgm.temperature_ceiling_K"
    )
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
    if args.node_angle_deg is not None:
        args.node_angle_deg = _resolve_per_galaxy_values(
            args.node_angle_deg, args.n_galaxies, "placement.node_angle_deg"
        )
    else:
        args.node_angle_deg = [0.0] * args.n_galaxies


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


def _total_galaxy_masses(args: argparse.Namespace) -> np.ndarray:
    return np.asarray(
        [
            args.m200_msun[i]
            + args.m_star_msun[i]
            + args.m_bulge_msun[i]
            + args.m_gas_msun[i]
            + args.black_hole_mass_msun[i]
            + (args.cgm_mass_msun[i] if args.cgm_enabled[i] else 0.0)
            for i in range(args.n_galaxies)
        ],
        dtype=float,
    )


def _resolve_parabolic_orbit_placement(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.n_galaxies != 2:
        raise ValueError("--orbit parabolic currently requires --n-galaxies 2")

    manual_position_args = (args.xs, args.ys, args.zs)
    manual_velocity_args = (args.vxs, args.vys, args.vzs)
    if any(value is not None for value in (*manual_position_args, *manual_velocity_args)):
        raise ValueError(
            "--orbit parabolic computes positions and velocities; do not also provide "
            "--xs, --ys, --zs, --vxs, --vys, or --vzs"
        )

    if args.orbit_r_init_kpc is None:
        raise ValueError("--orbit-r-init-kpc is required when --orbit parabolic")
    if args.orbit_r_peri_kpc is None:
        raise ValueError("--orbit-r-peri-kpc is required when --orbit parabolic")
    if args.orbit_r_init_kpc <= 0.0:
        raise ValueError("--orbit-r-init-kpc must be positive")
    if args.orbit_r_peri_kpc < 0.0:
        raise ValueError("--orbit-r-peri-kpc must be non-negative")

    masses = _total_galaxy_masses(args)
    rel_pos, rel_vel = parabolic_orbit_initial_conditions(
        masses[0],
        masses[1],
        args.orbit_r_init_kpc,
        args.orbit_r_peri_kpc,
        args.orbit_plane_angle_deg,
    )

    total_mass = masses.sum()
    positions = np.vstack([
        -masses[1] / total_mass * rel_pos,
        masses[0] / total_mass * rel_pos,
    ])
    velocities = np.vstack([
        -masses[1] / total_mass * rel_vel,
        masses[0] / total_mass * rel_vel,
    ])

    box_center = args.box_kpc / 2.0
    box_positions = positions + box_center
    if np.any(box_positions < 0.0) or np.any(box_positions > args.box_kpc):
        raise ValueError(
            "Parabolic orbit galaxy centres lie outside the box; increase --box-kpc or reduce "
            "--orbit-r-init-kpc"
        )

    return positions, velocities


def _rotate_orbit_plane(pos: np.ndarray, vel: np.ndarray, plane_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    if plane_angle_deg == 0.0:
        return pos, vel

    angle_rad = np.deg2rad(plane_angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot_matrix = np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])
    return rot_matrix @ pos, rot_matrix @ vel


def _galaxy_index_by_name(args: argparse.Namespace, name: str) -> int:
    try:
        return args.galaxy_names.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown host galaxy '{name}' in relative placement") from exc


def _apply_host_relative_placements(
    args: argparse.Namespace,
    positions: np.ndarray,
    velocities: np.ndarray,
    start_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    for i in range(start_index, args.n_galaxies):
        host_name = args.relative_to[i]
        rel_pos = args.relative_position_kpc[i]
        rel_vel = args.relative_velocity_kms[i]
        if host_name is None or rel_pos is None or rel_vel is None:
            raise ValueError(
                "Additional galaxies with orbit.type relative_velocity must specify "
                "placement.relative_to, placement.relative_position_kpc, and "
                "placement.relative_velocity_kms"
            )
        host_index = _galaxy_index_by_name(args, host_name)
        if host_index >= i:
            raise ValueError("Host-relative galaxies must appear after their host in the YAML file")
        positions[i] = positions[host_index] + np.asarray(rel_pos, dtype=float)
        velocities[i] = velocities[host_index] + np.asarray(rel_vel, dtype=float)
    return positions, velocities


def _center_all_galaxies_on_com(
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = positions - np.average(positions, axis=0, weights=masses)
    velocities = velocities - np.average(velocities, axis=0, weights=masses)
    return positions, velocities


def _resolve_relative_velocity_orbit_placement(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.n_galaxies < 2:
        raise ValueError("orbit.type relative_velocity requires at least two galaxies")

    manual_position_args = (args.xs, args.ys, args.zs)
    manual_velocity_args = (args.vxs, args.vys, args.vzs)
    if any(value is not None for value in (*manual_position_args, *manual_velocity_args)):
        raise ValueError(
            "orbit.type relative_velocity computes positions and velocities; do not also provide "
            "manual galaxy positions or velocities"
        )

    if args.orbit_separation_kpc is None:
        raise ValueError("orbit.separation_kpc is required when orbit.type is relative_velocity")
    if args.orbit_radial_velocity_kms is None:
        raise ValueError("orbit.radial_velocity_kms is required when orbit.type is relative_velocity")
    if args.orbit_tangential_velocity_kms is None:
        raise ValueError("orbit.tangential_velocity_kms is required when orbit.type is relative_velocity")
    if args.orbit_separation_kpc <= 0.0:
        raise ValueError("orbit.separation_kpc must be positive")

    rel_pos = np.array([args.orbit_separation_kpc, 0.0, 0.0], dtype=float)
    rel_vel = np.array([
        args.orbit_radial_velocity_kms,
        args.orbit_tangential_velocity_kms,
        0.0,
    ], dtype=float)
    rel_pos, rel_vel = _rotate_orbit_plane(rel_pos, rel_vel, args.orbit_plane_angle_deg)

    masses = _total_galaxy_masses(args)
    pair_mass = masses[:2].sum()
    positions = np.zeros((args.n_galaxies, 3), dtype=float)
    velocities = np.zeros((args.n_galaxies, 3), dtype=float)
    positions[:2] = np.vstack([
        -masses[1] / pair_mass * rel_pos,
        masses[0] / pair_mass * rel_pos,
    ])
    velocities[:2] = np.vstack([
        -masses[1] / pair_mass * rel_vel,
        masses[0] / pair_mass * rel_vel,
    ])
    positions, velocities = _apply_host_relative_placements(args, positions, velocities, 2)
    positions, velocities = _center_all_galaxies_on_com(masses, positions, velocities)

    box_center = args.box_kpc / 2.0
    box_positions = positions + box_center
    if np.any(box_positions < 0.0) or np.any(box_positions > args.box_kpc):
        raise ValueError(
            "Relative-velocity orbit galaxy centres lie outside the box; increase "
            "simulation.box_kpc or reduce orbit.separation_kpc"
        )

    return positions, velocities


def _resolve_galaxy_placement(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.orbit == "parabolic":
        return _resolve_parabolic_orbit_placement(args)
    if args.orbit == "relative_velocity":
        return _resolve_relative_velocity_orbit_placement(args)

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
    m_black_hole_msun = _get_galaxy_value(args.black_hole_mass_msun, galaxy_id)
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
            # Keep collisionless stellar discs axisymmetric. Imposed stellar
            # spiral/bar overdensities are not paired with a matching live
            # non-axisymmetric potential, so they are not equilibrium ICs.
            spiral_params=None,
            bar_params=None,
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
        spiral_params=None,
        bar_params=None,
    )
    vel_star = _remove_disc_streaming_modes(pos_star, vel_star)
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

    internal_energy_gas = _hydrostatic_disc_internal_energy(pos_gas, zd_gas_kpc, grid_solver)
    if _get_galaxy_value(args.cgm_enabled, galaxy_id):
        cgm_mass_msun = _get_galaxy_value(args.cgm_mass_msun, galaxy_id)
        if cgm_mass_msun > 0.0:
            n_cgm = _allocate_total_particles(cgm_mass_msun, args.gas_part_mass_msun, "cgm.mass_msun")
            pos_cgm = _sample_cgm_beta_profile(
                n_cgm,
                _get_galaxy_value(args.cgm_r_min_kpc, galaxy_id),
                _get_galaxy_value(args.cgm_r_max_kpc, galaxy_id),
                _get_galaxy_value(args.cgm_core_radius_kpc, galaxy_id),
                _get_galaxy_value(args.cgm_beta, galaxy_id),
                rng,
            )
            vel_cgm = np.zeros_like(pos_cgm)
            mass_cgm = _component_masses(cgm_mass_msun, n_cgm)
            radius_cgm = np.linalg.norm(pos_cgm, axis=1)
            internal_energy_cgm = _hydrostatic_cgm_internal_energy(
                radius_cgm,
                _get_galaxy_value(args.cgm_r_min_kpc, galaxy_id),
                _get_galaxy_value(args.cgm_r_max_kpc, galaxy_id),
                _get_galaxy_value(args.cgm_core_radius_kpc, galaxy_id),
                _get_galaxy_value(args.cgm_beta, galaxy_id),
                grid_solver,
                _get_galaxy_value(args.cgm_temperature_floor_K, galaxy_id),
                _get_galaxy_value(args.cgm_temperature_ceiling_K, galaxy_id),
            )
            pos_gas = np.vstack([pos_gas, pos_cgm])
            vel_gas = np.vstack([vel_gas, vel_cgm])
            mass_gas = np.concatenate([mass_gas, mass_cgm])
            internal_energy_gas = np.concatenate([internal_energy_gas, internal_energy_cgm])

    if m_black_hole_msun > 0.0:
        pos_black_hole = np.zeros((1, 3), dtype=float)
        vel_black_hole = np.zeros((1, 3), dtype=float)
        mass_black_hole = np.asarray([m_black_hole_msun], dtype=float)
    else:
        pos_black_hole = np.empty((0, 3), dtype=float)
        vel_black_hole = np.empty((0, 3), dtype=float)
        mass_black_hole = np.empty(0, dtype=float)

    # Combine positions and masses for a single galaxy
    galaxy_data = {
        "dm": {"pos": pos_halo, "vel": vel_halo, "mass": mass_halo},
        "gas": {
            "pos": pos_gas,
            "vel": vel_gas,
            "mass": mass_gas,
            "internal_energy": internal_energy_gas,
        },
        "stars": {"pos": pos_star, "vel": vel_star, "mass": mass_star},
        "bulge": {"pos": pos_bulge, "vel": vel_bulge, "mass": mass_bulge},
        "black_holes": {
            "pos": pos_black_hole,
            "vel": vel_black_hole,
            "mass": mass_black_hole,
            "subgrid_mass": mass_black_hole,
        },
    }

    _validate_generated_galaxy_stability(galaxy_id, galaxy_data, grid_solver)

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


def _sample_cgm_beta_profile(
    n_particles: int,
    r_min_kpc: float,
    r_max_kpc: float,
    core_radius_kpc: float,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_particles <= 0:
        return np.empty((0, 3), dtype=float)
    if r_min_kpc < 0 or r_max_kpc <= r_min_kpc:
        raise ValueError("CGM radii must satisfy 0 <= r_min_kpc < r_max_kpc")
    if core_radius_kpc <= 0:
        raise ValueError("CGM core_radius_kpc must be positive")
    if beta <= 0:
        raise ValueError("CGM beta must be positive")

    samples = []
    max_batch = max(1024, n_particles)
    while sum(len(batch) for batch in samples) < n_particles:
        radius = (
            rng.uniform(r_min_kpc**3, r_max_kpc**3, max_batch)
        ) ** (1.0 / 3.0)
        # Sample from volume and reject by the tapered beta-profile density term.
        density = _cgm_density_shape(radius, core_radius_kpc, beta, r_min_kpc, r_max_kpc)
        probe_radius = np.linspace(r_min_kpc, r_max_kpc, 512)
        density_max = float(np.max(_cgm_density_shape(
            probe_radius, core_radius_kpc, beta, r_min_kpc, r_max_kpc
        )))
        accept_prob = density / density_max
        draw = rng.uniform(0.0, 1.0, max_batch)
        accepted_radius = radius[draw < accept_prob]
        if accepted_radius.size == 0:
            continue
        cos_theta = rng.uniform(-1.0, 1.0, accepted_radius.size)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = rng.uniform(0.0, 2.0 * np.pi, accepted_radius.size)
        samples.append(np.column_stack([
            accepted_radius * sin_theta * np.cos(phi),
            accepted_radius * sin_theta * np.sin(phi),
            accepted_radius * cos_theta,
        ]))

    return np.vstack(samples)[:n_particles]


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, 0.0, 1.0)
    return x_clipped * x_clipped * (3.0 - 2.0 * x_clipped)


def _cgm_radial_taper(radius: np.ndarray, r_min_kpc: float, r_max_kpc: float) -> np.ndarray:
    span = r_max_kpc - r_min_kpc
    taper_width = max(0.05 * span, 5.0)
    taper_width = min(taper_width, 0.45 * span)
    inner = _smoothstep((radius - r_min_kpc) / taper_width)
    outer = _smoothstep((r_max_kpc - radius) / taper_width)
    return inner * outer


def _cgm_density_shape(
    radius: np.ndarray,
    core_radius_kpc: float,
    beta: float,
    r_min_kpc: float | None = None,
    r_max_kpc: float | None = None,
) -> np.ndarray:
    density = (1.0 + (radius / core_radius_kpc) ** 2) ** (-1.5 * beta)
    if r_min_kpc is not None and r_max_kpc is not None:
        density = density * _cgm_radial_taper(radius, r_min_kpc, r_max_kpc)
    return density


def _hydrostatic_disc_internal_energy(
    pos: np.ndarray,
    scale_height_kpc: float,
    grid_solver: GalaxyGridSolver,
) -> np.ndarray:
    """Estimate gas disc internal energy from vertical hydrostatic support."""

    if len(pos) == 0:
        return np.empty(0, dtype=float)
    if scale_height_kpc <= 0.0:
        raise ValueError("gas_disk.scale_height_kpc must be positive")

    radius = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    z_support = np.full_like(radius, max(scale_height_kpc, grid_solver.eps))
    support_force = np.abs(grid_solver.get_potential_and_forces(radius, z_support)["FZ"])

    # For an isothermal vertical layer, P/rho ~= H |dPhi/dz|. The SWIFT internal
    # energy is related by P/rho = (gamma - 1) u.
    gamma = 5.0 / 3.0
    sigma_sq = np.maximum(scale_height_kpc * support_force, 0.0)
    floor_u = compute_internal_energy(T=1.0e4)
    ceiling_u = compute_internal_energy(T=3.0e5)
    return np.clip(sigma_sq / (gamma - 1.0), floor_u, ceiling_u)


def _hydrostatic_cgm_internal_energy(
    radius: np.ndarray,
    r_min_kpc: float,
    r_max_kpc: float,
    core_radius_kpc: float,
    beta: float,
    grid_solver: GalaxyGridSolver,
    temperature_floor_K: float,
    temperature_ceiling_K: float,
) -> np.ndarray:
    if temperature_floor_K <= 0.0:
        raise ValueError("CGM temperature_floor_K must be positive")
    if temperature_ceiling_K < temperature_floor_K:
        raise ValueError("CGM temperature_ceiling_K must be >= temperature_floor_K")

    gamma = 5.0 / 3.0
    r_grid = np.linspace(r_min_kpc, r_max_kpc, 4096)
    density = _cgm_density_shape(r_grid, core_radius_kpc, beta, r_min_kpc, r_max_kpc)
    gravity = np.abs(grid_solver.get_potential_and_forces(r_grid, np.zeros_like(r_grid))["FR"])
    integrand = density * gravity

    dr = np.diff(r_grid)
    trapezoids = 0.5 * (integrand[:-1] + integrand[1:]) * dr
    pressure_integral = np.zeros_like(r_grid)
    pressure_integral[:-1] = np.cumsum(trapezoids[::-1])[::-1]

    boundary_internal_energy = compute_internal_energy(T=temperature_floor_K)
    boundary_sigma2 = (gamma - 1.0) * boundary_internal_energy
    boundary_pressure = max(float(np.max(density)) * 1.0e-6, density[-1]) * boundary_sigma2
    sigma2_grid = (boundary_pressure + pressure_integral) / np.maximum(density, 1.0e-300)
    internal_energy_grid = sigma2_grid / (gamma - 1.0)

    floor_u = compute_internal_energy(T=temperature_floor_K)
    ceiling_u = compute_internal_energy(T=temperature_ceiling_K)
    clipped_energy_grid = np.clip(internal_energy_grid, floor_u, ceiling_u)
    _validate_cgm_hydrostatic_residual(
        r_grid, density, gravity, clipped_energy_grid, floor_u, ceiling_u, internal_energy_grid
    )
    return np.interp(radius, r_grid, clipped_energy_grid)


def _validate_cgm_hydrostatic_residual(
    radius: np.ndarray,
    density: np.ndarray,
    gravity: np.ndarray,
    internal_energy: np.ndarray,
    floor_u: float,
    ceiling_u: float,
    unclipped_internal_energy: np.ndarray,
) -> None:
    gamma = 5.0 / 3.0
    pressure = density * (gamma - 1.0) * internal_energy
    dpressure_dr = np.gradient(pressure, radius)
    force_density = density * gravity
    taper_mask = density > 0.1 * np.max(density)
    valid = taper_mask & (force_density > 0.0)
    if not np.any(valid):
        raise ValueError("CGM hydrostatic validation failed: no supported CGM region")

    residual = np.abs(dpressure_dr[valid] + force_density[valid]) / force_density[valid]
    median_residual = float(np.median(residual))
    ceiling_clip_fraction = float(np.mean(unclipped_internal_energy[valid] > ceiling_u))
    if median_residual > 0.5 or ceiling_clip_fraction > 0.25:
        raise ValueError(
            "CGM is not hydrostatic for this configuration: "
            f"median residual={median_residual:.2f}, "
            f"ceiling-clipped fraction={ceiling_clip_fraction:.2f}. "
            "Increase cgm.temperature_ceiling_K, decrease the CGM edge sharpness by widening "
            "r_max_kpc-r_min_kpc, or reduce the CGM mass/inner density."
        )


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


def _remove_disc_streaming_modes(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Remove coherent annular radial/vertical drift from collisionless discs."""

    if len(pos) < 20:
        return vel

    radius = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    nonzero_radius = radius > 0.0
    if np.count_nonzero(nonzero_radius) < 20:
        return vel

    out = vel.copy()
    cos_phi = np.zeros_like(radius)
    sin_phi = np.zeros_like(radius)
    cos_phi[nonzero_radius] = pos[nonzero_radius, 0] / radius[nonzero_radius]
    sin_phi[nonzero_radius] = pos[nonzero_radius, 1] / radius[nonzero_radius]

    v_radial = out[:, 0] * cos_phi + out[:, 1] * sin_phi
    n_bins = min(64, max(4, len(pos) // 200))
    edges = np.quantile(radius[nonzero_radius], np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return out

    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (radius >= lo) & (radius <= hi) & nonzero_radius
        if np.count_nonzero(in_bin) < 10:
            continue
        radial_mean = float(np.mean(v_radial[in_bin]))
        vertical_mean = float(np.mean(out[in_bin, 2]))
        out[in_bin, 0] -= radial_mean * cos_phi[in_bin]
        out[in_bin, 1] -= radial_mean * sin_phi[in_bin]
        out[in_bin, 2] -= vertical_mean

    return out


def _validate_generated_galaxy_stability(
    galaxy_id: int,
    galaxy_data: dict,
    grid_solver: GalaxyGridSolver,
) -> None:
    """Reject generated galaxies with obvious disequilibrium before writing ICs."""

    for component_name in ("dm", "stars", "bulge"):
        component = galaxy_data[component_name]
        pos = component["pos"]
        vel = component["vel"]
        if len(pos) == 0:
            continue
        radius = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        speed = np.linalg.norm(vel, axis=1)
        v_escape = escape_velocity_from_grid(radius, pos[:, 2], grid_solver)
        near_unbound = float(np.mean(speed > 0.95 * v_escape))
        if near_unbound > 0.01:
            raise ValueError(
                f"Galaxy {galaxy_id} {component_name} is not stable: "
                f"{near_unbound:.2%} of particles are within 5% of escape speed"
            )

    stars = galaxy_data["stars"]
    if len(stars["pos"]) >= 100:
        _validate_stellar_disc_streaming(galaxy_id, stars["pos"], stars["vel"])


def _validate_stellar_disc_streaming(galaxy_id: int, pos: np.ndarray, vel: np.ndarray) -> None:
    radius = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    nonzero_radius = radius > 0.0
    if np.count_nonzero(nonzero_radius) < 100:
        return

    cos_phi = np.zeros_like(radius)
    sin_phi = np.zeros_like(radius)
    cos_phi[nonzero_radius] = pos[nonzero_radius, 0] / radius[nonzero_radius]
    sin_phi[nonzero_radius] = pos[nonzero_radius, 1] / radius[nonzero_radius]
    v_radial = vel[:, 0] * cos_phi + vel[:, 1] * sin_phi
    v_phi = -vel[:, 0] * sin_phi + vel[:, 1] * cos_phi

    n_bins = min(32, max(4, len(pos) // 500))
    edges = np.unique(np.quantile(radius[nonzero_radius], np.linspace(0.0, 1.0, n_bins + 1)))
    worst_radial = 0.0
    worst_vertical = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (radius >= lo) & (radius <= hi) & nonzero_radius
        if np.count_nonzero(in_bin) < 25:
            continue
        rotation = max(abs(float(np.median(v_phi[in_bin]))), 1.0)
        radial_ratio = abs(float(np.mean(v_radial[in_bin]))) / rotation
        vertical_ratio = abs(float(np.mean(vel[in_bin, 2]))) / rotation
        worst_radial = max(worst_radial, radial_ratio)
        worst_vertical = max(worst_vertical, vertical_ratio)

    if worst_radial > 0.03 or worst_vertical > 0.03:
        raise ValueError(
            f"Galaxy {galaxy_id} stellar disc is not stable: coherent streaming remains "
            f"(max <vR>/vphi={worst_radial:.3f}, max <vz>/vphi={worst_vertical:.3f})"
        )


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


def _rotate_z(values: np.ndarray, angle_deg: float) -> np.ndarray:
    if len(values) == 0 or angle_deg == 0.0:
        return values
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    out = values.copy()
    x = values[:, 0]
    y = values[:, 1]
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    return out


def _rotate_disc_orientation(values: np.ndarray, inclination: float, node_angle: float) -> np.ndarray:
    return _rotate_x(_rotate_z(values, node_angle), inclination)


def _place_galaxy(
    galaxy_data: dict,
    offset: np.ndarray,
    bulk_velocity: np.ndarray,
    inclination: float,
    node_angle: float,
):
    for component in galaxy_data.values():
        component["pos"] = _rotate_disc_orientation(component["pos"], inclination, node_angle) + offset
        component["vel"] = _rotate_disc_orientation(component["vel"], inclination, node_angle) + bulk_velocity


def add_uniform_background(
    initial_data: dict, # Renamed from combined_data for clarity
    box_size: float,
    m_part: float,
    rho_gas: float,
    rho_dm: float,
    grid_spacing: float,
    radius: float | None,
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
        radius: Optional spherical cutoff radius for background particles (kpc).
        rng: Random number generator.

    Returns:
        Dict with updated particle data (pos, vel, mass) including background.
    """
    volume = box_size**3

    # Initialize updated dictionary with existing galaxy data, and ensure 'vel' key exists
    updated = {
        "dm": {"pos": initial_data["dm"]["pos"], "mass": initial_data["dm"]["mass"], "vel": initial_data["dm"]["vel"]},
        "gas": {
            "pos": initial_data["gas"]["pos"],
            "mass": initial_data["gas"]["mass"],
            "vel": initial_data["gas"]["vel"],
            "internal_energy": initial_data["gas"].get(
                "internal_energy",
                np.full(len(initial_data["gas"]["pos"]), compute_internal_energy(T=1e4)),
            ),
        },
        "stars": {"pos": initial_data["stars"]["pos"], "mass": initial_data["stars"]["mass"], "vel": initial_data["stars"]["vel"]},
        "bulge": {"pos": initial_data["bulge"]["pos"], "mass": initial_data["bulge"]["mass"], "vel": initial_data["bulge"]["vel"]},
        "black_holes": {
            "pos": initial_data["black_holes"]["pos"],
            "mass": initial_data["black_holes"]["mass"],
            "vel": initial_data["black_holes"]["vel"],
            "subgrid_mass": initial_data["black_holes"].get(
                "subgrid_mass", initial_data["black_holes"]["mass"]
            ),
        },
    }

    use_grid = grid_spacing > 0
    half_box = box_size / 2.0
    if radius is not None and radius <= 0:
        raise ValueError("background radius must be positive")

    radius_sq = None if radius is None else radius**2

    def _keep_in_background_region(pos: np.ndarray) -> np.ndarray:
        if radius_sq is None or pos.size == 0:
            return pos
        return pos[np.sum(pos**2, axis=1) <= radius_sq]

    def _limit_to_background_region(pos: np.ndarray) -> np.ndarray:
        if radius is None or pos.size == 0:
            return pos
        radii = np.linalg.norm(pos, axis=1)
        mask = radii > radius
        if not np.any(mask):
            return pos
        clipped = pos.copy()
        clipped[mask] *= (radius / radii[mask])[:, None]
        return clipped

    if radius is None:
        target_volume = volume
    else:
        target_volume = (4.0 / 3.0) * np.pi * radius**3

    if not use_grid:
        # Add random background particles
        if rho_dm > 0:
            n_dm = int(round(rho_dm * target_volume / m_part)) # Uses m_part for random bg
            if n_dm > 0:
                if radius is None:
                    pos_dm = rng.uniform(-half_box, half_box, (n_dm, 3))
                else:
                    pos_dm = np.empty((0, 3), dtype=float)
                    while len(pos_dm) < n_dm:
                        trial_pos = rng.uniform(-radius, radius, (max(n_dm, 1024), 3))
                        accepted = _keep_in_background_region(trial_pos)
                        if accepted.size == 0:
                            continue
                        pos_dm = np.vstack([pos_dm, accepted])
                    pos_dm = pos_dm[:n_dm]
                vel_dm = np.zeros((n_dm, 3), dtype=float)
                mass_dm = np.full(n_dm, m_part)

                updated["dm"]["pos"] = np.vstack([updated["dm"]["pos"], pos_dm])
                updated["dm"]["vel"] = np.vstack([updated["dm"]["vel"], vel_dm])
                updated["dm"]["mass"] = np.concatenate([updated["dm"]["mass"], mass_dm])
                region_label = f", r<={radius:.1f} kpc" if radius is not None else ""
                print(f"  Added uniform DM background (random): N={n_dm}, rho={rho_dm:.3e} Msun/kpc^3{region_label}")

        if rho_gas > 0:
            n_gas = int(round(rho_gas * target_volume / m_part)) # Uses m_part for random bg
            if n_gas > 0:
                if radius is None:
                    pos_gas = rng.uniform(-half_box, half_box, (n_gas, 3))
                else:
                    pos_gas = np.empty((0, 3), dtype=float)
                    while len(pos_gas) < n_gas:
                        trial_pos = rng.uniform(-radius, radius, (max(n_gas, 1024), 3))
                        accepted = _keep_in_background_region(trial_pos)
                        if accepted.size == 0:
                            continue
                        pos_gas = np.vstack([pos_gas, accepted])
                    pos_gas = pos_gas[:n_gas]
                vel_gas = np.zeros((n_gas, 3), dtype=float)
                mass_gas = np.full(n_gas, m_part)
                internal_energy_gas = np.full(n_gas, compute_internal_energy(T=1e4), dtype=float)

                updated["gas"]["pos"] = np.vstack([updated["gas"]["pos"], pos_gas])
                updated["gas"]["vel"] = np.vstack([updated["gas"]["vel"], vel_gas])
                updated["gas"]["mass"] = np.concatenate([updated["gas"]["mass"], mass_gas])
                updated["gas"]["internal_energy"] = np.concatenate([
                    updated["gas"]["internal_energy"],
                    internal_energy_gas,
                ])
                region_label = f", r<={radius:.1f} kpc" if radius is not None else ""
                print(f"  Added uniform gas background (random): N={n_gas}, rho={rho_gas:.3e} Msun/kpc^3{region_label}")

    else:
        # Add grid background particles (centered on origin)
        coords_1d = np.arange(-half_box, half_box, grid_spacing)
        if coords_1d.size > 0:
            gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
            grid_positions = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
            grid_positions = _keep_in_background_region(grid_positions)
            n_grid = len(grid_positions)

            # Add small random jitter
            jitter_gas = rng.normal(scale=0.1 * grid_spacing, size=grid_positions.shape)
            jitter_dm = rng.normal(scale=0.1 * grid_spacing, size=grid_positions.shape)
            gas_grid = _limit_to_background_region(grid_positions + jitter_gas)
            dm_grid = _limit_to_background_region(grid_positions + jitter_dm)

            # Calculate mass per particle to match target density
            if rho_gas > 0:
                m_gas_bg = (rho_gas * target_volume) / n_grid
                vel_gas = np.zeros((n_grid, 3), dtype=float)
                mass_gas = np.full(n_grid, m_gas_bg)
                internal_energy_gas = np.full(n_grid, compute_internal_energy(T=1e4), dtype=float)

                updated["gas"]["pos"] = np.vstack([updated["gas"]["pos"], gas_grid])
                updated["gas"]["vel"] = np.vstack([updated["gas"]["vel"], vel_gas])
                updated["gas"]["mass"] = np.concatenate([updated["gas"]["mass"], mass_gas])
                updated["gas"]["internal_energy"] = np.concatenate([
                    updated["gas"]["internal_energy"],
                    internal_energy_gas,
                ])
                region_label = f", r<={radius:.1f} kpc" if radius is not None else ""
                print(f"  Added grid gas background: spacing={grid_spacing} kpc, N={n_grid}, m={m_gas_bg:.2e} Msun{region_label}")

            if rho_dm > 0:
                m_dm_bg = (rho_dm * target_volume) / n_grid
                vel_dm = np.zeros((n_grid, 3), dtype=float)
                mass_dm = np.full(n_grid, m_dm_bg)

                updated["dm"]["pos"] = np.vstack([updated["dm"]["pos"], dm_grid])
                updated["dm"]["vel"] = np.vstack([updated["dm"]["vel"], vel_dm])
                updated["dm"]["mass"] = np.concatenate([updated["dm"]["mass"], mass_dm])
                region_label = f", r<={radius:.1f} kpc" if radius is not None else ""
                print(f"  Added grid DM background: spacing={grid_spacing} kpc, N={n_grid}, m={m_dm_bg:.2e} Msun{region_label}")

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


def _set_if_present(args: argparse.Namespace, attr: str, section: dict, key: str) -> None:
    if key in section:
        setattr(args, attr, section[key])


def _coerce_numeric_strings(value):
    if isinstance(value, dict):
        return {key: _coerce_numeric_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_numeric_strings(item) for item in value]
    if isinstance(value, str):
        try:
            if value.strip().isdigit():
                return int(value)
            return float(value)
        except ValueError:
            return value
    return value


def _collect_galaxy_values(galaxies: list[dict], path: tuple[str, ...]) -> list | None:
    values = []
    seen = False
    for galaxy in galaxies:
        section = galaxy
        for key in path[:-1]:
            section = section.get(key, {})
        if path[-1] in section:
            seen = True
            values.append(section[path[-1]])
        else:
            values.append(None)

    if not seen:
        return None
    if any(value is None for value in values):
        dotted_path = ".".join(path)
        raise ValueError(f"If galaxies specify {dotted_path}, every galaxy must specify it")
    return values


def _collect_optional_galaxy_values(galaxies: list[dict], path: tuple[str, ...]) -> list | None:
    values = []
    seen = False
    for galaxy in galaxies:
        section = galaxy
        for key in path[:-1]:
            section = section.get(key, {})
        value = section.get(path[-1])
        seen = seen or value is not None
        values.append(value)
    return values if seen else None


def _collect_galaxy_vectors(galaxies: list[dict], key: str) -> tuple[list, list, list] | None:
    vectors = _collect_galaxy_values(galaxies, ("placement", key))
    if vectors is None:
        return None
    for vector in vectors:
        if len(vector) != 3:
            raise ValueError(f"galaxies[].placement.{key} must have three values")
    return ([vector[0] for vector in vectors], [vector[1] for vector in vectors], [vector[2] for vector in vectors])


def _apply_config_file(args: argparse.Namespace, config_path: str) -> argparse.Namespace:
    with open(config_path) as handle:
        config = _coerce_numeric_strings(yaml.safe_load(handle) or {})

    if not isinstance(config, dict):
        raise ValueError("Generator config must be a YAML mapping")

    output = config.get("output", {})
    _set_if_present(args, "out_ics", output, "ics")
    _set_if_present(args, "out_params", output, "params")
    _set_if_present(args, "run_name", output, "run_name")
    _set_if_present(args, "snapshot_basename", output, "snapshot_basename")
    _set_if_present(args, "param_template", output, "param_template")

    simulation = config.get("simulation", {})
    _set_if_present(args, "box_kpc", simulation, "box_kpc")
    _set_if_present(args, "seed", simulation, "seed")
    _set_if_present(args, "max_timestep_gyr", simulation, "max_timestep_gyr")
    _set_if_present(args, "dt_min_gyr", simulation, "dt_min_gyr")
    _set_if_present(args, "time_end_gyr", simulation, "time_end_gyr")
    _set_if_present(args, "snapshot_dt_myr", simulation, "snapshot_dt_myr")
    _set_if_present(args, "feedback_scale", simulation, "feedback_scale")

    particle_masses = config.get("particle_masses", {})
    _set_if_present(args, "dm_part_mass_msun", particle_masses, "dm_msun")
    _set_if_present(args, "star_part_mass_msun", particle_masses, "stars_msun")
    _set_if_present(args, "gas_part_mass_msun", particle_masses, "gas_msun")

    orbit = config.get("orbit", {})
    _set_if_present(args, "orbit", orbit, "type")
    _set_if_present(args, "orbit_r_init_kpc", orbit, "r_init_kpc")
    _set_if_present(args, "orbit_r_peri_kpc", orbit, "r_peri_kpc")
    _set_if_present(args, "orbit_separation_kpc", orbit, "separation_kpc")
    _set_if_present(args, "orbit_radial_velocity_kms", orbit, "radial_velocity_kms")
    _set_if_present(args, "orbit_tangential_velocity_kms", orbit, "tangential_velocity_kms")
    _set_if_present(args, "orbit_plane_angle_deg", orbit, "plane_angle_deg")

    grid = config.get("grid", {})
    _set_if_present(args, "nR_grid", grid, "nR")
    _set_if_present(args, "nz_grid", grid, "nz")
    _set_if_present(args, "eps_grid", grid, "eps_kpc")
    _set_if_present(args, "h_max_cell_fraction", grid, "h_max_cell_fraction")
    _set_if_present(args, "scheduler_tasks_per_cell", grid, "scheduler_tasks_per_cell")
    _set_if_present(args, "max_top_level_cells", grid, "max_top_level_cells")

    background = config.get("background", {})
    _set_if_present(args, "bg_gas_density_msun_kpc3", background, "gas_density_msun_kpc3")
    _set_if_present(args, "bg_dm_density_msun_kpc3", background, "dm_density_msun_kpc3")
    _set_if_present(args, "bg_grid_kpc", background, "grid_kpc")
    _set_if_present(args, "bg_radius_kpc", background, "radius_kpc")

    galaxies = config.get("galaxies")
    if galaxies is None:
        return args
    if not isinstance(galaxies, list) or len(galaxies) == 0:
        raise ValueError("galaxies must be a non-empty YAML list")

    args.n_galaxies = len(galaxies)
    args.galaxy_names = [galaxy.get("name", f"galaxy_{i}") for i, galaxy in enumerate(galaxies)]
    galaxy_mappings = {
        "dm_mass_msun": ("masses", "dm_msun"),
        "star_mass_msun": ("masses", "stars_msun"),
        "gas_mass_msun": ("masses", "gas_msun"),
        "bulge_fraction": ("masses", "bulge_fraction"),
        "c200": ("halo", "c200"),
        "bulge_a_kpc": ("bulge", "a_kpc"),
        "bulge_rmax_scale": ("bulge", "rmax_scale"),
        "stellar_disk_scale_length_kpc": ("stellar_disk", "scale_length_kpc"),
        "stellar_disk_scale_height_kpc": ("stellar_disk", "scale_height_kpc"),
        "Q_star": ("stellar_disk", "Q"),
        "gas_disk_scale_length_kpc": ("gas_disk", "scale_length_kpc"),
        "gas_disk_scale_height_kpc": ("gas_disk", "scale_height_kpc"),
        "Q_gas": ("gas_disk", "Q"),
        "black_hole_mass_msun": ("black_hole", "mass_msun"),
        "n_arms": ("spiral", "n_arms"),
        "pitch_deg": ("spiral", "pitch_deg"),
        "arm_strength": ("spiral", "strength"),
        "arm_stream_frac": ("spiral", "stream_frac"),
        "bar_strength": ("bar", "strength"),
        "bar_radius": ("bar", "radius_kpc"),
        "bar_q": ("bar", "q"),
        "bar_angle": ("bar", "angle_deg"),
        "inclination_deg": ("placement", "inclination_deg"),
        "node_angle_deg": ("placement", "node_angle_deg"),
        "cgm_enabled": ("cgm", "enabled"),
        "cgm_mass_msun": ("cgm", "mass_msun"),
        "cgm_r_min_kpc": ("cgm", "r_min_kpc"),
        "cgm_r_max_kpc": ("cgm", "r_max_kpc"),
        "cgm_core_radius_kpc": ("cgm", "core_radius_kpc"),
        "cgm_beta": ("cgm", "beta"),
        "cgm_temperature_floor_K": ("cgm", "temperature_floor_K"),
        "cgm_temperature_ceiling_K": ("cgm", "temperature_ceiling_K"),
    }
    for attr, path in galaxy_mappings.items():
        if attr.startswith("cgm_"):
            values = _collect_optional_galaxy_values(galaxies, path)
            if values is not None:
                default_value = getattr(args, attr)[0]
                values = [default_value if value is None else value for value in values]
        else:
            values = _collect_galaxy_values(galaxies, path)
        if values is not None:
            setattr(args, attr, values)

    positions = _collect_galaxy_vectors(galaxies, "position_kpc")
    if positions is not None:
        args.xs, args.ys, args.zs = positions
    velocities = _collect_galaxy_vectors(galaxies, "velocity_kms")
    if velocities is not None:
        args.vxs, args.vys, args.vzs = velocities

    relative_to = _collect_optional_galaxy_values(galaxies, ("placement", "relative_to"))
    if relative_to is not None:
        args.relative_to = relative_to
    relative_positions = _collect_optional_galaxy_values(galaxies, ("placement", "relative_position_kpc"))
    if relative_positions is not None:
        for vector in relative_positions:
            if vector is not None and len(vector) != 3:
                raise ValueError("galaxies[].placement.relative_position_kpc must have three values")
        args.relative_position_kpc = relative_positions
    relative_velocities = _collect_optional_galaxy_values(galaxies, ("placement", "relative_velocity_kms"))
    if relative_velocities is not None:
        for vector in relative_velocities:
            if vector is not None and len(vector) != 3:
                raise ValueError("galaxies[].placement.relative_velocity_kms must have three values")
        args.relative_velocity_kms = relative_velocities

    if any(galaxy.get("bar", {}).get("enabled", False) for galaxy in galaxies):
        args.bar_enabled = True

    return args


def _default_generator_args() -> argparse.Namespace:
    return argparse.Namespace(
        out_ics="galaxy_ic.hdf5",
        out_params="galaxy_params.yml",
        box_kpc=100.0,
        dm_mass_msun=None,
        dm_part_mass_msun=None,
        bulge_fraction=None,
        star_mass_msun=None,
        star_part_mass_msun=None,
        gas_mass_msun=None,
        gas_part_mass_msun=None,
        n_galaxies=1,
        galaxy_names=["galaxy_0"],
        inclination_deg=None,
        node_angle_deg=None,
        relative_to=[None],
        relative_position_kpc=[None],
        relative_velocity_kms=[None],
        xs=None,
        ys=None,
        zs=None,
        vxs=None,
        vys=None,
        vzs=None,
        orbit="manual",
        orbit_r_init_kpc=None,
        orbit_r_peri_kpc=None,
        orbit_separation_kpc=None,
        orbit_radial_velocity_kms=None,
        orbit_tangential_velocity_kms=None,
        orbit_plane_angle_deg=0.0,
        c200=[10.0],
        bulge_a_kpc=[0.8],
        bulge_rmax_scale=[50.0],
        stellar_disk_scale_length_kpc=[3.5],
        stellar_disk_scale_height_kpc=[0.35],
        Q_star=[1.5],
        gas_disk_scale_length_kpc=[7.0],
        gas_disk_scale_height_kpc=[0.1],
        Q_gas=[1.0],
        black_hole_mass_msun=[0.0],
        cgm_enabled=[False],
        cgm_mass_msun=[0.0],
        cgm_r_min_kpc=[20.0],
        cgm_r_max_kpc=[250.0],
        cgm_core_radius_kpc=[3.0],
        cgm_beta=[0.5],
        cgm_temperature_floor_K=[1.0e5],
        cgm_temperature_ceiling_K=[3.0e6],
        n_arms=[2],
        pitch_deg=[15.0],
        arm_strength=[0.3],
        arm_stream_frac=[0.1],
        bar_enabled=False,
        bar_strength=[0.1],
        bar_radius=[3.0],
        bar_q=[0.3],
        bar_angle=[0.0],
        max_timestep_gyr=0.8,
        dt_min_gyr=1e-5,
        time_end_gyr=10.0,
        snapshot_dt_myr=10.0,
        feedback_scale=1.0,
        nR_grid=256,
        nz_grid=256,
        eps_grid=0.1,
        h_max_cell_fraction=0.5,
        scheduler_tasks_per_cell=100,
        max_top_level_cells=16,
        bg_gas_density_msun_kpc3=0.0,
        bg_dm_density_msun_kpc3=0.0,
        bg_grid_kpc=0.0,
        bg_radius_kpc=None,
        seed=42,
        run_name=None,
        param_template="eagle_ref_cosmo",
        snapshot_basename="snapshot",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate SWIFT ICs from a YAML config file.")
    parser.add_argument("config", type=str, help="Generator YAML config file.")

    cli_args = parser.parse_args()
    args = _apply_config_file(_default_generator_args(), cli_args.config)
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
        _place_galaxy(
            galaxy_data,
            galaxy_offsets[i],
            galaxy_bulk_velocities[i],
            args.inclination_deg[i],
            args.node_angle_deg[i],
        )

        all_galaxies_pos_mass.append(galaxy_data)

    # --- Combine all galaxies and add background ---
    initial_combined_data = {
        "dm": {"pos": np.vstack([g["dm"]["pos"] for g in all_galaxies_pos_mass]),
               "vel": np.vstack([g["dm"]["vel"] for g in all_galaxies_pos_mass]),
               "mass": np.concatenate([g["dm"]["mass"] for g in all_galaxies_pos_mass])},
        "gas": {"pos": np.vstack([g["gas"]["pos"] for g in all_galaxies_pos_mass]),
                "vel": np.vstack([g["gas"]["vel"] for g in all_galaxies_pos_mass]),
                "mass": np.concatenate([g["gas"]["mass"] for g in all_galaxies_pos_mass]),
                "internal_energy": np.concatenate([
                    g["gas"]["internal_energy"] for g in all_galaxies_pos_mass
                ])},
        "stars": {"pos": np.vstack([g["stars"]["pos"] for g in all_galaxies_pos_mass]),
                  "vel": np.vstack([g["stars"]["vel"] for g in all_galaxies_pos_mass]),
                  "mass": np.concatenate([g["stars"]["mass"] for g in all_galaxies_pos_mass])},
        "bulge": {"pos": np.vstack([g["bulge"]["pos"] for g in all_galaxies_pos_mass]),
                  "vel": np.vstack([g["bulge"]["vel"] for g in all_galaxies_pos_mass]),
                  "mass": np.concatenate([g["bulge"]["mass"] for g in all_galaxies_pos_mass])},
        "black_holes": {
            "pos": np.vstack([g["black_holes"]["pos"] for g in all_galaxies_pos_mass]),
            "vel": np.vstack([g["black_holes"]["vel"] for g in all_galaxies_pos_mass]),
            "mass": np.concatenate([g["black_holes"]["mass"] for g in all_galaxies_pos_mass]),
            "subgrid_mass": np.concatenate([
                g["black_holes"]["subgrid_mass"] for g in all_galaxies_pos_mass
            ]),
        },
    }

    # Add uniform background particles if specified
    if args.bg_gas_density_msun_kpc3 > 0 or args.bg_dm_density_msun_kpc3 > 0:
        component_particle_masses = []
        for galaxy_data in all_galaxies_pos_mass:
            for component in ("dm", "gas", "stars", "bulge"):
                masses = galaxy_data[component]["mass"]
                if masses.size > 0 and masses[0] > 0:
                    component_particle_masses.append(float(masses[0]))
        background_particle_mass = min(component_particle_masses) if component_particle_masses else 1e7
        initial_combined_data = add_uniform_background(
            initial_combined_data,
            args.box_kpc,
            background_particle_mass,
            args.bg_gas_density_msun_kpc3,
            args.bg_dm_density_msun_kpc3,
            args.bg_grid_kpc,
            args.bg_radius_kpc,
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
        max_top_level_cells=args.max_top_level_cells,
    )
    with open(args.out_params, "w") as f:
        f.write(params)
    print(f"Parameter file written to {args.out_params}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
