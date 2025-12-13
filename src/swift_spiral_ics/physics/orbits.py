"""Orbital dynamics for galaxy mergers."""

import numpy as np


def parabolic_orbit_initial_conditions(
    M_primary: float,
    M_secondary: float,
    r_init: float,
    r_peri: float,
    orbit_plane_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate initial position and velocity for parabolic orbit.

    Uses the "head-onness" parameter r_peri to control the orbital
    angular momentum. r_peri = 0 gives head-on collision, larger
    values give more grazing/orbital encounters.

    Args:
        M_primary: Primary galaxy total mass (Msun).
        M_secondary: Secondary galaxy total mass (Msun).
        r_init: Initial separation (kpc).
        r_peri: Pericentre distance (kpc) - the "head-onness" knob.
        orbit_plane_angle: Angle of orbit plane normal from z-axis (degrees).

    Returns:
        Tuple of (pos, vel) where pos is 3D position (kpc) and vel is 3D velocity (km/s).
    """
    G = 4.302e-6  # kpc (km/s)^2 / Msun
    M_total = M_primary + M_secondary

    # For parabolic orbit, energy E = 0
    # At pericentre: E = 0.5 * v_peri^2 - G*M/r_peri = 0
    # v_peri = sqrt(2*G*M/r_peri)

    # Angular momentum L = r_peri * v_peri = r_peri * sqrt(2*G*M/r_peri) = sqrt(2*G*M*r_peri)
    L = np.sqrt(2 * G * M_total * r_peri)

    # At initial position r_init, use energy and angular momentum conservation
    # E = 0.5 * v^2 - G*M/r = 0
    # v^2 = 2*G*M/r
    # Also, v = sqrt(v_r^2 + v_t^2), where v_t = L/r

    v_t = L / r_init  # Tangential velocity
    v_total_sq = 2 * G * M_total / r_init
    v_r_sq = v_total_sq - v_t**2

    if v_r_sq < 0:
        # This means r_init < r_peri, which is unphysical
        raise ValueError(f"r_init ({r_init:.2f}) must be >= r_peri ({r_peri:.2f})")

    v_r = -np.sqrt(v_r_sq)  # Negative because moving inward

    # Set up orbit in x-y plane initially, then rotate
    # Position: at (r_init, 0, 0)
    # Velocity: radial + tangential components

    # Start with position along x-axis
    pos_orbit = np.array([r_init, 0.0, 0.0])

    # Velocity: radial along x-direction, tangential along y-direction
    vel_orbit = np.array([v_r, v_t, 0.0])

    # Rotate orbit plane if requested
    if orbit_plane_angle != 0:
        angle_rad = np.deg2rad(orbit_plane_angle)
        # Rotate around y-axis
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rot_matrix = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        pos_orbit = rot_matrix @ pos_orbit
        vel_orbit = rot_matrix @ vel_orbit

    return pos_orbit, vel_orbit


def rotate_disc(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    inclination: float,
    node_angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate disc to specified orientation.

    Args:
        x, y, z: Positions (kpc).
        vx, vy, vz: Velocities (km/s).
        inclination: Inclination angle from face-on (degrees).
        node_angle: Longitude of ascending node (degrees).

    Returns:
        Tuple of (x, y, z, vx, vy, vz) rotated positions and velocities.
    """
    inc_rad = np.deg2rad(inclination)
    node_rad = np.deg2rad(node_angle)

    # Rotation matrices
    # First rotate around z-axis by node_angle
    cos_node = np.cos(node_rad)
    sin_node = np.sin(node_rad)
    R_node = np.array([[cos_node, -sin_node, 0], [sin_node, cos_node, 0], [0, 0, 1]])

    # Then rotate around new x-axis by inclination
    cos_inc = np.cos(inc_rad)
    sin_inc = np.sin(inc_rad)
    R_inc = np.array([[1, 0, 0], [0, cos_inc, -sin_inc], [0, sin_inc, cos_inc]])

    # Combined rotation
    R_total = R_inc @ R_node

    # Apply to positions
    pos = np.array([x, y, z])
    pos_rot = R_total @ pos
    x_rot, y_rot, z_rot = pos_rot

    # Apply to velocities
    vel = np.array([vx, vy, vz])
    vel_rot = R_total @ vel
    vx_rot, vy_rot, vz_rot = vel_rot

    return x_rot, y_rot, z_rot, vx_rot, vy_rot, vz_rot


def place_galaxy_in_orbit(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    orbit_pos: np.ndarray,
    orbit_vel: np.ndarray,
    inclination: float = 0.0,
    node_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Place galaxy particles in orbital configuration.

    Args:
        x, y, z: Galaxy-frame positions (kpc).
        vx, vy, vz: Galaxy-frame velocities (km/s).
        orbit_pos: Orbital position of galaxy COM (kpc).
        orbit_vel: Orbital velocity of galaxy COM (km/s).
        inclination: Disc inclination (degrees).
        node_angle: Longitude of ascending node (degrees).

    Returns:
        Tuple of (x, y, z, vx, vy, vz) in simulation frame.
    """
    # First rotate the disc to desired orientation
    x_rot, y_rot, z_rot, vx_rot, vy_rot, vz_rot = rotate_disc(
        x, y, z, vx, vy, vz, inclination, node_angle
    )

    # Then translate to orbital position
    x_final = x_rot + orbit_pos[0]
    y_final = y_rot + orbit_pos[1]
    z_final = z_rot + orbit_pos[2]

    # Add orbital velocity
    vx_final = vx_rot + orbit_vel[0]
    vy_final = vy_rot + orbit_vel[1]
    vz_final = vz_rot + orbit_vel[2]

    return x_final, y_final, z_final, vx_final, vy_final, vz_final


def center_of_mass_correction(
    masses: list[np.ndarray],
    positions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    velocities: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray, np.ndarray]]
]:
    """Correct positions and velocities to center-of-mass frame.

    Args:
        masses: List of mass arrays for each component.
        positions: List of (x, y, z) tuples for each component.
        velocities: List of (vx, vy, vz) tuples for each component.

    Returns:
        Tuple of (positions_corrected, velocities_corrected) lists.
    """
    # Calculate total mass and COM
    M_total = sum(np.sum(m) for m in masses)

    com_pos = np.zeros(3)
    com_vel = np.zeros(3)

    for m, (x, y, z), (vx, vy, vz) in zip(masses, positions, velocities):
        com_pos[0] += np.sum(m * x)
        com_pos[1] += np.sum(m * y)
        com_pos[2] += np.sum(m * z)
        com_vel[0] += np.sum(m * vx)
        com_vel[1] += np.sum(m * vy)
        com_vel[2] += np.sum(m * vz)

    com_pos /= M_total
    com_vel /= M_total

    # Subtract COM
    positions_corrected = []
    velocities_corrected = []

    for (x, y, z), (vx, vy, vz) in zip(positions, velocities):
        positions_corrected.append((x - com_pos[0], y - com_pos[1], z - com_pos[2]))
        velocities_corrected.append((vx - com_vel[0], vy - com_vel[1], vz - com_vel[2]))

    return positions_corrected, velocities_corrected
