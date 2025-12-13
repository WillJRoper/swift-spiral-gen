"""SWIFT HDF5 initial conditions writer."""

import h5py
import numpy as np


def write_swift_ic(
    filename: str,
    box_size: float,
    particle_data: dict[str, dict[str, np.ndarray]],
    m_part: float,
) -> None:
    """Write SWIFT-compatible HDF5 initial conditions file.

    Args:
        filename: Output HDF5 filename.
        box_size: Simulation box size (kpc).
        particle_data: Dict with keys 'dm', 'gas', 'stars', each containing
                      'pos' (N,3), 'vel' (N,3) arrays.
        m_part: Particle mass (Msun).
    """
    with h5py.File(filename, "w") as f:
        # Particle counts
        N_dm = len(particle_data.get("dm", {}).get("pos", []))
        N_gas = len(particle_data.get("gas", {}).get("pos", []))
        N_stars = len(particle_data.get("stars", {}).get("pos", []))

        # Header
        header = f.create_group("Header")
        header.attrs["Dimension"] = 3
        header.attrs["BoxSize"] = box_size
        header.attrs["NumPart_Total"] = np.array([N_gas, N_dm, 0, 0, N_stars, 0], dtype=np.uint32)
        header.attrs["NumPart_Total_HighWord"] = np.array([0, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs["NumPart_ThisFile"] = np.array(
            [N_gas, N_dm, 0, 0, N_stars, 0], dtype=np.uint32
        )
        header.attrs["MassTable"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        header.attrs["Flag_Entropy_ICs"] = 0
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Time"] = 0.0

        # Units group
        units = f.create_group("Units")
        units.attrs["Unit length in cgs (U_L)"] = 3.085678e21  # kpc
        units.attrs["Unit mass in cgs (U_M)"] = 1.98841e43  # Msun
        units.attrs["Unit time in cgs (U_t)"] = 3.08568e16  # Myr (roughly)
        units.attrs["Unit current in cgs (U_I)"] = 1.0
        units.attrs["Unit temperature in cgs (U_T)"] = 1.0

        # Particle ID counter
        current_id = 1

        # Write gas particles (PartType0)
        if N_gas > 0:
            gas_data = particle_data["gas"]
            gas_group = f.create_group("PartType0")

            pos = gas_data["pos"]
            vel = gas_data["vel"]

            # Wrap positions into box
            pos_wrapped = np.mod(pos, box_size)

            gas_group.create_dataset("Coordinates", data=pos_wrapped.astype(np.float64))
            gas_group.create_dataset("Velocities", data=vel.astype(np.float32))
            gas_group.create_dataset("Masses", data=np.full(N_gas, m_part, dtype=np.float32))
            gas_group.create_dataset(
                "ParticleIDs", data=np.arange(current_id, current_id + N_gas, dtype=np.uint64)
            )
            current_id += N_gas

            # Internal energy and smoothing length
            u_gas = compute_internal_energy(T=1e4)  # 10^4 K
            gas_group.create_dataset("InternalEnergy", data=np.full(N_gas, u_gas, dtype=np.float32))

            # Estimate smoothing length
            h = estimate_smoothing_length(pos_wrapped, N_gas, box_size)
            gas_group.create_dataset("SmoothingLength", data=h.astype(np.float32))

        # Write DM particles (PartType1)
        if N_dm > 0:
            dm_data = particle_data["dm"]
            dm_group = f.create_group("PartType1")

            pos = dm_data["pos"]
            vel = dm_data["vel"]

            # Wrap positions into box
            pos_wrapped = np.mod(pos, box_size)

            dm_group.create_dataset("Coordinates", data=pos_wrapped.astype(np.float64))
            dm_group.create_dataset("Velocities", data=vel.astype(np.float32))
            dm_group.create_dataset("Masses", data=np.full(N_dm, m_part, dtype=np.float32))
            dm_group.create_dataset(
                "ParticleIDs", data=np.arange(current_id, current_id + N_dm, dtype=np.uint64)
            )
            current_id += N_dm

        # Write star particles (PartType4)
        if N_stars > 0:
            star_data = particle_data["stars"]
            star_group = f.create_group("PartType4")

            pos = star_data["pos"]
            vel = star_data["vel"]

            # Wrap positions into box
            pos_wrapped = np.mod(pos, box_size)

            star_group.create_dataset("Coordinates", data=pos_wrapped.astype(np.float64))
            star_group.create_dataset("Velocities", data=vel.astype(np.float32))
            star_group.create_dataset("Masses", data=np.full(N_stars, m_part, dtype=np.float32))
            star_group.create_dataset(
                "ParticleIDs", data=np.arange(current_id, current_id + N_stars, dtype=np.uint64)
            )
            current_id += N_stars


def compute_internal_energy(T: float) -> float:
    """Compute specific internal energy from temperature.

    Args:
        T: Temperature (K).

    Returns:
        Specific internal energy (erg/g) in code units.
    """
    k_B = 1.38e-16  # erg/K
    m_p = 1.673e-24  # g
    mu = 0.6  # Mean molecular weight
    gamma = 5.0 / 3.0  # Adiabatic index

    # u = k_B * T / ((gamma - 1) * mu * m_p)
    u = k_B * T / ((gamma - 1) * mu * m_p)

    # Convert to code units (km/s)^2
    u_code = u * 1e-10  # erg/g to (km/s)^2

    return u_code


def estimate_smoothing_length(
    pos: np.ndarray,
    N: int,
    box_size: float,
    n_ngb: int = 58,
) -> np.ndarray:
    """Estimate smoothing lengths for gas particles.

    Uses simple density-based estimate.

    Args:
        pos: Particle positions (N, 3) in kpc.
        N: Number of particles.
        box_size: Box size (kpc).
        n_ngb: Target number of neighbors.

    Returns:
        Smoothing lengths (kpc).
    """
    # Simple estimate: assume uniform distribution
    # Volume per particle = box_size^3 / N
    # Sphere volume with n_ngb neighbors: 4/3 * pi * h^3 * n_density = n_ngb
    # n_density = N / box_size^3
    # h = (3 * n_ngb / (4 * pi * n_density))^(1/3)

    n_density = N / box_size**3
    h_avg = (3 * n_ngb / (4 * np.pi * n_density)) ** (1.0 / 3.0)

    # Return constant smoothing length (could be refined with tree search)
    return np.full(N, h_avg)


def print_ic_summary(
    particle_data: dict[str, dict[str, np.ndarray]],
    m_part: float,
    requested_masses: dict[str, float],
    box_size: float,
) -> None:
    """Print summary of initial conditions.

    Args:
        particle_data: Particle data dict.
        m_part: Particle mass (Msun).
        requested_masses: Dict of requested masses for each component.
        box_size: Box size (kpc).
    """
    print("\n" + "=" * 70)
    print("SWIFT INITIAL CONDITIONS SUMMARY")
    print("=" * 70)

    print(f"\nBox size: {box_size:.2f} kpc")
    print(f"Particle mass: {m_part:.2e} Msun")

    print("\nParticle counts and masses:")
    print("-" * 70)
    print(f"{'Component':<15} {'N_particles':<15} {'Requested (Msun)':<20} {'Achieved (Msun)':<20}")
    print("-" * 70)

    total_particles = 0

    for component in ["dm", "gas", "stars"]:
        if component in particle_data and len(particle_data[component].get("pos", [])) > 0:
            N = len(particle_data[component]["pos"])
            M_achieved = N * m_part
            M_requested = requested_masses.get(component, 0.0)
            total_particles += N

            print(f"{component:<15} {N:<15} {M_requested:<20.3e} {M_achieved:<20.3e}")

    print("-" * 70)
    print(f"{'TOTAL':<15} {total_particles:<15}")
    print("=" * 70 + "\n")
