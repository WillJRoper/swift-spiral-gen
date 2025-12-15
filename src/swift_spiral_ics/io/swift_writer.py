"""SWIFT HDF5 initial conditions writer."""

from __future__ import annotations

import h5py
import numpy as np
import unyt


def write_swift_ic(
    filename: str,
    box_size: float,
    particle_data: dict[str, dict[str, np.ndarray]],
) -> None:
    # Convert to cosmological SWIFT units (Mpc, 1e10 Msun)
    length_conv = 1.0 / 1000.0  # kpc -> Mpc
    mass_conv = 1.0 / 1e10  # Msun -> 1e10 Msun

    with h5py.File(filename, "w") as f:
        # Particle counts
        N_dm = len(particle_data.get("dm", {}).get("pos", []))
        N_gas = len(particle_data.get("gas", {}).get("pos", []))
        N_stars = len(particle_data.get("stars", {}).get("pos", []))

        # Small jitter helper to avoid exact duplicates at cell boundaries
        rng_jitter = np.random.default_rng(42)

        # Header
        header = f.create_group("Header")
        header.attrs["Dimension"] = 3
        header.attrs["BoxSize"] = box_size * length_conv
        header.attrs["NumPart_Total"] = np.array([N_gas, N_dm, 0, 0, N_stars, 0], dtype=np.uint32)
        header.attrs["NumPart_Total_HighWord"] = np.array([0, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs["NumPart_ThisFile"] = np.array(
            [N_gas, N_dm, 0, 0, N_stars, 0], dtype=np.uint32
        )
        header.attrs["MassTable"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        header.attrs["Flag_Entropy_ICs"] = 0
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Time"] = 0.0
        # Explicitly mark as non-cosmological
        header.attrs["HubbleParam"] = 0.0
        header.attrs["Omega0"] = 0.0
        header.attrs["OmegaLambda"] = 0.0

        # Units group
        units = f.create_group("Units")
        units.attrs["Unit length in cgs (U_L)"] = 3.08567758e24  # Mpc
        units.attrs["Unit mass in cgs (U_M)"] = 1.98841e43  # 1e10 Msun in grams
        units.attrs["Unit time in cgs (U_t)"] = 3.085678e19  # s so that 1 velocity unit = 1 km/s
        units.attrs["Unit current in cgs (U_I)"] = 1.0
        units.attrs["Unit temperature in cgs (U_T)"] = 1.0
        # Particle ID counter
        current_id = 1

        # Write gas particles (PartType0)
        if N_gas > 0:
            gas_data = particle_data["gas"]
            gas_group = f.create_group("PartType0")

            pos = np.mod(gas_data["pos"], box_size)
            pos = _jitter_duplicates(pos, rng_jitter)
            vel = gas_data["vel"]
            mass = gas_data["mass"]

            # Wrap positions into box
            pos_wrapped = np.mod(pos, box_size)
            gas_group.create_dataset(
                "Coordinates", data=(pos_wrapped * length_conv).astype(np.float64)
            )
            gas_group.create_dataset("Velocities", data=vel.astype(np.float32))
            gas_group.create_dataset(
                "Masses", data=(mass * mass_conv).astype(np.float32)
            )
            gas_group.create_dataset(
                "ParticleIDs", data=np.arange(current_id, current_id + N_gas, dtype=np.uint64)
            )
            current_id += N_gas

            # Internal energy and smoothing length
            u_gas = compute_internal_energy(T=1e4)  # 10^4 K
            gas_group.create_dataset("InternalEnergy", data=np.full(N_gas, u_gas, dtype=np.float32))

            # Estimate smoothing length
            h = estimate_smoothing_length(pos_wrapped, N_gas, box_size)
            gas_group.create_dataset("SmoothingLength", data=(h * length_conv).astype(np.float32))

        # Write DM particles (PartType1)
        if N_dm > 0:
            dm_data = particle_data["dm"]
            dm_group = f.create_group("PartType1")

            pos = np.mod(dm_data["pos"], box_size)
            pos = _jitter_duplicates(pos, rng_jitter)
            vel = dm_data["vel"]
            mass = dm_data["mass"]

            # Wrap positions into box
            pos_wrapped = np.mod(pos, box_size)
            dm_group.create_dataset(
                "Coordinates", data=(pos_wrapped * length_conv).astype(np.float64)
            )
            dm_group.create_dataset("Velocities", data=vel.astype(np.float32))
            dm_group.create_dataset(
                "Masses", data=(mass * mass_conv).astype(np.float32)
            )
            dm_group.create_dataset(
                "ParticleIDs", data=np.arange(current_id, current_id + N_dm, dtype=np.uint64)
            )
            current_id += N_dm

        # Write star particles (PartType4)
        if N_stars > 0:
            star_data = particle_data["stars"]
            star_group = f.create_group("PartType4")

            pos = np.mod(star_data["pos"], box_size)
            pos = _jitter_duplicates(pos, rng_jitter)
            vel = star_data["vel"]
            mass = star_data["mass"]

            # Wrap positions into box
            pos_wrapped = np.mod(pos, box_size)
            star_group.create_dataset(
                "Coordinates", data=(pos_wrapped * length_conv).astype(np.float64)
            )
            star_group.create_dataset("Velocities", data=vel.astype(np.float32))
            star_group.create_dataset(
                "Masses", data=(mass * mass_conv).astype(np.float32)
            )
            star_group.create_dataset(
                "ParticleIDs", data=np.arange(current_id, current_id + N_stars, dtype=np.uint64)
            )
            current_id += N_stars

            # Provide a finite smoothing length; tie to gas distribution if present
            if N_gas > 0:
                gas_pos_wrapped = np.mod(particle_data["gas"]["pos"], box_size)
                h_star = estimate_smoothing_length(gas_pos_wrapped, N_gas, box_size)
                # Match lengths to nearest gas smoothing via simple nearest-neighbour assignment
                from scipy.spatial import cKDTree

                gas_tree = cKDTree(gas_pos_wrapped)
                _, gas_idx = gas_tree.query(pos_wrapped, k=1)
                h_star = h_star[gas_idx]
            else:
                h_star = estimate_smoothing_length(pos_wrapped, N_stars, box_size)

            star_group.create_dataset(
                "SmoothingLength", data=(h_star * length_conv).astype(np.float32)
            )


def compute_internal_energy(T: float) -> float:
    """Compute specific internal energy from temperature.

    Args:
        T: Temperature (K).

    Returns:
        Specific internal energy (erg/g) in code units.
    """

    # u = k_B * T / ((gamma - 1) * mu * m_p)
    gamma = 5.0 / 3.0
    mu = 0.6

    T_unyt = T * unyt.K
    u = unyt.kb * T_unyt / ((gamma - 1) * mu * unyt.mp)

    # Convert to code units: (km/s)^2
    # 1 erg/g = 1e-10 (km/s)^2
    u_code = u.to((unyt.km / unyt.s) ** 2).value

    return u_code


def estimate_smoothing_length(
    pos: np.ndarray,
    N: int,
    box_size: float,
    n_ngb: int = 58,
) -> np.ndarray:
    """Estimate smoothing lengths using swiftsimio."""
    if N == 0:
        return np.array([])

    return _smoothing_length_swiftsimio(pos, box_size, n_ngb)


def _smoothing_length_swiftsimio(pos: np.ndarray, box_size: float, n_ngb: int) -> np.ndarray:
    """Compute smoothing lengths via swiftsimio."""
    from swiftsimio.visualisation.smoothing_length.generate import (
        generate_smoothing_lengths as swift_sl,
    )

    coords = unyt.unyt_array(pos, units="kpc")
    box = unyt.unyt_array([box_size, box_size, box_size], units="kpc")
    # Use a fixed gamma appropriate for Wendland-C2 / cubic spline kernels
    kernel_gamma = np.float32(1.2348)
    h = swift_sl(
        coords,
        boxsize=box,
        kernel_gamma=kernel_gamma,
        neighbours=n_ngb,
        speedup_fac=1,
    )
    return np.asarray(h.value, dtype=np.float64)


def _jitter_duplicates(pos: np.ndarray, rng: np.random.Generator, eps: float = 1e-4) -> np.ndarray:
    """Ensure no exact duplicate positions by adding tiny random jitter."""
    out = pos.copy()
    for _ in range(3):
        # Find duplicates via lexsort on rounded coordinates
        rounded = np.round(out / eps).astype(np.int64)
        _, idx, counts = np.unique(rounded, axis=0, return_index=True, return_counts=True)
        dup_mask = counts > 1
        if not np.any(dup_mask):
            break
        dup_indices = idx[dup_mask]
        # Jitter all entries matching duplicate keys
        dup_keys = rounded[dup_indices]
        key_set = {tuple(k) for k in dup_keys}
        for i, key in enumerate(map(tuple, rounded)):
            if key in key_set:
                out[i] += rng.normal(scale=eps, size=3)
    return out


def print_ic_summary(
    particle_data: dict[str, dict[str, np.ndarray]],
    requested_masses: dict[str, float],
    box_size: float,
) -> None:
    """Print summary of initial conditions.

    Args:
        particle_data: Particle data dict.
        requested_masses: Dict of requested masses for each component.
        box_size: Box size (kpc).
    """
    print("\n" + "=" * 70)
    print("SWIFT INITIAL CONDITIONS SUMMARY")
    print("=" * 70)

    print(f"\nBox size: {box_size:.2f} kpc")

    print("\nParticle counts and masses:")
    print("-" * 70)
    print(f"{'Component':<15} {'N_particles':<15} {'Requested (Msun)':<20} {'Achieved (Msun)':<20}")
    print("-" * 70)

    total_particles = 0

    for component in ["dm", "gas", "stars"]:
        if component in particle_data and len(particle_data[component].get("pos", [])) > 0:
            N = len(particle_data[component]["pos"])
            M_achieved = np.sum(particle_data[component]["mass"])
            M_requested = requested_masses.get(component, 0.0)
            total_particles += N

            print(f"{component:<15} {N:<15} {M_requested:<20.3e} {M_achieved:<20.3e}")

    print("-" * 70)
    print(f"{'TOTAL':<15} {total_particles:<15}")
    print("=" * 70 + "\n")
