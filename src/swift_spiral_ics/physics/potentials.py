"""Gravitational potential functions for galaxy components."""

import numpy as np


def nfw_potential(R: np.ndarray, z: np.ndarray, m200: float, r_s: float, c200: float) -> np.ndarray:
    """NFW halo gravitational potential (approximate for disc plane).

    Args:
        R: Cylindrical radial positions (kpc).
        z: Vertical positions (kpc).
        m200: M200 halo mass (Msun).
        r_s: Scale radius (kpc).
        c200: Concentration parameter.

    Returns:
        Potential at each position (km^2/s^2).
    """
    G = 4.302e-6  # kpc (km/s)^2 / Msun
    r = np.sqrt(R**2 + z**2)
    x = r / r_s
    f_c = np.log(1 + c200) - c200 / (1 + c200)
    return -G * m200 / f_c * np.log(1 + x) / r


def hernquist_potential(R: np.ndarray, z: np.ndarray, m_bulge: float, a: float) -> np.ndarray:
    """Hernquist bulge gravitational potential.

    Args:
        R: Cylindrical radial positions (kpc).
        z: Vertical positions (kpc).
        m_bulge: Total bulge mass (Msun).
        a: Hernquist scale length (kpc).

    Returns:
        Potential at each position (km^2/s^2).
    """
    G = 4.302e-6  # kpc (km/s)^2 / Msun
    r = np.sqrt(R**2 + z**2)
    return -G * m_bulge / (r + a)


def miyamoto_nagai_potential(
    R: np.ndarray, z: np.ndarray, M_disc: float, R_d: float, z_d: float
) -> np.ndarray:
    """Miyamoto-Nagai disc gravitational potential.

    Approximation for exponential disc with finite thickness.

    Args:
        R: Cylindrical radial positions (kpc).
        z: Vertical positions (kpc).
        M_disc: Total disc mass (Msun).
        R_d: Disc scale length (kpc).
        z_d: Disc scale height (kpc).

    Returns:
        Potential at each position (km^2/s^2).
    """
    G = 4.302e-6  # kpc (km/s)^2 / Msun
    # Use a = R_d, b = z_d for Miyamoto-Nagai parameters
    a = R_d
    b = z_d
    return -G * M_disc / np.sqrt(R**2 + (a + np.sqrt(z**2 + b**2)) ** 2)


def total_circular_velocity(
    R: np.ndarray,
    m200: float,
    c200: float,
    m_bulge: float,
    a_bulge: float,
    M_disc_star: float,
    R_d_star: float,
    z_d_star: float,
    M_disc_gas: float,
    R_d_gas: float,
    z_d_gas: float,
) -> np.ndarray:
    """Compute total circular velocity from all components.

    Args:
        R: Cylindrical radial positions (kpc).
        m200: Halo M200 mass (Msun).
        c200: Halo concentration.
        m_bulge: Bulge mass (Msun).
        a_bulge: Bulge scale length (kpc).
        M_disc_star: Stellar disc mass (Msun).
        R_d_star: Stellar disc scale length (kpc).
        z_d_star: Stellar disc scale height (kpc).
        M_disc_gas: Gas disc mass (Msun).
        R_d_gas: Gas disc scale length (kpc).
        z_d_gas: Gas disc scale height (kpc).

    Returns:
        Circular velocity at each radius (km/s).
    """
    from .profiles import nfw_params

    # Get NFW parameters
    r_s, _ = nfw_params(m200, c200)

    # Compute potential derivatives at z=0
    dR = 0.01  # kpc, small step for numerical derivative
    R_plus = R + dR
    R_minus = np.maximum(R - dR, 1e-3)

    # NFW contribution
    psi_nfw_plus = nfw_potential(R_plus, 0.0, m200, r_s, c200)
    psi_nfw_minus = nfw_potential(R_minus, 0.0, m200, r_s, c200)
    dpsi_nfw = (psi_nfw_plus - psi_nfw_minus) / (R_plus - R_minus)

    # Bulge contribution
    if m_bulge > 0:
        psi_bulge_plus = hernquist_potential(R_plus, 0.0, m_bulge, a_bulge)
        psi_bulge_minus = hernquist_potential(R_minus, 0.0, m_bulge, a_bulge)
        dpsi_bulge = (psi_bulge_plus - psi_bulge_minus) / (R_plus - R_minus)
    else:
        dpsi_bulge = 0.0

    # Stellar disc contribution
    if M_disc_star > 0:
        psi_star_plus = miyamoto_nagai_potential(R_plus, 0.0, M_disc_star, R_d_star, z_d_star)
        psi_star_minus = miyamoto_nagai_potential(R_minus, 0.0, M_disc_star, R_d_star, z_d_star)
        dpsi_star = (psi_star_plus - psi_star_minus) / (R_plus - R_minus)
    else:
        dpsi_star = 0.0

    # Gas disc contribution
    if M_disc_gas > 0:
        psi_gas_plus = miyamoto_nagai_potential(R_plus, 0.0, M_disc_gas, R_d_gas, z_d_gas)
        psi_gas_minus = miyamoto_nagai_potential(R_minus, 0.0, M_disc_gas, R_d_gas, z_d_gas)
        dpsi_gas = (psi_gas_plus - psi_gas_minus) / (R_plus - R_minus)
    else:
        dpsi_gas = 0.0

    # Total derivative
    dpsi_total = dpsi_nfw + dpsi_bulge + dpsi_star + dpsi_gas

    # v_c^2 = R * dPsi/dR
    v_c_squared = R * dpsi_total
    v_c_squared = np.maximum(v_c_squared, 0.0)  # Ensure non-negative

    return np.sqrt(v_c_squared)
