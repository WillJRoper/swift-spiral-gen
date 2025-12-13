"""Density profile functions for galaxy components."""

import numpy as np


def nfw_density(r: np.ndarray, m200: float, c200: float, delta_c: float, r_s: float) -> np.ndarray:
    """NFW halo density profile.

    Args:
        r: Radial positions (kpc).
        m200: M200 halo mass (Msun).
        c200: Concentration parameter.
        delta_c: Characteristic overdensity.
        r_s: Scale radius (kpc).

    Returns:
        Density at each position (Msun/kpc^3).
    """
    x = r / r_s
    rho_0 = delta_c * 200.0 * critical_density() / 3.0
    return rho_0 / (x * (1 + x) ** 2)


def nfw_mass(r: np.ndarray, m200: float, c200: float, r_s: float) -> np.ndarray:
    """Enclosed mass for NFW profile.

    Args:
        r: Radial positions (kpc).
        m200: M200 halo mass (Msun).
        c200: Concentration parameter.
        r_s: Scale radius (kpc).

    Returns:
        Enclosed mass at each radius (Msun).
    """
    x = r / r_s
    f_c = np.log(1 + c200) - c200 / (1 + c200)
    return m200 * (np.log(1 + x) - x / (1 + x)) / f_c


def hernquist_density(r: np.ndarray, m_bulge: float, a: float) -> np.ndarray:
    """Hernquist bulge density profile.

    Args:
        r: Radial positions (kpc).
        m_bulge: Total bulge mass (Msun).
        a: Hernquist scale length (kpc).

    Returns:
        Density at each position (Msun/kpc^3).
    """
    return m_bulge / (2 * np.pi) * a / (r * (r + a) ** 3)


def hernquist_mass(r: np.ndarray, m_bulge: float, a: float) -> np.ndarray:
    """Enclosed mass for Hernquist profile.

    Args:
        r: Radial positions (kpc).
        m_bulge: Total bulge mass (Msun).
        a: Hernquist scale length (kpc).

    Returns:
        Enclosed mass at each radius (Msun).
    """
    return m_bulge * r**2 / (r + a) ** 2


def exponential_surface_density(R: np.ndarray, sigma_0: float, R_d: float) -> np.ndarray:
    """Exponential disc surface density.

    Args:
        R: Cylindrical radial positions (kpc).
        sigma_0: Central surface density (Msun/kpc^2).
        R_d: Disc scale length (kpc).

    Returns:
        Surface density at each position (Msun/kpc^2).
    """
    return sigma_0 * np.exp(-R / R_d)


def exponential_disc_mass(R: np.ndarray, M_disc: float, R_d: float) -> np.ndarray:
    """Enclosed mass for exponential disc.

    Args:
        R: Cylindrical radial positions (kpc).
        M_disc: Total disc mass (Msun).
        R_d: Disc scale length (kpc).

    Returns:
        Enclosed mass at each radius (Msun).
    """
    x = R / R_d
    return M_disc * (1 - np.exp(-x) * (1 + x))


def sech2_vertical(z: np.ndarray, z_d: float) -> np.ndarray:
    """Vertical sech^2 profile for disc.

    Args:
        z: Vertical positions (kpc).
        z_d: Disc scale height (kpc).

    Returns:
        Vertical density factor (normalized to integrate to 1).
    """
    return np.cosh(z / (2 * z_d)) ** (-2) / (4 * z_d)


def critical_density() -> float:
    """Critical density of the universe at z=0.

    Returns:
        Critical density in Msun/kpc^3.
    """
    # Using H0 = 70 km/s/Mpc, rho_crit = 3 H0^2 / (8 pi G)
    # Result in Msun/kpc^3
    return 277.5  # Approximate value


def nfw_params(m200: float, c200: float) -> tuple[float, float]:
    """Calculate NFW profile parameters.

    Args:
        m200: M200 halo mass (Msun).
        c200: Concentration parameter.

    Returns:
        Tuple of (r_s, delta_c) - scale radius (kpc) and characteristic overdensity.
    """
    # r200 from M200
    rho_c = critical_density()
    r200 = (3 * m200 / (4 * np.pi * 200 * rho_c)) ** (1.0 / 3.0)
    r_s = r200 / c200

    # Characteristic overdensity
    f_c = np.log(1 + c200) - c200 / (1 + c200)
    delta_c = (200.0 / 3.0) * c200**3 / f_c

    return r_s, delta_c


def disc_sigma_0(M_disc: float, R_d: float) -> float:
    """Calculate central surface density for exponential disc.

    Args:
        M_disc: Total disc mass (Msun).
        R_d: Disc scale length (kpc).

    Returns:
        Central surface density (Msun/kpc^2).
    """
    return M_disc / (2 * np.pi * R_d**2)
