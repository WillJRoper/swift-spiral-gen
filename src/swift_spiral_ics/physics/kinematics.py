"""Kinematic calculations for galaxy components."""

import numpy as np


def epicyclic_frequency(R: np.ndarray, v_c: np.ndarray) -> np.ndarray:
    """Calculate epicyclic frequency from rotation curve.

    Args:
        R: Cylindrical radial positions (kpc).
        v_c: Circular velocity at each radius (km/s).

    Returns:
        Epicyclic frequency kappa at each radius (km/s/kpc).
    """
    # Numerical derivative of v_c using gradient
    dv_dR = np.gradient(v_c, R, edge_order=2)

    # kappa^2 = (2*v_c/R)^2 + 2*v_c/R * dv_c/dR
    kappa_sq = (2 * v_c / R) ** 2 + 2 * v_c / R * dv_dR
    kappa_sq = np.maximum(kappa_sq, 0.0)

    return np.sqrt(kappa_sq)


def toomre_q_dispersion(
    R: np.ndarray, v_c: np.ndarray, sigma_surf: np.ndarray, Q_target: float
) -> np.ndarray:
    """Calculate radial velocity dispersion from Toomre Q.

    Args:
        R: Cylindrical radial positions (kpc).
        v_c: Circular velocity at each radius (km/s).
        sigma_surf: Surface density at each radius (Msun/kpc^2).
        Q_target: Target Toomre Q parameter.

    Returns:
        Radial velocity dispersion sigma_R (km/s).
    """
    G = 4.302e-6  # kpc (km/s)^2 / Msun
    kappa = epicyclic_frequency(R, v_c)

    # Q = sigma_R * kappa / (pi * G * Sigma)
    # sigma_R = Q * pi * G * Sigma / kappa
    sigma_R = Q_target * np.pi * G * sigma_surf / kappa

    # Floor to avoid instabilities
    sigma_R = np.maximum(sigma_R, 5.0)  # Minimum 5 km/s

    return sigma_R


def asymmetric_drift_correction(
    R: np.ndarray,
    v_c: np.ndarray,
    sigma_R: np.ndarray,
    sigma_surf: np.ndarray,
    R_d: float,
) -> np.ndarray:
    """Calculate asymmetric drift correction to mean azimuthal velocity.

    Args:
        R: Cylindrical radial positions (kpc).
        v_c: Circular velocity at each radius (km/s).
        sigma_R: Radial velocity dispersion (km/s).
        sigma_surf: Surface density at each radius (Msun/kpc^2).
        R_d: Disc scale length (kpc).

    Returns:
        Mean azimuthal velocity v_phi (km/s).
    """
    # Asymmetric drift: v_c^2 - v_phi^2 = sigma_R^2 * (1 - sigma_phi^2/sigma_R^2 - R/sigma_R * d(sigma_R^2)/dR)
    # Simplified: assume sigma_phi = 0.7 * sigma_R (epicyclic approximation)
    # and use exponential disc gradient

    sigma_phi = 0.7 * sigma_R

    # Gradient term (analytical for exponential disc)
    d_sigma_R_sq_dR = -2 * sigma_R**2 / R_d  # Approximate

    # v_c^2 - v_phi^2 = sigma_R^2 * (1 - (sigma_phi/sigma_R)^2 + R/(2*sigma_R^2) * d_sigma_R^2/dR)
    correction = sigma_R**2 * (
        1 - (sigma_phi / sigma_R) ** 2 + R / (2 * sigma_R**2) * d_sigma_R_sq_dR
    )
    correction = np.maximum(correction, 0.0)

    v_phi_sq = v_c**2 - correction
    v_phi_sq = np.maximum(v_phi_sq, 0.0)

    return np.sqrt(v_phi_sq)


def jeans_dispersion_spherical(
    r: np.ndarray, m_enc: np.ndarray, rho: np.ndarray, beta: float = 0.0
) -> np.ndarray:
    """Calculate velocity dispersion from spherical Jeans equation.

    Args:
        r: Radial positions (kpc).
        m_enc: Enclosed mass at each radius (Msun).
        rho: Density at each radius (Msun/kpc^3).
        beta: Anisotropy parameter (0 = isotropic, 0.5 = radial).

    Returns:
        Radial velocity dispersion sigma_r (km/s).
    """
    G = 4.302e-6  # kpc (km/s)^2 / Msun

    # sigma_r^2 = (1/rho) * integral_r^infty (rho * G * M(<r') / r'^2 * dr')
    # Simplified: assume constant beta and use local approximation

    # For isotropic case and power-law profiles, approximate as:
    sigma_r_sq = G * m_enc / (2 * r) * (1 - beta)
    sigma_r_sq = np.maximum(sigma_r_sq, 0.0)

    return np.sqrt(sigma_r_sq)


def gas_dispersion_from_temperature(T: float) -> float:
    """Calculate gas velocity dispersion from temperature.

    Args:
        T: Gas temperature (K).

    Returns:
        1D velocity dispersion (km/s).
    """
    k_B = 1.38e-16  # erg/K
    m_p = 1.673e-24  # g
    mu = 0.6  # Mean molecular weight (ionized gas)

    # sigma = sqrt(k_B * T / (mu * m_p))
    sigma_cgs = np.sqrt(k_B * T / (mu * m_p))
    sigma_km_s = sigma_cgs * 1e-5  # cm/s to km/s

    return sigma_km_s


def disc_velocity_dispersions(R: np.ndarray, sigma_R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate phi and z velocity dispersions from radial dispersion.

    Uses epicyclic approximation relationships.

    Args:
        R: Cylindrical radial positions (kpc).
        sigma_R: Radial velocity dispersion (km/s).

    Returns:
        Tuple of (sigma_phi, sigma_z) velocity dispersions (km/s).
    """
    # Epicyclic approximation
    sigma_phi = 0.7 * sigma_R
    sigma_z = 0.6 * sigma_R

    return sigma_phi, sigma_z


def escape_velocity(
    R: np.ndarray,
    z: np.ndarray,
    m200: float,
    c200: float,
    m_bulge: float,
    a_bulge: float,
    M_disc: float,
    R_d: float,
    z_d: float,
) -> np.ndarray:
    """Calculate escape velocity at given positions.

    Args:
        R: Cylindrical radial positions (kpc).
        z: Vertical positions (kpc).
        m200: Halo M200 mass (Msun).
        c200: Halo concentration.
        m_bulge: Bulge mass (Msun).
        a_bulge: Bulge scale length (kpc).
        M_disc: Disc mass (Msun).
        R_d: Disc scale length (kpc).
        z_d: Disc scale height (kpc).

    Returns:
        Escape velocity at each position (km/s).
    """
    from .potentials import hernquist_potential, miyamoto_nagai_potential, nfw_potential
    from .profiles import nfw_params

    r_s, _ = nfw_params(m200, c200)

    # Total potential
    psi_total = nfw_potential(R, z, m200, r_s, c200)

    if m_bulge > 0:
        psi_total += hernquist_potential(R, z, m_bulge, a_bulge)

    if M_disc > 0:
        psi_total += miyamoto_nagai_potential(R, z, M_disc, R_d, z_d)

    # v_esc^2 = -2 * Psi (assuming Psi -> 0 at infinity)
    v_esc_sq = -2 * psi_total
    v_esc_sq = np.maximum(v_esc_sq, 0.0)

    return np.sqrt(v_esc_sq)
