"""Kinematic calculations for galaxy components."""

import numpy as np
from galpy.potential import epifreq
from scipy.integrate import quad
from tqdm import tqdm
from .constants import G
from .potentials import (
    get_galpy_potentials,
    hernquist_potential,
    miyamoto_nagai_potential,
    nfw_potential,
)
from .profiles import (
    exponential_disc_mass,
    hernquist_density,
    hernquist_mass,
    nfw_density,
    nfw_mass,
    nfw_params,
)


def epicyclic_frequency(
    R: np.ndarray,
    m200: float,
    c200: float,
    m_bulge: float,
    a_bulge: float,
    M_disc_star: float,
    R_d_star: float,
    z_d_star: float,
    M_disc_gas: float = 0.0,
    R_d_gas: float = 1.0,
    z_d_gas: float = 0.1,
) -> np.ndarray:
    """Calculate epicyclic frequency using galpy.

    Args:
        R: Cylindrical radial positions (kpc).
        [Mass parameters...]

    Returns:
        Epicyclic frequency kappa at each radius (km/s/kpc).
    """
    from tqdm import tqdm
    
    pots = get_galpy_potentials(
        m200, c200, m_bulge, a_bulge, M_disc_star, R_d_star, z_d_star, M_disc_gas, R_d_gas, z_d_gas
    )
    
    # galpy.potential.epifreq can take array inputs, but DoubleExponentialDiskPotential
    # force methods require scalar inputs. So we loop manually.
    # R needs to be >= 0
    R_safe = np.maximum(R, 1e-4)
    
    kappa_values = []
    for r_val in tqdm(R_safe, desc="Calculating kappa"):
        kappa_values.append(epifreq(pots, r_val))
    
    kappa = np.array(kappa_values)
    
    return kappa


def toomre_q_dispersion(
    R: np.ndarray,
    v_c: np.ndarray,
    sigma_surf: np.ndarray,
    Q_target: float,
    m200: float,
    c200: float,
    m_bulge: float,
    a_bulge: float,
    M_disc_star: float,
    R_d_star: float,
    z_d_star: float,
    M_disc_gas: float = 0.0,
    R_d_gas: float = 1.0,
    z_d_gas: float = 0.1,
) -> np.ndarray:
    """Calculate radial velocity dispersion from Toomre Q.

    Args:
        R: Cylindrical radial positions (kpc).
        v_c: Circular velocity at each radius (km/s).
        sigma_surf: Surface density at each radius (Msun/kpc^2).
        Q_target: Target Toomre Q parameter.
        [Mass parameters for kappa calculation...]

    Returns:
        Radial velocity dispersion sigma_R (km/s).
    """
    # Calculate kappa using galpy
    kappa = epicyclic_frequency(
        R, m200, c200, m_bulge, a_bulge, M_disc_star, R_d_star, z_d_star, M_disc_gas, R_d_gas, z_d_gas
    )

    # Q = sigma_R * kappa / (pi * G * Sigma)
    # sigma_R = Q * pi * G * Sigma / kappa
    sigma_R = Q_target * np.pi * G.value * sigma_surf / kappa

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
    # Jeans equation for azimuthal velocity:
    # v_phi^2 = v_c^2 + sigma_R^2 * (dln(nu)/dlnR + dln(sigma_R^2)/dlnR + 1 - (sigma_phi/sigma_R)^2)
    
    # Assume sigma_phi/sigma_R from epicyclic approximation
    sigma_phi = 0.7 * sigma_R
    ratio_sq = (sigma_phi / sigma_R)**2
    
    # Derivatives for exponential disc
    # nu ~ exp(-R/R_d) -> dln(nu)/dlnR = -R/R_d
    # sigma_R^2 ~ exp(-R/R_d) (approximation for constant Q/flat v_c) -> dln(sigma^2)/dlnR = -R/R_d
    
    term_density = -R / R_d
    term_pressure = -R / R_d
    term_anisotropy = 1.0 - ratio_sq
    
    bracket = term_density + term_pressure + term_anisotropy
    
    v_phi_sq = v_c**2 + sigma_R**2 * bracket
    v_phi_sq = np.maximum(v_phi_sq, 0.0)

    return np.sqrt(v_phi_sq)


def jeans_dispersion_spherical(
    r_coords: np.ndarray, # These are the particle radii to evaluate sigma at
    profile_type: str, # "nfw" or "hernquist"
    m200: float,
    c200: float,
    m_bulge: float,
    a_bulge: float,
    M_disc_star: float,
    R_d_star: float,
    z_d_star: float,
    M_disc_gas: float = 0.0,
    R_d_gas: float = 1.0,
    z_d_gas: float = 0.1,
    beta: float = 0.0,
) -> np.ndarray:
    """Calculate velocity dispersion from spherical Jeans equation.

    Solves the integral: sigma_r^2(r) = (1/rho_comp(r)) * integral_r^infty rho_comp(r') dPhi/dr' dr'

    Args:
        r_coords: Radial positions (kpc) of particles.
        profile_type: Type of component ('nfw' or 'hernquist').
        [All galaxy mass parameters...]
        beta: Anisotropy parameter (0 = isotropic).

    Returns:
        Radial velocity dispersion sigma_r (km/s).
    """
    from scipy.integrate import quad
    from tqdm import tqdm
    from .constants import G
    from .profiles import (
        exponential_disc_mass,
        hernquist_density,
        hernquist_mass,
        nfw_density,
        nfw_mass,
        nfw_params,
    )

    G_val = G.value
    sigma_r_sq = np.zeros_like(r_coords)

    # Calculate NFW halo parameters once for scope
    r_s_nfw, delta_c_nfw = nfw_params(m200, c200)

    # Define helper functions for density and total enclosed mass
    # These must be passed to the integrand or accessed from closure
    
    # Total enclosed mass (used for dPhi/dr')
    def total_enclosed_mass_func(r_prime):
        m_halo = nfw_mass(r_prime, m200, c200, r_s_nfw)
        
        m_b = 0.0
        if m_bulge > 0:
            m_b = hernquist_mass(r_prime, m_bulge, a_bulge)
        
        m_ds = 0.0
        if M_disc_star > 0:
            m_ds = exponential_disc_mass(r_prime, M_disc_star, R_d_star)
            
        m_dg = 0.0
        if M_disc_gas > 0:
            m_dg = exponential_disc_mass(r_prime, M_disc_gas, R_d_gas)
            
        return m_halo + m_b + m_ds + m_dg

    # Component density (used for rho_comp(r) and rho_comp(r'))
    def component_density_func(r_prime, comp_type):
        if comp_type == "nfw":
            r_s_nfw, delta_c_nfw = nfw_params(m200, c200)
            return nfw_density(r_prime, m200, c200, delta_c_nfw, r_s_nfw)
        elif comp_type == "hernquist":
            return hernquist_density(r_prime, m_bulge, a_bulge)
        else:
            raise ValueError("Unknown profile type for Jeans solver")

    # Integrand function: rho_comp(r') * G * M_total(r') / r'^2
    def integrand(r_prime, comp_type):
        # Handle r_prime = 0 for numerical stability
        r_prime_safe = np.maximum(r_prime, 1e-4)
        return component_density_func(r_prime_safe, comp_type) * G_val * total_enclosed_mass_func(r_prime_safe) / r_prime_safe**2

    # Loop over each radial coordinate for the particles
    for i, r_val in enumerate(tqdm(r_coords, desc=f"Solving Jeans for {profile_type}")):
        # Ensure r_val is not too small for initial rho_comp(r_val) evaluation
        r_safe = np.maximum(r_val, 1e-4)
        
        # Denominator: component density at current radius
        rho_comp_r = component_density_func(r_safe, profile_type)
        
        # Integrate from r to infinity
        # Use a large upper bound for infinity (e.g., 1000 kpc or 10*R200)
        # Assuming r_max is max extent of halo/bulge
        r_max_integration = max(200.0, r_s_nfw * 10) # Roughly 2*R200 of halo
        
        integral_val, _ = quad(integrand, r_safe, r_max_integration, args=(profile_type,))
        
        # Apply the (1 - beta) term and divide by density
        sigma_r_sq[i] = (1.0 - beta) * integral_val / rho_comp_r

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
    M_disc_star: float,
    R_d_star: float,
    z_d_star: float,
    M_disc_gas: float = 0.0,
    R_d_gas: float = 1.0,
    z_d_gas: float = 0.1,
) -> np.ndarray:
    """Calculate escape velocity at given positions.

    Args:
        R: Cylindrical radial positions (kpc).
        z: Vertical positions (kpc).
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
        Escape velocity at each position (km/s).
    """
    r_s, _ = nfw_params(m200, c200)

    # Total potential
    psi_total = nfw_potential(R, z, m200, r_s, c200)

    if m_bulge > 0:
        psi_total += hernquist_potential(R, z, m_bulge, a_bulge)

    if M_disc_star > 0:
        psi_total += miyamoto_nagai_potential(R, z, M_disc_star, R_d_star, z_d_star)

    if M_disc_gas > 0:
        psi_total += miyamoto_nagai_potential(R, z, M_disc_gas, R_d_gas, z_d_gas)

    # v_esc^2 = -2 * Psi (assuming Psi -> 0 at infinity)
    v_esc_sq = -2 * psi_total
    v_esc_sq = np.maximum(v_esc_sq, 0.0)

    return np.sqrt(v_esc_sq)
