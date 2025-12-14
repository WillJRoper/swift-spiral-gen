import numpy as np
import unyt
from unyt import K, m, s, kg
from .constants import G, k_B, m_p
from scipy.integrate import quad
from tqdm import tqdm


def epicyclic_frequency(R: np.ndarray, v_c: np.ndarray) -> np.ndarray:
    """
    Calculate the epicyclic frequency (kappa) for a given rotation curve.
    kappa^2 = 2/R * d(R v_c)/dR
    """
    # Numerically differentiate for kappa. Assumes R is sorted.
    # To handle R=0, add a small epsilon
    R_safe = np.maximum(R, 1e-6)
    
    # Calculate angular velocity Omega = v_c / R
    Omega = v_c / R_safe
    
    # Calculate d(Omega^2)/dR = d(v_c^2/R^2)/dR
    Omega_sq = Omega**2
    d_Omega_sq_dR = np.gradient(Omega_sq, R)
    
    # kappa^2 = R * d(Omega^2)/dR + 4 * Omega^2
    kappa_sq = R_safe * d_Omega_sq_dR + 4 * Omega_sq
    
    kappa_sq = np.maximum(kappa_sq, 1e-6) # Ensure non-negative
    return np.sqrt(kappa_sq)


def disc_velocity_dispersions(R: np.ndarray, sigma_R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate sigma_phi and sigma_z from sigma_R using epicyclic approximation.
    Assumes an isotropic velocity ellipsoid in the meridional plane and a flat rotation curve.
    """
    # From Binney & Tremaine, 2nd ed. eq. 4.227b for an axisymmetric system
    # sigma_phi^2 / sigma_R^2 = kappa^2 / (2 * Omega^2)
    # For a flat rotation curve, kappa = sqrt(2) * Omega, so sigma_phi = sigma_R / sqrt(2)
    # Assuming local approximation: sigma_phi ~ 0.7 * sigma_R is common.
    # Sigma_z often assumed to be ~0.5-0.7 * sigma_R
    
    sigma_phi = 0.7 * sigma_R # A common approximation
    sigma_z = 0.5 * sigma_R # Another common approximation

    return sigma_phi, sigma_z


def gas_dispersion_from_temperature(T: float) -> float:
    """
    Calculate gas velocity dispersion from temperature, assuming primordial composition.
    sigma = sqrt(k_B * T / mu * m_p)
    where mu = 0.59 for primordial gas.
    """
    if not isinstance(T, unyt.unyt_quantity):
        T = T * unyt.K
        
    mu = 0.59 # Mean molecular weight for primordial gas (X=0.76, Y=0.24)
    sigma_sq = (k_B * T / (mu * m_p)).to("km**2/s**2").value
    return np.sqrt(sigma_sq)


def escape_velocity_from_grid(
    R: np.ndarray,
    z: np.ndarray,
    grid_solver,
) -> np.ndarray:
    """Calculate escape velocity using grid potential.

    Args:
        R: Cylindrical radial positions (kpc).
        z: Vertical positions (kpc).
        grid_solver: GalaxyGridSolver instance.

    Returns:
        Escape velocity at each position (km/s).
    """
    res = grid_solver.get_potential_and_forces(R, z)
    Phi = res["Phi"]
    
    # v_esc = sqrt(-2 * Phi)
    # Assuming Phi -> 0 at infinity. Grid solver sums -G*M/r, so Phi is negative.
    
    v_esc_sq = -2.0 * Phi
    v_esc_sq = np.maximum(v_esc_sq, 0.0)
    
    return np.sqrt(v_esc_sq)


def jeans_dispersion_spherical_from_grid(
    r_coords: np.ndarray,
    grid_solver,
    pos_comp: np.ndarray,
    mass_comp: np.ndarray,
    beta: float = 0.0,
) -> np.ndarray:
    """Calculate velocity dispersion from spherical Jeans equation using grid potential and numerical density.

    Args:
        r_coords: Radial positions (kpc) of particles to query sigma_r.
        grid_solver: GalaxyGridSolver instance.
        pos_comp: (N,3) array of positions for the component.
        mass_comp: (N,) array of masses for the component.
        beta: Anisotropy parameter (0 = isotropic).

    Returns:
        Radial velocity dispersion sigma_r (km/s).
    """
    
    # Define a radial grid for solving Jeans equation
    r_min_prof = 1e-4
    r_max_prof = max(np.max(r_coords) * 1.5, grid_solver.R_grid[-1])
    n_prof_grid = 1000 # Sufficient for smooth profile
    r_prof_grid = np.geomspace(r_min_prof, r_max_prof, n_prof_grid)
    
    # Get numerical density profile of THIS component
    rho_comp_prof = grid_solver.get_component_density_profile(r_prof_grid, pos_comp, mass_comp, profile_type="spherical")
    
    # Get numerical radial force profile from TOTAL potential (interpolated from grid)
    # FR = -dPhi/dR, for spherical system, this is F_r
    Fr_prof_total = grid_solver.get_spherical_force_profile(r_prof_grid)
    
    # Integrand for Jeans equation: rho_comp(r') * |F_r(r')|
    # F_r from grid_solver is the physical radial force component
    # It should be negative (inward) for attractive gravity.
    
    def integrand(r_prime_val):
        rho_at_r_prime = np.interp(r_prime_val, r_prof_grid, rho_comp_prof, left=0, right=0)
        Fr_at_r_prime = np.interp(r_prime_val, r_prof_grid, Fr_prof_total, left=0, right=0)
        
        return rho_at_r_prime * np.abs(Fr_at_r_prime) # Force magnitude

    sigma_r_sq_prof_grid = np.zeros_like(r_prof_grid)

    for i, r_val in enumerate(tqdm(r_prof_grid, desc=f"Solving Jeans (Grid) for component")):
        # Ensure r_val is not too small
        r_safe = np.maximum(r_val, r_min_prof)
        
        rho_comp_r = np.interp(r_safe, r_prof_grid, rho_comp_prof, left=0, right=0)
        
        # Integrate from r to large distance
        # Use a large upper bound, but ensure it's within range of Fr_prof_total
        r_max_integration = r_prof_grid[-1]
        
        if rho_comp_r > 0:
            integral_val, _ = quad(
                integrand, r_safe, r_max_integration,
                args=() # No args needed as integrand uses closure
            )
            sigma_r_sq_prof_grid[i] = (1.0 - beta) * integral_val / rho_comp_r
        else:
            sigma_r_sq_prof_grid[i] = 0.0

    sigma_r_sq = np.interp(r_coords, r_prof_grid, sigma_r_sq_prof_grid, left=0, right=0)
    sigma_r_sq = np.maximum(sigma_r_sq, 0.0)
    return np.sqrt(sigma_r_sq)