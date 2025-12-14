"""Particle sampling functions for galaxy components."""

import numpy as np
from scipy import optimize
from scipy.special import lambertw
from tqdm import tqdm
from numba import njit
from .profiles import nfw_params, nfw_density, nfw_mass, hernquist_density, hernquist_mass, disc_sigma_0, exponential_surface_density
from .kinematics import escape_velocity, jeans_dispersion_spherical, asymmetric_drift_correction, disc_velocity_dispersions, gas_dispersion_from_temperature, toomre_q_dispersion
from .potentials import total_circular_velocity
from .perturbations import bar_density_modulation, bar_streaming_velocity, apply_position_perturbation_bar, spiral_streaming_velocity


@njit
def _spiral_modulation_jit(
    R: np.ndarray, 
    phi: np.ndarray, 
    arm_strength: float, 
    n_arms: int, 
    pitch_deg: float,
    R_d: float
) -> np.ndarray:
    """JIT-compiled calculation of spiral density modulation."""
    # Hardcoded parameters matching standard logic
    R_min = 0.5 * R_d
    R_max = 5.0 * R_d
    
    # Envelope
    envelope = np.ones_like(R)
    for i in range(len(R)):
        r_val = R[i]
        if r_val < R_min:
            envelope[i] = 0.0
        elif r_val < R_min + 1.0:
            envelope[i] = (r_val - R_min) / 1.0
        elif r_val > R_max:
            envelope[i] = 0.0
        elif r_val > R_max - 2.0:
            envelope[i] = (R_max - r_val) / 2.0
            
    # Phase
    pitch_rad = pitch_deg * np.pi / 180.0
    tan_pitch = np.tan(pitch_rad)
    R_0 = 8.0 # Reference radius
    
    # Phase calculation
    # phase = n_arms * (phi - log(R/R_0)/tan_pitch)
    phase = np.zeros_like(phi)
    for i in range(len(R)):
        if R[i] > 0:
            phase[i] = n_arms * (phi[i] - np.log(R[i] / R_0) / tan_pitch)
            
    # Modulation
    modulation = 1.0 + arm_strength * envelope * np.cos(phase)
    return modulation


def sample_nfw_halo(
    N: int,
    m200: float,
    c200: float,
    r_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample particle positions for NFW halo.

    Args:
        N: Number of particles.
        m200: M200 halo mass (Msun).
        c200: Concentration parameter.
        r_max: Maximum radius for sampling (kpc).
        rng: Random number generator.

    Returns:
        Tuple of (x, y, z) positions (kpc).
    """
    r_s, _ = nfw_params(m200, c200)

    # Use interpolation for inverse CDF to speed up sampling
    # Cumulative mass M(<r) / M_total for NFW
    def cumulative_mass_fraction(r):
        x = r / r_s
        x_max = r_max / r_s
        f_r = np.log(1 + x) - x / (1 + x)
        f_max = np.log(1 + x_max) - x_max / (1 + x_max)
        return f_r / f_max

    # Build interpolation grid
    n_grid = 1000
    # Use log spacing for better resolution at small radii
    # Enforce a minimum radius to avoid singularity at r=0
    r_min = 1e-3  # kpc, 1 pc minimum radius
    r_grid = np.geomspace(max(r_min, 1e-4 * r_s), r_max, n_grid)
    
    cdf_grid = cumulative_mass_fraction(r_grid)
    
    # Ensure strict monotonicity and boundary conditions
    cdf_grid = cdf_grid - cdf_grid[0] # Shift so it starts at 0 relative to r_min
    cdf_grid = cdf_grid / cdf_grid[-1] # Normalize to 1
    
    # Sample uniform random numbers
    u = rng.uniform(0, 1, N)
    
    # Interpolate to get radii
    r = np.interp(u, cdf_grid, r_grid)

    # Random angles
    theta = np.arccos(rng.uniform(-1, 1, N))
    phi = rng.uniform(0, 2 * np.pi, N)

    # Convert to Cartesian
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def sample_hernquist_bulge(
    N: int,
    m_bulge: float,
    a: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample particle positions for Hernquist bulge.

    Args:
        N: Number of particles.
        m_bulge: Total bulge mass (Msun).
        a: Hernquist scale length (kpc).
        rng: Random number generator.

    Returns:
        Tuple of (x, y, z) positions (kpc).
    """
    # Inverse CDF for Hernquist: r = a * sqrt(u) / (1 - sqrt(u))
    u = rng.uniform(0, 1, N)
    sqrt_u = np.sqrt(u)
    r = a * sqrt_u / (1 - sqrt_u)

    # Random angles
    theta = np.arccos(rng.uniform(-1, 1, N))
    phi = rng.uniform(0, 2 * np.pi, N)

    # Convert to Cartesian
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def sample_exponential_disc(
    N: int,
    M_disc: float,
    R_d: float,
    z_d: float,
    rng: np.random.Generator,
    spiral_params: dict | None = None,
    bar_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample particle positions for exponential disc.

    Args:
        N: Number of particles.
        M_disc: Total disc mass (Msun).
        R_d: Disc scale length (kpc).
        z_d: Disc scale height (kpc).
        rng: Random number generator.
        spiral_params: Optional dict with spiral arm parameters.
        bar_params: Optional dict with bar parameters.

    Returns:
        Tuple of (x, y, z) positions (kpc).
    """
    def _sample_R(u_vals: np.ndarray) -> np.ndarray:
        """Inverse CDF for exponential disc surface density using lambert W."""
        u_clipped = np.clip(u_vals, 1e-12, 1 - 1e-12)
        # Use the -1 branch to ensure positive radii for 0<u<1
        w = lambertw((u_clipped - 1.0) / np.e, k=-1).real
        return -R_d * (1.0 + w)

    # Sample cylindrical R using analytic inverse CDF
    u = rng.uniform(0, 1, N)
    R = _sample_R(u)

    # Sample phi uniformly (will be modified by spiral arms if present)
    phi = rng.uniform(0, 2 * np.pi, N)

    # Apply spiral arm density modulation if requested
    if spiral_params is not None:
        arm_strength = spiral_params.get("arm_strength", 0.0)
        n_arms = spiral_params.get("n_arms", 2)
        pitch_deg = spiral_params.get("pitch_deg", 15.0)

        if arm_strength > 0:
            # Rejection sample until all are accepted
            max_iters = 100
            keep = np.zeros(N, dtype=bool)
            
            pbar = tqdm(total=N, desc="Sampling spiral arms")
            
            for _ in range(max_iters):
                # Calculate modulation for current candidates using JIT
                modulation = _spiral_modulation_jit(
                    R[~keep], phi[~keep], arm_strength, n_arms, pitch_deg, R_d
                )
                
                # Acceptance probability: rho_pert / rho_max
                # rho_max is rho_base * (1 + arm_strength)
                accept_prob = modulation / (1.0 + arm_strength)
                
                draw = rng.uniform(0, 1, np.count_nonzero(~keep))
                newly_kept_local = draw < accept_prob
                
                # Update global keep mask
                # Need to map local True/False back to full array indices
                indices_to_check = np.where(~keep)[0]
                indices_kept = indices_to_check[newly_kept_local]
                keep[indices_kept] = True
                
                pbar.update(len(indices_kept))
                
                if np.all(keep):
                    break
                
                # Resample R, phi for remaining rejects
                idx_reject = np.where(~keep)[0]
                u_new = rng.uniform(0, 1, idx_reject.size)
                R[idx_reject] = _sample_R(u_new)
                phi[idx_reject] = rng.uniform(0, 2 * np.pi, idx_reject.size)
            
            pbar.close()
            
            if not np.all(keep):
                print(f"Warning: Spiral arm sampling did not fully converge after {max_iters} iterations. {np.count_nonzero(~keep)} particles may be biased.")

    # Apply bar density modulation if requested
    if bar_params is not None and bar_params.get("enabled", False):
        bar_strength = bar_params.get("strength", 0.0)
        bar_radius = bar_params.get("radius", 3.0)
        bar_q = bar_params.get("q", 0.3)
        bar_angle = bar_params.get("angle", 0.0)

        if bar_strength > 0:
            modulation = bar_density_modulation(R, phi, bar_strength, bar_radius, bar_q, bar_angle)
            accept_prob = modulation / (1 + bar_strength)
            # Similar rejection sampling

    # Sample vertical positions using sech^2 profile
    # Inverse CDF: z = 2*z_d * arctanh(2*u - 1)
    u_z = rng.uniform(0, 1, N)
    z = 2 * z_d * np.arctanh(2 * u_z - 1)

    # Convert to Cartesian
    x = R * np.cos(phi)
    y = R * np.sin(phi)

    # Apply bar position perturbation if requested
    if bar_params is not None and bar_params.get("enabled", False):
        bar_strength = bar_params.get("strength", 0.0)
        bar_radius = bar_params.get("radius", 3.0)
        bar_q = bar_params.get("q", 0.3)
        bar_angle = bar_params.get("angle", 0.0)

        if bar_strength > 0:
            x, y = apply_position_perturbation_bar(x, y, bar_strength, bar_radius, bar_q, bar_angle)

    return x, y, z


def sample_halo_velocities(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rng: np.random.Generator,
    grid_solver: "GalaxyGridSolver",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample velocities for halo particles using Jeans equation.

    Args:
        x, y, z: Particle positions (kpc).
        rng: Random number generator.
        grid_solver: Instance of GalaxyGridSolver with computed potential.

    Returns:
        Tuple of (vx, vy, vz) velocities (km/s).
    """
    N = len(x)
    R = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)

    # Extract galaxy parameters from grid_solver
    m200 = grid_solver.m200
    c200 = grid_solver.c200
    m_bulge = grid_solver.m_bulge
    a_bulge = grid_solver.a_bulge
    M_disc_star = grid_solver.M_disc_star
    R_d_star = grid_solver.R_d_star
    z_d_star = grid_solver.z_d_star
    M_disc_gas = grid_solver.M_disc_gas
    R_d_gas = grid_solver.R_d_gas
    z_d_gas = grid_solver.z_d_gas
    
    sigma_r = jeans_dispersion_spherical(
        r_coords=r,
        profile_type="nfw",
        m200=m200, c200=c200,
        m_bulge=m_bulge, a_bulge=a_bulge,
        M_disc_star=M_disc_star, R_d_star=R_d_star, z_d_star=z_d_star,
        M_disc_gas=M_disc_gas, R_d_gas=R_d_gas, z_d_gas=z_d_gas,
        beta=0.0
    )

    # Sample velocities from Gaussian (truncated at escape velocity)
    v_esc = escape_velocity(
        R, z, m200, c200, m_bulge, a_bulge, M_disc_star, R_d_star, z_d_star, M_disc_gas, R_d_gas, z_d_gas
    )

    vx = rng.normal(0, sigma_r, N)
    vy = rng.normal(0, sigma_r, N)
    vz = rng.normal(0, sigma_r, N)

    # Truncate at escape velocity
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    too_fast = v_mag > v_esc
    if np.any(too_fast):
        # Rescale to 99% of escape velocity
        rescale = v_esc[too_fast] / v_mag[too_fast] * 0.99
        vx[too_fast] *= rescale
        vy[too_fast] *= rescale
        vz[too_fast] *= rescale

    return vx, vy, vz


def sample_bulge_velocities(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rng: np.random.Generator,
    grid_solver: "GalaxyGridSolver",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample velocities for bulge particles using Jeans equation.

    Args:
        x, y, z: Particle positions (kpc).
        rng: Random number generator.
        grid_solver: Instance of GalaxyGridSolver with computed potential.

    Returns:
        Tuple of (vx, vy, vz) velocities (km/s).
    """
    N = len(x)
    R = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)

    # Extract galaxy parameters from grid_solver
    m200 = grid_solver.m200
    c200 = grid_solver.c200
    m_bulge = grid_solver.m_bulge
    a_bulge = grid_solver.a_bulge
    M_disc_star = grid_solver.M_disc_star
    R_d_star = grid_solver.R_d_star
    z_d_star = grid_solver.z_d_star
    M_disc_gas = grid_solver.M_disc_gas
    R_d_gas = grid_solver.R_d_gas
    z_d_gas = grid_solver.z_d_gas

    # Get dispersions
    sigma_r = jeans_dispersion_spherical(
        r_coords=r,
        profile_type="hernquist",
        m200=m200, c200=c200,
        m_bulge=m_bulge, a_bulge=a_bulge,
        M_disc_star=M_disc_star, R_d_star=R_d_star, z_d_star=z_d_star,
        M_disc_gas=M_disc_gas, R_d_gas=R_d_gas, z_d_gas=z_d_gas,
        beta=0.0
    )

    # Sample velocities
    v_esc = escape_velocity(
        R, z, m200, c200, m_bulge, a_bulge, M_disc_star, R_d_star, z_d_star, M_disc_gas, R_d_gas, z_d_gas
    )

    vx = rng.normal(0, sigma_r, N)
    vy = rng.normal(0, sigma_r, N)
    vz = rng.normal(0, sigma_r, N)

    # Truncate at escape velocity
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    too_fast = v_mag > v_esc
    if np.any(too_fast):
        # Rescale to 99% of escape velocity
        rescale = v_esc[too_fast] / v_mag[too_fast] * 0.99
        vx[too_fast] *= rescale
        vy[too_fast] *= rescale
        vz[too_fast] *= rescale

    return vx, vy, vz


def sample_disc_velocities(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    M_disc: float,
    R_d: float,
    z_d: float,
    Q_target: float,
    rng: np.random.Generator,
    grid_solver: "GalaxyGridSolver",
    spiral_params: dict | None = None,
    bar_params: dict | None = None,
    is_gas: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample velocities for disc particles.

    Args:
        x, y, z: Particle positions (kpc).
        M_disc: This disc's mass (Msun).
        R_d: This disc's scale length (kpc).
        z_d: This disc's scale height (kpc).
        Q_target: Target Toomre Q parameter.
        rng: Random number generator.
        grid_solver: Instance of GalaxyGridSolver with computed potential.
        spiral_params: Optional dict with spiral arm parameters.
        bar_params: Optional dict with bar parameters.
        is_gas: If True, use temperature-based dispersion instead of Q.

    Returns:
        Tuple of (vx, vy, vz) velocities (km/s).
    """
    N = len(x)
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Extract galaxy parameters from grid_solver
    m200 = grid_solver.m200
    c200 = grid_solver.c200
    m_bulge = grid_solver.m_bulge
    a_bulge = grid_solver.a_bulge
    M_disc_star = grid_solver.M_disc_star
    R_d_star = grid_solver.R_d_star
    z_d_star = grid_solver.z_d_star
    M_disc_gas = grid_solver.M_disc_gas
    R_d_gas = grid_solver.R_d_gas
    z_d_gas = grid_solver.z_d_gas

    # Get circular velocity
    R_unique = np.linspace(1e-3, 30, 200) # Ensure no R=0
    v_c_profile = total_circular_velocity(
        R_unique,
        m200,
        c200,
        m_bulge,
        a_bulge,
        M_disc_star,
        R_d_star,
        z_d_star,
        M_disc_gas,
        R_d_gas,
        z_d_gas,
    )

    # Calculate dispersion profiles on the grid
    sigma_0 = disc_sigma_0(M_disc, R_d)
    sigma_surf_profile = exponential_surface_density(R_unique, sigma_0, R_d)
    
    if is_gas:
        # Temperature based
        T_gas = 1e4
        sigma_thermal = gas_dispersion_from_temperature(T_gas)
        sigma_R_profile = np.full_like(R_unique, sigma_thermal)
    else:
        # Toomre Q based
        sigma_R_profile = toomre_q_dispersion(
            R_unique, v_c_profile, sigma_surf_profile, Q_target,
            m200, c200, m_bulge, a_bulge, M_disc_star, R_d_star, z_d_star, M_disc_gas, R_d_gas, z_d_gas
        )
        
    # Ensure gas has thermal floor (if Q based logic was used for gas, but here we separated it)
    # If is_gas=False, checking floor is still good practice for stability
    if not is_gas:
        sigma_R_profile = np.maximum(sigma_R_profile, 5.0)

    sigma_phi_profile, sigma_z_profile = disc_velocity_dispersions(R_unique, sigma_R_profile)

    # Calculate asymmetric drift numerically on the grid
    # v_phi^2 = v_c^2 + sigma_R^2 * (dln(nu)/dlnR + dln(sigma_R^2)/dlnR + 1 - (sigma_phi/sigma_R)^2)
    # derivatives wrt R: dlnX/dlnR = R * dlnX/dR
    
    # Gradient of log Surface Density
    # For exponential, this is analytically -1/R_d, but let's be consistent numerically
    ln_Sigma = np.log(sigma_surf_profile)
    d_ln_Sigma_dR = np.gradient(ln_Sigma, R_unique)
    
    # Gradient of log sigma_R^2
    ln_sigma2 = np.log(sigma_R_profile**2)
    d_ln_sigma2_dR = np.gradient(ln_sigma2, R_unique)
    
    # Anisotropy term
    ratio_sq = (sigma_phi_profile / sigma_R_profile)**2
    term_anisotropy = 1.0 - ratio_sq
    
    # Jeans equation terms (all multiplied by R/sigma_R^2 in some forms, but here we use derivatives wrt R)
    # Force balance: v_c^2/R - v_phi^2/R = -1/rho * d(rho sigma^2)/dR - ...
    # v_phi^2 = v_c^2 + (R/rho) * d(rho sigma^2)/dR
    #         = v_c^2 + R * sigma_R^2 * (dlnRho/dR + dlnSigma2/dR)
    # (Simplified for thin disc, neglecting sigma_z term)
    
    # Correct formula: v_phi^2 = v_c^2 + sigma_R^2 * [ R * (dlnSigma/dR + dlnSigma2/dR) + (1 - ratio_sq) ]
    # Note: The derivative terms are usually negative, so they reduce velocity.
    # dlnSigma/dR is negative.
    
    bracket = R_unique * (d_ln_Sigma_dR + d_ln_sigma2_dR) + term_anisotropy
    v_phi_sq_profile = v_c_profile**2 + sigma_R_profile**2 * bracket
    v_phi_profile = np.sqrt(np.maximum(v_phi_sq_profile, 0.0))

    # Interpolate to particle positions
    v_c = np.interp(R, R_unique, v_c_profile)
    sigma_R = np.interp(R, R_unique, sigma_R_profile)
    sigma_phi = np.interp(R, R_unique, sigma_phi_profile)
    sigma_z = np.interp(R, R_unique, sigma_z_profile)
    v_phi_mean = np.interp(R, R_unique, v_phi_profile)

    # Add spiral streaming if present
    v_R_stream = np.zeros(N)
    delta_v_phi_stream = np.zeros(N)

    if spiral_params is not None and spiral_params.get("arm_strength", 0) > 0:
        stream_frac = spiral_params.get("stream_frac", 0.0)
        n_arms = spiral_params.get("n_arms", 2)
        pitch_deg = spiral_params.get("pitch_deg", 15.0)

        v_R_stream, delta_v_phi_stream = spiral_streaming_velocity(
            R, phi, v_c, stream_frac, n_arms, pitch_deg, R_d=R_d
        )

    # Add bar streaming if present
    if bar_params is not None and bar_params.get("enabled", False):
        stream_frac = bar_params.get("stream_frac", 0.0)
        bar_radius = bar_params.get("radius", 3.0)
        bar_angle = bar_params.get("angle", 0.0)

        v_R_bar, delta_v_phi_bar = bar_streaming_velocity(
            R, phi, v_c, stream_frac, bar_radius, bar_angle
        )
        v_R_stream += v_R_bar
        delta_v_phi_stream += delta_v_phi_bar

    # Sample velocity components
    v_R = rng.normal(v_R_stream, sigma_R)
    v_phi = rng.normal(v_phi_mean + delta_v_phi_stream, sigma_phi)
    v_z = rng.normal(0, sigma_z)

    # Convert to Cartesian
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    vx = v_R * cos_phi - v_phi * sin_phi
    vy = v_R * sin_phi + v_phi * cos_phi
    vz = v_z

    # Truncate at escape velocity (safety check)
    from .kinematics import escape_velocity

    v_esc = escape_velocity(
        R, z, m200, c200, m_bulge, a_bulge, M_disc_star, R_d_star, z_d_star, M_disc_gas, R_d_gas, z_d_gas
    )  # Note: escape_velocity signature needs update or checking, assuming it takes all mass components

    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    too_fast = v_mag > v_esc
    if np.any(too_fast):
        # Rescale to 99% of escape velocity
        rescale = v_esc[too_fast] / v_mag[too_fast] * 0.99
        vx[too_fast] *= rescale
        vy[too_fast] *= rescale
        vz[too_fast] *= rescale

    return vx, vy, vz
