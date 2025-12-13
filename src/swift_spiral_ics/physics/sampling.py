"""Particle sampling functions for galaxy components."""

import numpy as np
from scipy import optimize
from scipy.special import lambertw


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
    from .profiles import nfw_params

    r_s, _ = nfw_params(m200, c200)

    # Sample radii using inverse CDF method
    # Cumulative mass M(<r) / M_total for NFW
    def cumulative_mass_fraction(r):
        x = r / r_s
        x_max = r_max / r_s
        f_r = np.log(1 + x) - x / (1 + x)
        f_max = np.log(1 + x_max) - x_max / (1 + x_max)
        return f_r / f_max

    # Inverse CDF
    u = rng.uniform(0, 1, N)
    r = np.zeros(N)

    for i in range(N):
        # Solve cumulative_mass_fraction(r) = u[i]
        u_i = u[i]
        r[i] = optimize.brentq(lambda x, u_val=u_i: cumulative_mass_fraction(x) - u_val, 1e-3, r_max)

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
        w = lambertw((u_clipped - 1.0) / np.e).real
        return -R_d * (1.0 + w)

    # Sample cylindrical R using analytic inverse CDF
    u = rng.uniform(0, 1, N)
    R = _sample_R(u)

    # Sample phi uniformly (will be modified by spiral arms if present)
    phi = rng.uniform(0, 2 * np.pi, N)

    # Apply spiral arm density modulation if requested
    if spiral_params is not None:
        from .perturbations import spiral_density_modulation

        arm_strength = spiral_params.get("arm_strength", 0.0)
        n_arms = spiral_params.get("n_arms", 2)
        pitch_deg = spiral_params.get("pitch_deg", 15.0)

        if arm_strength > 0:
            # Rejection sample until all are accepted (cap iterations for speed)
            max_iters = 5
            keep = np.zeros(N, dtype=bool)
            for _ in range(max_iters):
                modulation = spiral_density_modulation(R[~keep], phi[~keep], arm_strength, n_arms, pitch_deg)
                accept_prob = modulation / (1 + arm_strength)
                draw = rng.uniform(0, 1, np.count_nonzero(~keep))
                newly_kept = draw < accept_prob
                # Update keep mask and resample rejected
                idx_reject = np.where(~keep)[0][~newly_kept]
                keep[~keep] = newly_kept
                if not idx_reject.size:
                    break
                # Resample R, phi for rejects
                u_new = rng.uniform(0, 1, idx_reject.size)
                R[idx_reject] = _sample_R(u_new)
                phi[idx_reject] = rng.uniform(0, 2 * np.pi, idx_reject.size)

    # Apply bar density modulation if requested
    if bar_params is not None and bar_params.get("enabled", False):
        from .perturbations import bar_density_modulation

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
        from .perturbations import apply_position_perturbation_bar

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
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample velocities for halo particles using Jeans equation.

    Args:
        x, y, z: Particle positions (kpc).
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
        rng: Random number generator.

    Returns:
        Tuple of (vx, vy, vz) velocities (km/s).
    """
    from .kinematics import escape_velocity, jeans_dispersion_spherical
    from .profiles import nfw_density, nfw_mass, nfw_params

    N = len(x)
    R = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)

    # Get dispersions from Jeans equation
    r_s, delta_c = nfw_params(m200, c200)
    rho = nfw_density(r, m200, c200, delta_c, r_s)
    m_enc = nfw_mass(r, m200, c200, r_s)

    sigma_r = jeans_dispersion_spherical(r, m_enc, rho, beta=0.0)

    # Sample velocities from Gaussian (truncated at escape velocity)
    v_esc = escape_velocity(R, z, m200, c200, m_bulge, a_bulge, M_disc_star, R_d_star, z_d_star)

    vx = rng.normal(0, sigma_r, N)
    vy = rng.normal(0, sigma_r, N)
    vz = rng.normal(0, sigma_r, N)

    # Truncate at escape velocity
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    too_fast = v_mag > v_esc
    if np.any(too_fast):
        rescale = v_esc[too_fast] / v_mag[too_fast] * 0.99
        vx[too_fast] *= rescale
        vy[too_fast] *= rescale
        vz[too_fast] *= rescale

    return vx, vy, vz


def sample_bulge_velocities(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    m200: float,
    c200: float,
    m_bulge: float,
    a_bulge: float,
    M_disc_star: float,
    R_d_star: float,
    z_d_star: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample velocities for bulge particles using Jeans equation.

    Similar to halo but uses Hernquist profile for dispersions.

    Args:
        x, y, z: Particle positions (kpc).
        m200: Halo M200 mass (Msun).
        c200: Halo concentration.
        m_bulge: Bulge mass (Msun).
        a_bulge: Bulge scale length (kpc).
        M_disc_star: Stellar disc mass (Msun).
        R_d_star: Stellar disc scale length (kpc).
        z_d_star: Stellar disc scale height (kpc).
        rng: Random number generator.

    Returns:
        Tuple of (vx, vy, vz) velocities (km/s).
    """
    from .kinematics import escape_velocity, jeans_dispersion_spherical
    from .profiles import hernquist_density, hernquist_mass

    N = len(x)
    R = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)

    # Get dispersions
    rho = hernquist_density(r, m_bulge, a_bulge)
    m_enc = hernquist_mass(r, m_bulge, a_bulge)
    sigma_r = jeans_dispersion_spherical(r, m_enc, rho, beta=0.0)

    # Sample velocities
    v_esc = escape_velocity(R, z, m200, c200, m_bulge, a_bulge, M_disc_star, R_d_star, z_d_star)

    vx = rng.normal(0, sigma_r, N)
    vy = rng.normal(0, sigma_r, N)
    vz = rng.normal(0, sigma_r, N)

    # Truncate at escape velocity
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    too_fast = v_mag > v_esc
    if np.any(too_fast):
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
    rng: np.random.Generator,
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
        m200: Halo M200 mass (Msun).
        c200: Halo concentration.
        m_bulge: Bulge mass (Msun).
        a_bulge: Bulge scale length (kpc).
        M_disc_star: Total stellar disc mass (Msun).
        R_d_star: Stellar disc scale length (kpc).
        z_d_star: Stellar disc scale height (kpc).
        M_disc_gas: Gas disc mass (Msun).
        R_d_gas: Gas disc scale length (kpc).
        z_d_gas: Gas disc scale height (kpc).
        rng: Random number generator.
        spiral_params: Optional dict with spiral arm parameters.
        bar_params: Optional dict with bar parameters.
        is_gas: If True, use temperature-based dispersion instead of Q.

    Returns:
        Tuple of (vx, vy, vz) velocities (km/s).
    """
    from .kinematics import (
        asymmetric_drift_correction,
        disc_velocity_dispersions,
        gas_dispersion_from_temperature,
        toomre_q_dispersion,
    )
    from .potentials import total_circular_velocity
    from .profiles import disc_sigma_0, exponential_surface_density

    N = len(x)
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Get circular velocity
    R_unique = np.linspace(0.1, 30, 100)
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

    # Interpolate to particle positions
    v_c = np.interp(R, R_unique, v_c_profile)

    # Get dispersions
    if is_gas:
        # Use temperature-based dispersion
        T_gas = 1e4  # K, typical for warm ISM
        sigma_R = gas_dispersion_from_temperature(T_gas) * np.ones(N)
    else:
        # Use Toomre Q-based dispersion
        sigma_0 = disc_sigma_0(M_disc, R_d)
        sigma_surf = exponential_surface_density(R, sigma_0, R_d)
        sigma_R = toomre_q_dispersion(R, v_c, sigma_surf, Q_target)

    sigma_phi, sigma_z = disc_velocity_dispersions(R, sigma_R)

    # Mean azimuthal velocity with asymmetric drift
    v_phi_mean = asymmetric_drift_correction(
        R, v_c, sigma_R, exponential_surface_density(R, disc_sigma_0(M_disc, R_d), R_d), R_d
    )

    # Add spiral streaming if present
    v_R_stream = np.zeros(N)
    delta_v_phi_stream = np.zeros(N)

    if spiral_params is not None and spiral_params.get("arm_strength", 0) > 0:
        from .perturbations import spiral_streaming_velocity

        stream_frac = spiral_params.get("stream_frac", 0.0)
        n_arms = spiral_params.get("n_arms", 2)
        pitch_deg = spiral_params.get("pitch_deg", 15.0)

        v_R_stream, delta_v_phi_stream = spiral_streaming_velocity(
            R, phi, v_c, stream_frac, n_arms, pitch_deg
        )

    # Add bar streaming if present
    if bar_params is not None and bar_params.get("enabled", False):
        from .perturbations import bar_streaming_velocity

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

    return vx, vy, vz
