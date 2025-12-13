"""Spiral arm and bar perturbation functions."""

import numpy as np


def spiral_arm_phase(
    R: np.ndarray, phi: np.ndarray, n_arms: int, pitch_deg: float, R_0: float = 8.0
) -> np.ndarray:
    """Calculate spiral arm phase for logarithmic spiral.

    Args:
        R: Cylindrical radial positions (kpc).
        phi: Azimuthal angles (radians).
        n_arms: Number of spiral arms.
        pitch_deg: Pitch angle (degrees).
        R_0: Reference radius for spiral pattern (kpc).

    Returns:
        Spiral phase at each position (0 to 2pi indicates position relative to arm).
    """
    pitch_rad = np.deg2rad(pitch_deg)
    tan_pitch = np.tan(pitch_rad)

    # Logarithmic spiral: phi - phi_0 = (1/tan(pitch)) * ln(R/R_0)
    # Phase relative to arm: m * (phi - phi_0)
    phase = n_arms * (phi - np.log(R / R_0) / tan_pitch)

    # Wrap to [0, 2pi)
    phase = np.mod(phase, 2 * np.pi)

    return phase


def spiral_arm_amplitude(
    R: np.ndarray, 
    R_min: float = 2.0, 
    R_max: float = 15.0,
    R_d: float | None = None
) -> np.ndarray:
    """Radial envelope for spiral arm strength.

    Args:
        R: Cylindrical radial positions (kpc).
        R_min: Inner radius where arms start (kpc) (used if R_d is None).
        R_max: Outer radius where arms end (kpc) (used if R_d is None).
        R_d: Disc scale length (kpc). If provided, scales R_min/R_max relative to this.

    Returns:
        Envelope amplitude (0 to 1).
    """
    if R_d is not None:
        # Scale relative to disc size if provided
        # Typical: start at ~0.5 scale lengths, extend to ~5
        R_min = 0.5 * R_d
        R_max = 5.0 * R_d

    envelope = np.ones_like(R)

    # Smooth turnon at inner radius
    inner_mask = R < R_min
    inner_transition = (R_min < R) & (R < R_min + 1.0)
    envelope[inner_mask] = 0.0
    envelope[inner_transition] = (R[inner_transition] - R_min) / 1.0

    # Smooth turnoff at outer radius
    outer_transition = (R_max - 2.0 < R) & (R < R_max)
    outer_mask = R >= R_max
    envelope[outer_transition] = (R_max - R[outer_transition]) / 2.0
    envelope[outer_mask] = 0.0

    return envelope


def spiral_density_modulation(
    R: np.ndarray, 
    phi: np.ndarray, 
    arm_strength: float, 
    n_arms: int, 
    pitch_deg: float,
    R_d: float | None = None
) -> np.ndarray:
    """Calculate density modulation factor for spiral arms.

    Args:
        R: Cylindrical radial positions (kpc).
        phi: Azimuthal angles (radians).
        arm_strength: Amplitude of density perturbation (fractional).
        n_arms: Number of spiral arms.
        pitch_deg: Pitch angle (degrees).
        R_d: Disc scale length (kpc), optional.

    Returns:
        Density modulation factor (multiply base density by this).
    """
    phase = spiral_arm_phase(R, phi, n_arms, pitch_deg)
    envelope = spiral_arm_amplitude(R, R_d=R_d)

    # Density perturbation: rho' = rho * (1 + A * cos(phase))
    modulation = 1.0 + arm_strength * envelope * np.cos(phase)

    return modulation


def spiral_streaming_velocity(
    R: np.ndarray,
    phi: np.ndarray,
    v_c: np.ndarray,
    stream_frac: float,
    n_arms: int,
    pitch_deg: float,
    R_d: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate streaming velocity perturbations for spiral arms.

    Args:
        R: Cylindrical radial positions (kpc).
        phi: Azimuthal angles (radians).
        v_c: Circular velocity at each radius (km/s).
        stream_frac: Streaming amplitude as fraction of v_c.
        n_arms: Number of spiral arms.
        pitch_deg: Pitch angle (degrees).
        R_d: Disc scale length (kpc), optional.

    Returns:
        Tuple of (v_R, delta_v_phi) velocity perturbations (km/s).
    """
    phase = spiral_arm_phase(R, phi, n_arms, pitch_deg)
    envelope = spiral_arm_amplitude(R, R_d=R_d)
    pitch_rad = np.deg2rad(pitch_deg)

    # Streaming amplitude
    v_stream = stream_frac * v_c * envelope

    # Velocity perturbations aligned with spiral arms
    # v_R perturbation (positive = outward near arm)
    v_R = v_stream * np.sin(phase) * np.sin(pitch_rad)

    # v_phi perturbation (leads/lags circular velocity)
    delta_v_phi = -v_stream * np.sin(phase) * np.cos(pitch_rad)

    return v_R, delta_v_phi


def bar_density_modulation(
    R: np.ndarray,
    phi: np.ndarray,
    bar_strength: float,
    bar_radius: float,
    bar_q: float,
    bar_angle: float = 0.0,
) -> np.ndarray:
    """Calculate density modulation for a bar.

    Args:
        R: Cylindrical radial positions (kpc).
        phi: Azimuthal angles (radians).
        bar_strength: Amplitude of bar perturbation (fractional).
        bar_radius: Extent of bar (kpc).
        bar_q: Axis ratio of bar (< 1 for elongated along major axis).
        bar_angle: Orientation of bar major axis (radians).

    Returns:
        Density modulation factor.
    """
    # Rotate to bar frame
    phi_bar = phi - bar_angle

    # Elliptical radius in bar frame
    x = R * np.cos(phi_bar)
    y = R * np.sin(phi_bar)
    R_ell = np.sqrt(x**2 + (y / bar_q) ** 2)

    # Smooth envelope
    envelope = np.exp(-((R_ell / bar_radius) ** 2))

    # m=2 perturbation aligned with bar
    modulation = 1.0 + bar_strength * envelope * np.cos(2 * phi_bar)

    return modulation


def bar_streaming_velocity(
    R: np.ndarray,
    phi: np.ndarray,
    v_c: np.ndarray,
    stream_frac: float,
    bar_radius: float,
    bar_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate streaming velocity perturbations for a bar.

    Args:
        R: Cylindrical radial positions (kpc).
        phi: Azimuthal angles (radians).
        v_c: Circular velocity at each radius (km/s).
        stream_frac: Streaming amplitude as fraction of v_c.
        bar_radius: Extent of bar (kpc).
        bar_angle: Orientation of bar major axis (radians).

    Returns:
        Tuple of (v_R, delta_v_phi) velocity perturbations (km/s).
    """
    # Rotate to bar frame
    phi_bar = phi - bar_angle

    # Envelope
    envelope = np.exp(-((R / bar_radius) ** 2))

    # Streaming amplitude
    v_stream = stream_frac * v_c * envelope

    # Velocity perturbations along bar (m=2 pattern)
    v_R = v_stream * np.sin(2 * phi_bar)
    delta_v_phi = v_stream * np.cos(2 * phi_bar)

    return v_R, delta_v_phi


def apply_position_perturbation_bar(
    x: np.ndarray,
    y: np.ndarray,
    bar_strength: float,
    bar_radius: float,
    bar_q: float,
    bar_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply elliptical position perturbation for bar.

    Args:
        x: x positions (kpc).
        y: y positions (kpc).
        bar_strength: Strength of bar deformation (fractional).
        bar_radius: Extent of bar (kpc).
        bar_q: Axis ratio of bar.
        bar_angle: Orientation of bar major axis (radians).

    Returns:
        Tuple of (x_pert, y_pert) perturbed positions (kpc).
    """
    # Rotate to bar frame
    cos_angle = np.cos(bar_angle)
    sin_angle = np.sin(bar_angle)

    x_bar = x * cos_angle + y * sin_angle
    y_bar = -x * sin_angle + y * cos_angle

    # Apply elliptical deformation
    R = np.sqrt(x_bar**2 + y_bar**2)
    envelope = np.exp(-((R / bar_radius) ** 2))

    # Compress along minor axis within bar region
    deform_factor = 1.0 + bar_strength * envelope * (bar_q - 1.0)
    y_bar_pert = y_bar * deform_factor

    # Rotate back
    x_pert = x_bar * cos_angle - y_bar_pert * sin_angle
    y_pert = x_bar * sin_angle + y_bar_pert * cos_angle

    return x_pert, y_pert
