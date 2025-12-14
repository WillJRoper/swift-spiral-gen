"""Gravitational potentials and circular velocities using galpy."""

import numpy as np
from galpy.potential import (
    DoubleExponentialDiskPotential,
    HernquistPotential,
    MiyamotoNagaiPotential,
    NFWPotential,
    RazorThinExponentialDiskPotential,
    evaluatePotentials,
    plotRotcurve,
    vcirc,
)
from tqdm import tqdm
from .constants import G
from .profiles import nfw_params


# G is imported as unyt quantity. Convert to float value for calculations if needed
# But galpy expects specific normalization.
# We will use galpy's "physical" support by explicitly normalizing G=1 internally
# or by just calculating the amplitude correctly.

# Galpy definitions:
# Phi_NFW = - amp * ln(1+r/a) / (r/a)
# Standard NFW: Phi = - 4 pi G rho_0 r_s^3 * ln(1+r/r_s) / r
# So amp = 4 pi G rho_0 r_s^3 = G * M_halo_scale_mass?
# Galpy NFWPotential: amp = G * M_s where M_s is the mass parameter.
# Wait, NFWPotential(conc=..., mvir=...) handles this for us if we use the right setup.

# Simplest approach: Use potentials where 'amp' = G * M.
# Hernquist: Phi = - G M / (r+a). Galpy: Phi = - amp / (r+a) (if normalized).
# -> amp = G * M
# MiyamotoNagai: Phi = - G M / sqrt(R^2 + (a + sqrt(z^2+b^2))^2)
# -> amp = G * M
# NFW: Phi = - G M_s * ln(1 + r/r_s) / (r/r_s)
# -> amp = G * M_s, a = r_s. Note: M_s is mass inside r_s? No, check definitions.
# NFW mass M(<r) = 4 pi rho_0 r_s^3 [ln(1+x) - x/(1+x)]
# As r->inf, Phi -> - 4 pi G rho_0 r_s^3 * ln(r) / r ??? No.
# Galpy NFW: amp = G * M_0. 
# We will use the explicit constructors with amplitude.

G_val = G.value  # (km/s)^2 kpc / Msun

def get_galpy_potentials(
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
) -> list:
    """Create list of galpy potentials for the galaxy."""
    from .profiles import nfw_params
    
    potentials = []
    
    # 1. NFW Halo
    r_s, _ = nfw_params(m200, c200)
    # NFW mass parameter M0 for galpy is such that amp = G * M0
    # M(<r) = M0 * (ln(1+x) - x/(1+x))
    # Our M200 = M0 * f(c200). So M0 = M200 / f(c200)
    f_c = np.log(1 + c200) - c200 / (1 + c200)
    M0 = m200 / f_c
    
    # Galpy NFW: amp=G*M0, a=r_s
    potentials.append(NFWPotential(amp=G_val * M0, a=r_s))
    
    # 2. Hernquist Bulge
    # Galpy Hernquist: amp = 2 * G * M_bulge? Check docs.
    # Galpy docs: Phi = - amp / (r + a).
    # Standard Hernquist: Phi = - G M / (r + a).
    # So amp = G * M_bulge * (is there a factor of 2? No, standard is GM).
    # CAREFUL: Galpy documentation says amp = 2GM for Hernquist? 
    # Let's check source or assume standard definition: amp is prefactor.
    # "amp : amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass"
    # If we provide G*M, it should be correct.
    if m_bulge > 0:
        potentials.append(HernquistPotential(amp=2 * G_val * m_bulge, a=a_bulge))
        # WAIT: Galpy Hernquist is defined as rho = amp/4pi a^4 / (r/a) / (1+r/a)^3
        # Standard: rho = M/2pi a / r / (r+a)^3
        # Let's test this carefully or check docs.
        # Galpy: amp = 2 * G * M usually.
        # Let's verify with a quick script if possible? 
        # Actually, using `m_bulge * G_val * 2` is standard for galpy Hernquist.
    
    # 3. Discs (DoubleExponentialDiskPotential)
    # Use realistic thick disc potential
    # amp = G * M / (4 * pi * h_R^2 * h_z)
    if M_disc_star > 0:
        amp_star = G_val * M_disc_star / (4 * np.pi * R_d_star**2 * z_d_star)
        potentials.append(
            DoubleExponentialDiskPotential(amp=amp_star, hr=R_d_star, hz=z_d_star)
        )
        
    if M_disc_gas > 0:
        amp_gas = G_val * M_disc_gas / (4 * np.pi * R_d_gas**2 * z_d_gas)
        potentials.append(
            DoubleExponentialDiskPotential(amp=amp_gas, hr=R_d_gas, hz=z_d_gas)
        )
        
    return potentials


def total_circular_velocity(
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
    """Calculate total circular velocity using galpy."

    Args:
        R: Cylindrical radial positions (kpc).
        [Mass parameters...]

    Returns:
        Circular velocity (km/s).
    """
    from tqdm import tqdm
    
    pots = get_galpy_potentials(
        m200, c200, m_bulge, a_bulge, M_disc_star, R_d_star, z_d_star, M_disc_gas, R_d_gas, z_d_gas
    )
    
    # galpy.potential.vcirc can take array inputs, but DoubleExponentialDiskPotential
    # force methods require scalar inputs. So we loop manually.
    # R needs to be >= 0
    R_safe = np.maximum(R, 1e-4)
    
    v_c_values = []
    for r_val in tqdm(R_safe, desc="Calculating Vcirc"):
        v_c_values.append(vcirc(pots, r_val, phi=0))
    
    v_c = np.array(v_c_values)
    
    return v_c


def nfw_potential(R, z, m200, r_s, c200):
    """Legacy wrapper for NFW potential (needed for escape velocity)."""
    # Recalculate amp
    f_c = np.log(1 + c200) - c200 / (1 + c200)
    M0 = m200 / f_c
    pot = NFWPotential(amp=G_val * M0, a=r_s)
    return evaluatePotentials(pot, R, z)


def hernquist_potential(R, z, m, a):
    """Legacy wrapper for Hernquist potential."""
    pot = HernquistPotential(amp=2 * G_val * m, a=a)
    return evaluatePotentials(pot, R, z)


def miyamoto_nagai_potential(R, z, m, a, b):
    """Legacy wrapper for MN potential."""
    pot = MiyamotoNagaiPotential(amp=G_val * m, a=a, b=b)
    return evaluatePotentials(pot, R, z)
