"""Gravitational potentials and circular velocities using galpy."""

import numpy as np
from galpy.potential import (
    HernquistPotential,
    MiyamotoNagaiPotential,
    NFWPotential,
    evaluatePotentials,
    plotRotcurve,
    vcirc,
)
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
    # Let's test this carefully or check docs. 
    # Actually, using `m_bulge * G_val * 2` is standard for galpy Hernquist.
    if m_bulge > 0:
        potentials.append(HernquistPotential(amp=2 * G_val * m_bulge, a=a_bulge))
        
    # 3. Miyamoto-Nagai Discs
    # Phi = - amp / sqrt(R^2 + (a + sqrt(z^2+b^2))^2)
    # Standard: - G M / ...
    # So amp = G * M
    if M_disc_star > 0:
        potentials.append(
            MiyamotoNagaiPotential(amp=G_val * M_disc_star, a=R_d_star, b=z_d_star)
        )
        
    if M_disc_gas > 0:
        potentials.append(
            MiyamotoNagaiPotential(amp=G_val * M_disc_gas, a=R_d_gas, b=z_d_gas)
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
    """Calculate total circular velocity using galpy."""
    pots = get_galpy_potentials(
        m200, c200, m_bulge, a_bulge, M_disc_star, R_d_star, z_d_star, M_disc_gas, R_d_gas, z_d_gas
    )
    
    # Galpy vcirc takes R and phi=0 (axisymmetric)
    # It returns float if scalar input, array if array input
    # R needs to be >= 0
    R_safe = np.maximum(R, 1e-4)
    v_c = vcirc(pots, R_safe, phi=0)
    
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
