"""Physical constants used throughout the package, using unyt."""

import unyt
from unyt import Msun, km, kpc, s

# Define the internal unit system for the project
# This matches the typical Gadget/SWIFT galaxy simulation units
internal_system = unyt.UnitSystem(
    "galaxy_sim",
    length_unit=kpc,
    mass_unit=Msun,
    time_unit=0.9778 * unyt.Gyr,  # Derived so that velocity is km/s
)

# Gravitational constant in (km/s)^2 kpc / Msun
# We compute this directly from unyt to be precise
G = unyt.G.in_units((km / s) ** 2 * kpc / Msun)

# Boltzmann constant
k_B = unyt.kb.in_units("erg/K")

# Proton mass
m_p = unyt.mp.in_units("g")

# Hubble constant reference (km/s/Mpc)
H0_val = 67.77
H0 = H0_val * (km / s / unyt.Mpc)
