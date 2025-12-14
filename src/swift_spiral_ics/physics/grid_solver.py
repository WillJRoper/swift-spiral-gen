"""Python wrapper for the C++ grid-based Poisson solver."""

import numpy as np
import swift_spiral_ics.physics._grid_solver_cpp as _grid_solver_cpp
from .constants import G


class GalaxyGridSolver:
    def __init__(self, R_grid, z_grid, eps,
                 m200, c200, m_bulge, a_bulge, M_disc_star, R_d_star, z_d_star, M_disc_gas, R_d_gas, z_d_gas):
        """
        Initializes the Python wrapper for the C++ grid solver.

        Args:
            R_grid (np.ndarray): 1D array of radial grid points (kpc).
            z_grid (np.ndarray): 1D array of vertical grid points (kpc).
            eps (float): Gravitational softening length (kpc).
            [All galaxy mass parameters]
        """
        self._cpp_solver = _grid_solver_cpp.GridSolverCpp(
            R_grid, z_grid, eps,
            m200, c200, m_bulge, a_bulge, M_disc_star, R_d_star, z_d_star, M_disc_gas, R_d_gas, z_d_gas
        )
        self.R_grid = R_grid
        self.z_grid = z_grid
        self.eps = eps
        
        # Store galaxy parameters (for convenience/forwarding to other funcs)
        self.m200 = m200
        self.c200 = c200
        self.m_bulge = m_bulge
        self.a_bulge = a_bulge
        self.M_disc_star = M_disc_star
        self.R_d_star = R_d_star
        self.z_d_star = z_d_star
        self.M_disc_gas = M_disc_gas
        self.R_d_gas = R_d_gas
        self.z_d_gas = z_d_gas

    def bin_particles_to_grid(self, all_pos_dict, all_mass_dict):
        """Delegates to C++ solver."""
        self._cpp_solver.bin_particles_to_grid(all_pos_dict, all_mass_dict)

    def compute_potential_grid(self):
        """Delegates to C++ solver."""
        self._cpp_solver.compute_potential_grid()
        
    def get_potential_and_forces(self, R, z):
        """Delegates to C++ solver."""
        return self._cpp_solver.get_potential_and_forces(R, z)
        
    @property
    def Phi_grid(self):
        return self._cpp_solver.Phi_grid
        
    @property
    def FR_grid(self):
        return self._cpp_solver.FR_grid

    @property
    def FZ_grid(self):
        return self._cpp_solver.FZ_grid

    @property
    def rho_grid(self):
        return self._cpp_solver.rho_grid