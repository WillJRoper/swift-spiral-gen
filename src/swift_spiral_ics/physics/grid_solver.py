"""Python wrapper for the C++ grid-based Poisson solver."""

import numpy as np
from scipy import interpolate

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

        self._spline_Phi = None

    def bin_particles_to_grid(self, all_pos_dict, all_mass_dict):
        """Delegates to C++ solver."""
        self._cpp_solver.bin_particles_to_grid(all_pos_dict, all_mass_dict)

    def compute_potential_grid(self):
        """Delegates to C++ solver."""
        self._cpp_solver.compute_potential_grid()
        self._spline_Phi = None # Invalidate cache

    def get_potential_and_forces(self, R, z):
        """
        Interpolates potential and forces to given R, z coordinates using Bicubic Spline.
        """
        # Ensure R and z are numpy arrays
        R = np.atleast_1d(R)
        z = np.atleast_1d(z)

        # Create spline from the computed grid if not cached
        if self._spline_Phi is None:
            # Phi_grid is (nR, nz)
            # R_grid and z_grid are 1D arrays
            self._spline_Phi = interpolate.RectBivariateSpline(self.R_grid, self.z_grid, self.Phi_grid)

        # Interpolate Potential
        Phi = self._spline_Phi(R, z, grid=False)

        # Calculate Forces: F = -grad(Phi)
        # FR = -dPhi/dR
        FR = -self._spline_Phi(R, z, dx=1, grid=False)

        # FZ = -dPhi/dz
        FZ = -self._spline_Phi(R, z, dy=1, grid=False)

        return {"Phi": Phi, "FR": FR, "FZ": FZ}

    @property
    def Phi_grid(self):
        return self._cpp_solver.Phi_grid

    @property
    def rho_grid(self):
        return self._cpp_solver.rho_grid

    def get_cylindrical_force_profile(self, R_query):
        """
        Returns the radial force F_R(R, z=0) from the grid potential.
        """
        # Ensure R_query is a numpy array
        R_query = np.atleast_1d(R_query)

        # Interpolate FR at z=0
        FR_z0 = self._spline_Phi(R_query, 0.0, dx=1, grid=False) # FR = -dPhi/dR

        return -FR_z0 # Force in galpy is often -F_R, convert to a magnitude if needed.
                      # My get_potential_and_forces returns FR = -dPhi/dR, so it's the force component.
                      # We need the magnitude, so abs. Or careful use of sign.
                      # For v_c, we need R * |FR|.

    def get_spherical_force_profile(self, r_query):
        """
        Returns the spherically-averaged force F_r(r) from the grid potential.
        Approximated by evaluating FR at z=0 and assuming spherical for integral.
        """
        # Ensure r_query is a numpy array
        r_query = np.atleast_1d(r_query)

        # We need magnitude of force: F_r = -dPhi/dr.
        # From our grid, we have F_R and F_Z.
        # For a spherical system, F_r = FR. FZ=0.
        # So we can approximate F_r by FR at z=0.

        res = self.get_potential_and_forces(r_query, np.zeros_like(r_query))
        FR = res["FR"]
        # F_r should be negative (attractive).
        return FR # This is -dPhi/dR, so it's the radial component of force


    def get_component_density_profile(self, r_query, pos, mass, profile_type="spherical"):
        """
        Calculates the 1D density profile of a specific component from its particles.
        profile_type can be 'spherical' or 'cylindrical' (for surface density).
        """
        if pos.size == 0:
            return np.zeros_like(r_query)

        if profile_type == "spherical":
            r_part = np.sqrt(np.sum(pos**2, axis=1))
            # Use log bins for density to resolve center
            r_bins = np.geomspace(1e-4, np.max(r_part) * 1.1, 100)
            hist, _ = np.histogram(r_part, bins=r_bins, weights=mass)
            # Volume of shells
            vol = 4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
            # Avoid division by zero
            mask = vol > 0
            rho_prof = np.zeros_like(vol)
            rho_prof[mask] = hist[mask] / vol[mask]

            r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])

            # Interpolate to r_query
            return np.interp(r_query, r_centers, rho_prof, left=0, right=0)

        elif profile_type == "cylindrical":
            R_part = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
            R_bins = np.linspace(0, np.max(R_part) * 1.1, 100)
            hist, _ = np.histogram(R_part, bins=R_bins, weights=mass)
            # Area of annuli
            area = np.pi * (R_bins[1:]**2 - R_bins[:-1]**2)
            mask = area > 0
            Sigma_prof = np.zeros_like(area)
            Sigma_prof[mask] = hist[mask] / area[mask]

            R_centers = 0.5 * (R_bins[1:] + R_bins[:-1])

            # Interpolate to r_query
            return np.interp(r_query, R_centers, Sigma_prof, left=0, right=0)

        else:
            raise ValueError("Invalid profile_type")

    def get_component_density_on_grid(self, pos, mass):
        """Calculates density of a specific component on the grid."""
        # Convert to cylindrical R, z
        R_part = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
        z_part = pos[:,2]

        H, _, _ = np.histogram2d(R_part, z_part, bins=(self.R_grid, self.z_grid), weights=mass)

        # Normalize by volume (using same logic as C++ or simpler Python approximation)
        # Python histogram uses bin EDGES. self.R_grid are points?
        # My C++ code assumed R_grid were points and did nearest neighbor.
        # Here I should use bin centers/edges carefully.
        # Let's assume linear interpolation or simple density estimation.
        # For simplicity and robustness, let's use a KDE or just radial binning for 1D profiles?
        # For Jeans solver, we usually need 1D profiles (spherical or cylindrical).
        return H # Placeholder, logic below is better

    def get_spherical_dispersion(self, r_query, pos, mass, beta=0.0):
        """Solves spherical Jeans equation for a component."""
        # 1. Calculate spherical density profile of the component
        r_part = np.sqrt(np.sum(pos**2, axis=1))
        # Use log bins for density
        r_bins = np.geomspace(1e-3, max(np.max(r_part), 100), 100)
        hist, _ = np.histogram(r_part, bins=r_bins, weights=mass)
        # Volume of shells
        vol = 4/3 * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
        rho_prof = hist / vol
        r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])

        # 2. Get Total Force F_r(r) from Grid Potential
        # Assume spherical symmetry: F_r(r) approx F_R(R=r, z=0) from grid solver
        res = self.get_potential_and_forces(r_centers, np.zeros_like(r_centers))
        Fr_prof = res["FR"] # Negative (inward)

        # 3. Integrate Jeans Equation: d(rho sigma^2)/dr + 2 beta rho sigma^2 / r = rho Fr
        # Integral: rho(r) sigma^2(r) = int_r^inf rho(r') |Fr(r')| (r'/r)^(2beta) dr'
        # For beta=0: rho sigma^2 = int_r^inf rho |Fr| dr'

        pressure = np.zeros_like(r_centers)
        force_mag = np.abs(Fr_prof)

        # Integrate outwards-in
        # P(r) = integral_{r}^{inf} rho * F dr
        # Using cumulative sum from outside
        integrand = rho_prof * force_mag
        # Better: trapezoidal rule
        # integral[i] = sum_{j=i}^{N-1} 0.5*(y[j]+y[j+1]) * (x[j+1]-x[j])

        # Quick integration
        pressure_integral = 0.0
        for i in range(len(r_centers)-2, -1, -1):
            step = 0.5 * (integrand[i] + integrand[i+1]) * (r_centers[i+1] - r_centers[i])
            pressure_integral += step
            pressure[i] = pressure_integral

        sigma_sq = np.zeros_like(pressure)
        mask = rho_prof > 0
        sigma_sq[mask] = pressure[mask] / rho_prof[mask]

        # Interpolate to particle positions
        sigma_r = np.interp(r_query, r_centers, np.sqrt(sigma_sq))
        return sigma_r

    def get_cylindrical_kinematics(self, R_query, pos_comp, mass_comp, M_disc, R_d, z_d, Q_target, is_gas):
        """
        Calculates cylindrical kinematic profiles (v_c, sigma_R, sigma_phi, sigma_z, v_phi_mean)
        for a disc component.
        """
        from .kinematics import (
            disc_velocity_dispersions,
            epicyclic_frequency,
            gas_dispersion_from_temperature,
        )

        # Define a radial grid for solving. Cover the requested particle radii
        # instead of using a fixed 30 kpc edge, otherwise outer-disc particles all
        # inherit the same extrapolated kinematics.
        max_radius = max(30.0, 8.0 * R_d, float(np.max(R_query)) * 1.1 if len(R_query) else 30.0)
        R_unique = np.linspace(1e-3, max_radius, 1000)

        # 1. Get v_c from Grid Potential Total Force
        res = self.get_potential_and_forces(R_unique, np.zeros_like(R_unique))
        FR = res["FR"] # -dPhi/dR

        v_c_sq_profile = R_unique * np.abs(FR) # v_c^2 = R * |F_R|
        v_c_profile = np.sqrt(np.maximum(v_c_sq_profile, 0.0))

        # 2. Get Kappa from v_c_profile
        kappa_profile = epicyclic_frequency(R_unique, v_c_profile)

        # 3. Calculate the target smooth surface density for this disc. Using a
        # binned particle estimate here feeds Poisson noise into asymmetric drift
        # and can launch coherent stellar rings/bands at startup.
        Sigma_comp_prof = M_disc / (2.0 * np.pi * R_d**2) * np.exp(-R_unique / R_d)

        # 4. Calculate sigma_R
        if is_gas:
            # Use the vertical restoring force to set the gas pressure support for
            # the requested scale height. A fixed 1e4 K floor under-supports the
            # sampled disc and causes immediate vertical relaxation transients.
            z_support = np.full_like(R_unique, max(z_d, self.eps))
            support_force = np.abs(self.get_potential_and_forces(R_unique, z_support)["FZ"])
            sigma_R_prof = np.sqrt(np.maximum(z_d * support_force, 0.0))
            sigma_floor = gas_dispersion_from_temperature(1.0e4)
            sigma_ceiling = gas_dispersion_from_temperature(3.0e5)
            sigma_R_prof = np.clip(sigma_R_prof, sigma_floor, sigma_ceiling)
        else:
            # Toomre Q based: sigma_R = Q * pi * G * Sigma / kappa
            sigma_R_prof = Q_target * np.pi * G.value * Sigma_comp_prof / kappa_profile
            sigma_R_prof = np.maximum(sigma_R_prof, 5.0) # Floor dispersion

        # 5. Calculate sigma_phi, sigma_z
        if is_gas:
            sigma_phi_prof = sigma_R_prof
            sigma_z_prof = sigma_R_prof
        else:
            sigma_phi_prof, sigma_z_prof = disc_velocity_dispersions(R_unique, sigma_R_prof)
            z_support = np.full_like(R_unique, max(z_d, self.eps))
            support_force = np.abs(self.get_potential_and_forces(R_unique, z_support)["FZ"])
            sigma_z_support = np.sqrt(np.maximum(z_d * support_force, 0.0))
            sigma_z_prof = np.maximum(sigma_z_prof, sigma_z_support)

        # 6. Asymmetric Drift (v_phi_mean)
        # v_phi^2 = v_c^2 + sigma_R^2 * [ R * (dlnSigma/dR + dlnSigma_R^2/dR) + (1 - sigma_phi^2/sigma_R^2) ]

        # Gradient of log Surface Density (ensure non-zero)
        ln_Sigma = np.log(np.maximum(Sigma_comp_prof, 1e-10))
        dlnSigma_dR = np.gradient(ln_Sigma, R_unique)

        # Gradient of log sigma_R^2 (ensure non-zero)
        ln_sigma2 = np.log(np.maximum(sigma_R_prof**2, 1e-10))
        dlnsigma2_dR = np.gradient(ln_sigma2, R_unique)

        # Anisotropy term
        ratio_sq = (sigma_phi_prof / np.maximum(sigma_R_prof, 1e-10))**2 # Avoid div by zero
        term_anisotropy = 1.0 - ratio_sq

        bracket = R_unique * (dlnSigma_dR + dlnsigma2_dR) + term_anisotropy
        v_phi_sq_profile = v_c_sq_profile + sigma_R_prof**2 * bracket
        v_phi_prof = np.sqrt(np.maximum(v_phi_sq_profile, 0.0))

        # Interpolate to R_query (particle positions)
        v_c_out = np.interp(R_query, R_unique, v_c_profile)
        sigma_R_out = np.interp(R_query, R_unique, sigma_R_prof)
        sigma_phi_out = np.interp(R_query, R_unique, sigma_phi_prof)
        sigma_z_out = np.interp(R_query, R_unique, sigma_z_prof)
        v_phi_out = np.interp(R_query, R_unique, v_phi_prof)

        return v_c_out, sigma_R_out, sigma_phi_out, sigma_z_out, v_phi_out
