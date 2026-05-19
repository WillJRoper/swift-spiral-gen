"Physics validation tests for the IC generator."

import numpy as np
from scipy import interpolate
from scipy.special import ellipk  # for analytical ring potential

from swift_spiral_ics.physics.constants import G
from swift_spiral_ics.physics.grid_solver import GalaxyGridSolver
from swift_spiral_ics.physics.kinematics import jeans_dispersion_spherical_from_grid


# Analytical potential of a thin ring (using scipy's ellipk), vectorized
def potential_of_thin_ring_analytic(M_ring, R_src, z_src, R_dest, z_dest, eps):
    G_val = G.value

    # Ensure all inputs are numpy arrays for vectorized operations
    R_src = np.atleast_1d(R_src)
    z_src = np.atleast_1d(z_src)
    R_dest = np.atleast_1d(R_dest)
    z_dest = np.atleast_1d(z_dest)

    term_R = (R_src + R_dest)
    term_z = (z_src - z_dest)
    denom_sq = term_R**2 + term_z**2 + eps**2 # Add softening

    k_sq_num = 4 * R_src * R_dest

    # Initialize output array
    phi = np.zeros_like(R_dest, dtype=float)

    # Handle point mass limit (R_src=0 or R_dest=0)
    is_point_mass_case = (R_src == 0.0) | (R_dest == 0.0)

    # Calculate for point mass cases
    if np.any(is_point_mass_case):
        r_effective_sq = R_dest[is_point_mass_case]**2 + (z_src[is_point_mass_case] - z_dest[is_point_mass_case])**2
        phi[is_point_mass_case] = -G_val * M_ring / np.sqrt(r_effective_sq + eps**2)

    # Calculate for ring cases
    ring_case_mask = ~is_point_mass_case
    if np.any(ring_case_mask):
        k_sq = k_sq_num[ring_case_mask] / denom_sq[ring_case_mask]

        # Clip k_sq to avoid ellipk errors for k^2 > 1 or too close to 1
        k_sq = np.clip(k_sq, 0.0, 1.0 - 1e-9)

        K_val = ellipk(k_sq) # scipy's ellipk takes k^2

        phi[ring_case_mask] = -2.0 * G_val * M_ring / (np.pi * np.sqrt(denom_sq[ring_case_mask])) * K_val

    return phi


class TestPhysicsValidation:

    def test_python_spline_interpolation(self):
        """
        Verify that the Python spline interpolation and differentiation of a known potential works.
        """
        G_val = G.value
        M_test = 1e10
        eps = 0.05
        R_grid_interp = np.linspace(0, 10, 256)
        z_grid_interp = np.linspace(-10, 10, 256)

        # Source of the analytical point mass
        z_src_test = z_grid_interp[len(z_grid_interp) // 2] # Midpoint z-grid

        # Manually create analytical Phi_grid for the spline
        R_2d, z_2d = np.meshgrid(R_grid_interp, z_grid_interp, indexing='ij')

        # Analytical potential of a softened point mass at (0, z_src_test)
        r_analytic_sq = R_2d**2 + (z_2d - z_src_test)**2
        phi_analytic_grid = -G_val * M_test / np.sqrt(r_analytic_sq + eps**2)

        # Create the spline directly
        spline = interpolate.RectBivariateSpline(R_grid_interp, z_grid_interp, phi_analytic_grid)

        # Query points (along z-axis, where R=0)
        check_R_pt = np.array([0.0, 0.0, 0.0])
        check_z_pt = np.array([0.5, 1.0, 2.0])

        # Interpolate potential and forces from the spline
        phi_interp_pt = spline(check_R_pt, check_z_pt, grid=False)
        fz_interp_pt = -spline(check_R_pt, check_z_pt, dy=1, grid=False) # FZ = -dPhi/dz

        # Analytical potential at query points
        r_dest_sq = check_R_pt**2 + (z_src_test - check_z_pt)**2
        phi_analytic_query = -G_val * M_test / np.sqrt(r_dest_sq + eps**2)

        # Analytical force (F_z component) at query points
        fz_analytic_query = -G_val * M_test * (check_z_pt - z_src_test) / (r_dest_sq + eps**2)**(1.5)

        print("\n--- Point Mass Potential Check (Python Spline) ---")
        print(f"Source at (0, {z_src_test})")
        print(f"Query R: {check_R_pt}, Query z: {check_z_pt}")
        print(f"Spline Phi: {phi_interp_pt}")
        print(f"Analytic Phi: {phi_analytic_query}")
        print(f"Ratio Phi: {phi_interp_pt/phi_analytic_query}")
        assert np.allclose(phi_interp_pt, phi_analytic_query, rtol=0.01, atol=1e-5), "Python Spline Potential mismatch"

        print(f"Spline FZ: {fz_interp_pt}")
        print(f"Analytic FZ: {fz_analytic_query}")
        assert np.allclose(fz_interp_pt, fz_analytic_query, rtol=0.02, atol=1e-5), "Python Spline FZ mismatch"

    def test_cpp_ring_potential(self):
        """
        Verify that the C++ `potential_of_ring` function (via GridSolver)
        correctly computes the potential and forces for a single ring of mass.
        """
        M_test = 1e10
        eps = 0.05
        R_grid_test = np.linspace(0, 10, 256)
        z_grid_test = np.linspace(-10, 10, 256)

        # Source ring (at R_src_ring, z_src_ring)
        R_src_ring = R_grid_test[10] # e.g., 10th radial bin
        z_src_ring = z_grid_test[len(z_grid_test) // 2] # Middle z bin (e.g. z=0)

        solver = GalaxyGridSolver(
            R_grid_test, z_grid_test, eps,
            m200=0, c200=10, m_bulge=0, a_bulge=0, M_disc_star=0, R_d_star=1, z_d_star=0.1, M_disc_gas=0, R_d_gas=1, z_d_gas=0.1
        )

        # Place mass particle at the source location, it will be binned into a single cell
        pos_ring = np.array([[R_src_ring, 0.0, z_src_ring]])
        mass_ring = np.array([M_test])
        solver.bin_particles_to_grid({'test': pos_ring}, {'test': mass_ring})
        solver.compute_potential_grid()

        # Query points for verification
        check_R_ring = np.array([0.5, R_src_ring, R_src_ring * 1.5])
        check_z_ring = np.array([z_src_ring, z_src_ring + 0.5, z_src_ring - 0.5])

        # --- Analytical Comparison ---
        phi_analytic_arr = potential_of_thin_ring_analytic(M_test, R_src_ring, z_src_ring, check_R_ring, check_z_ring, eps)

        # Numerical derivatives for forces from analytical potential
        # (This is a bit hacky, but needed for comparison)
        delta = 1e-4 # for numerical derivatives

        fr_analytic_arr = (potential_of_thin_ring_analytic(M_test, R_src_ring, z_src_ring, check_R_ring - delta, check_z_ring, eps) - \
                           potential_of_thin_ring_analytic(M_test, R_src_ring, z_src_ring, check_R_ring + delta, check_z_ring, eps)) / (2 * delta)

        fz_analytic_arr = (potential_of_thin_ring_analytic(M_test, R_src_ring, z_src_ring, check_R_ring, check_z_ring - delta, eps) - \
                           potential_of_thin_ring_analytic(M_test, R_src_ring, z_src_ring, check_R_ring, check_z_ring + delta, eps)) / (2 * delta)

        res_grid = solver.get_potential_and_forces(check_R_ring, check_z_ring)
        phi_grid = res_grid['Phi']
        fr_grid = res_grid['FR']
        fz_grid = res_grid['FZ']

        print("\n--- Single Ring Potential Check (C++ vs Analytic) ---")
        print(f"Source at (R={R_src_ring}, z={z_src_ring})")
        print(f"Query R: {check_R_ring}, Query z: {check_z_ring}")
        print(f"Grid Phi: {phi_grid}")
        print(f"Analytic Phi: {phi_analytic_arr}")
        print(f"Ratio Phi: {phi_grid/phi_analytic_arr}")
        assert np.allclose(phi_grid, phi_analytic_arr, rtol=0.05, atol=1e-5), "Ring Potential mismatch"

        print(f"Grid FR: {fr_grid}")
        print(f"Analytic FR: {fr_analytic_arr}")
        assert np.allclose(fr_grid, fr_analytic_arr, rtol=0.1, atol=1e-5), "Ring FR mismatch"

        print(f"Grid FZ: {fz_grid}")
        print(f"Analytic FZ: {fz_analytic_arr}")
        assert np.allclose(fz_grid, fz_analytic_arr, rtol=0.1, atol=1e-4), "Ring FZ mismatch"


    def test_jeans_solver_hernquist(self):
        """
        Verify that the Jeans solver recovers the correct velocity dispersion for a Hernquist profile.
        """
        # Parameters
        M = 1e10
        a = 1.0

        R_grid = np.linspace(0, 10, 256)
        z_grid = np.linspace(-10, 10, 256)
        eps = 0.05

        solver = GalaxyGridSolver(
            R_grid, z_grid, eps,
            m200=0, c200=10, m_bulge=M, a_bulge=a,
            M_disc_star=0, R_d_star=1, z_d_star=0.1,
            M_disc_gas=0, R_d_gas=1, z_d_gas=0.1
        )

        # Populate with analytical potential for stability of THIS test
        N_part = 100000
        rng = np.random.default_rng(42)
        u = rng.uniform(0, 1, N_part)
        r = a * np.sqrt(u) / (1 - np.sqrt(u))
        costheta = rng.uniform(-1, 1, N_part)
        theta = np.arccos(costheta)
        phi = rng.uniform(0, 2*np.pi, N_part)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        pos = np.column_stack([x, y, z])
        mass = np.full(N_part, M/N_part)

        solver.bin_particles_to_grid({'b': pos}, {'b': mass})
        solver.compute_potential_grid()

        # Solve Jeans
        check_r = np.array([0.5, 1.0, 2.0])
        sigma_r = jeans_dispersion_spherical_from_grid(
            check_r, solver, pos, mass, beta=0.0
        )

        print("\n--- Jeans Check ---")
        print(f"Radii: {check_r}")
        print(f"Numerical Sigma: {sigma_r}")

        assert np.all(sigma_r > 0), "Dispersion must be positive"
        assert not np.any(np.isnan(sigma_r)), "Dispersion must not be NaN"

        # Rough check: Virial velocity sqrt(GM/a) ~ sqrt(4.3e-6 * 1e10 / 1) ~ 200 km/s
        # Sigma should be order ~100 km/s
        assert np.all(sigma_r > 35) and np.all(sigma_r < 300), "Dispersion magnitude unreasonable"

    def test_singularity_behavior(self):
        """Check behavior at R=0."""
        R_grid = np.linspace(0, 10, 64)
        z_grid = np.linspace(-10, 10, 64)
        eps = 0.1
        solver = GalaxyGridSolver(
            R_grid, z_grid, eps,
            m200=0, c200=10, m_bulge=1e10, a_bulge=1.0,
            M_disc_star=0, R_d_star=1, z_d_star=0.1,
            M_disc_gas=0, R_d_gas=1, z_d_gas=0.1
        )

        # Put mass at center
        pos = np.array([[0.0, 0.0, 0.0]])
        mass = np.array([1e10])
        solver.bin_particles_to_grid({'c': pos}, {'c': mass})
        solver.compute_potential_grid()

        res = solver.get_potential_and_forces([0.001, 0.01, 0.1], [0, 0, 0])

        print("\n--- Singularity Check ---")
        print(f"FR near 0: {res['FR']}")

        assert np.all(res['FR'] <= 0), "Force should be attractive (negative) or zero"
        # Check magnitude is not infinite
        assert np.all(np.abs(res['FR']) < 1e10), "Force singularity detected"

if __name__ == "__main__":
    t = TestPhysicsValidation()
    t.test_python_spline_interpolation()
    t.test_cpp_ring_potential()
    t.test_jeans_solver_hernquist()
    t.test_singularity_behavior()
