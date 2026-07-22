"""Tests for physics modules."""

import numpy as np

from swift_spiral_ics.physics import kinematics, perturbations, potentials, profiles
from swift_spiral_ics.physics.sampling import _cap_speed_at_escape_fraction


class TestProfiles:
    """Test density profile functions."""

    def test_nfw_params(self):
        """Test NFW parameter calculation."""
        m200 = 1e12  # Msun
        c200 = 10.0

        r_s, delta_c = profiles.nfw_params(m200, c200)

        assert r_s > 0
        assert delta_c > 0
        assert isinstance(r_s, float)
        assert isinstance(delta_c, float)

    def test_nfw_density_positive(self):
        """Test NFW density is positive."""
        r = np.linspace(0.1, 100, 50)
        m200 = 1e12
        c200 = 10.0
        r_s, delta_c = profiles.nfw_params(m200, c200)

        rho = profiles.nfw_density(r, m200, c200, delta_c, r_s)

        assert np.all(rho > 0)
        assert np.all(np.isfinite(rho))

    def test_hernquist_mass_enclosed(self):
        """Test Hernquist enclosed mass at infinity equals total mass."""
        m_bulge = 1e10
        a = 1.0
        r_large = 1000.0

        m_enc = profiles.hernquist_mass(r_large, m_bulge, a)

        assert np.isclose(m_enc, m_bulge, rtol=0.01)

    def test_exponential_disc_mass_conservation(self):
        """Test exponential disc mass conservation."""
        M_disc = 1e10
        R_d = 3.0
        R_large = 100.0

        m_enc = profiles.exponential_disc_mass(R_large, M_disc, R_d)

        assert np.isclose(m_enc, M_disc, rtol=0.01)


class TestKinematics:
    """Test kinematic functions."""

    def test_circular_velocity_positive(self):
        """Test circular velocity is positive."""
        R = np.linspace(0.5, 20, 20)

        v_c = potentials.total_circular_velocity(
            R,
            m200=1e12,
            c200=10.0,
            m_bulge=1e10,
            a_bulge=1.0,
            M_disc_star=1e10,
            R_d_star=3.0,
            z_d_star=0.3,
            M_disc_gas=1e9,
            R_d_gas=5.0,
            z_d_gas=0.1,
        )

        assert np.all(v_c > 0)
        assert np.all(np.isfinite(v_c))

    def test_toomre_q_dispersion(self):
        """Test Toomre Q dispersion calculation."""
        R = np.array([1.0, 5.0, 10.0])
        v_c = np.array([150.0, 200.0, 180.0])
        sigma_surf = np.array([1000.0, 500.0, 100.0])
        Q_target = 1.5

        sigma_R = kinematics.toomre_q_dispersion(R, v_c, sigma_surf, Q_target)

        assert np.all(sigma_R > 0)
        assert np.all(np.isfinite(sigma_R))

    def test_velocity_cap_avoids_near_escape_pileup(self):
        """Truncation does not place particles just below escape speed."""

        vx = np.array([100.0, 300.0])
        vy = np.array([0.0, 0.0])
        vz = np.array([0.0, 0.0])
        v_escape = np.array([100.0, 100.0])

        vx, vy, vz = _cap_speed_at_escape_fraction(vx, vy, vz, v_escape)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)

        assert np.all(speed <= 0.9 * v_escape)


class TestPerturbations:
    """Test spiral and bar perturbation functions."""

    def test_spiral_phase_range(self):
        """Test spiral phase is in valid range."""
        R = np.linspace(1, 15, 100)
        phi = np.linspace(0, 2 * np.pi, 100)

        phase = perturbations.spiral_arm_phase(R, phi, n_arms=2, pitch_deg=15.0)

        assert np.all(phase >= 0)
        assert np.all(phase < 2 * np.pi)

    def test_spiral_density_modulation_bounds(self):
        """Test spiral density modulation is reasonable."""
        R = np.linspace(1, 15, 100)
        phi = np.linspace(0, 2 * np.pi, 100)

        modulation = perturbations.spiral_density_modulation(
            R, phi, arm_strength=0.3, n_arms=2, pitch_deg=15.0
        )

        assert np.all(modulation > 0)
        assert np.all(modulation < 2.0)  # Should not exceed 1 + arm_strength

    def test_bar_density_modulation_bounds(self):
        """Test bar density modulation is reasonable."""
        R = np.linspace(0.1, 10, 100)
        phi = np.linspace(0, 2 * np.pi, 100)

        modulation = perturbations.bar_density_modulation(
            R, phi, bar_strength=0.5, bar_radius=3.0, bar_q=0.3
        )

        assert np.all(modulation > 0)
        assert np.all(modulation < 2.0)
