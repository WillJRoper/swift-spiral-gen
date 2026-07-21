"""Integration tests for full IC generation pipeline."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from swift_spiral_ics.cli.generate import (
    _normalise_per_galaxy_args,
    _resolve_galaxy_placement,
)
from swift_spiral_ics.physics.sampling import (
    sample_exponential_disc,
    sample_hernquist_bulge,
    sample_nfw_halo,
)
from swift_spiral_ics.utils.random import get_rng


class TestFullPipeline:
    """Test complete IC generation pipeline."""

    def test_parabolic_orbit_placement_is_com_balanced(self):
        """Parabolic orbit mode computes two COM-balanced galaxy centres."""
        args = argparse.Namespace(
            n_galaxies=2,
            box_kpc=200.0,
            dm_mass_msun=[1.0e12, 2.0e12],
            star_mass_msun=[6.0e10, 1.0e11],
            gas_mass_msun=[1.0e10, 2.0e10],
            bulge_fraction=[0.2, 0.3],
            dm_part_mass_msun=1.0e9,
            star_part_mass_msun=1.0e8,
            gas_part_mass_msun=1.0e8,
            c200=[10.0, 10.0],
            bulge_a_kpc=[0.8, 1.0],
            bulge_rmax_scale=[50.0],
            stellar_disk_scale_length_kpc=[3.5, 5.0],
            stellar_disk_scale_height_kpc=[0.35, 0.5],
            Q_star=[2.0],
            gas_disk_scale_length_kpc=[7.0, 10.0],
            gas_disk_scale_height_kpc=[0.1, 0.15],
            Q_gas=[1.5],
            n_arms=[2],
            pitch_deg=[15.0],
            arm_strength=[0.15],
            arm_stream_frac=[0.02],
            bar_strength=[0.1],
            bar_radius=[3.0],
            bar_q=[0.3],
            bar_angle=[0.0],
            inclination_deg=None,
            xs=None,
            ys=None,
            zs=None,
            vxs=None,
            vys=None,
            vzs=None,
            orbit="parabolic",
            orbit_r_init_kpc=80.0,
            orbit_r_peri_kpc=10.0,
            orbit_plane_angle_deg=0.0,
        )

        _normalise_per_galaxy_args(args)
        positions, velocities = _resolve_galaxy_placement(args)
        masses = np.asarray([
            args.m200_msun[i] + args.m_star_msun[i] + args.m_bulge_msun[i] + args.m_gas_msun[i]
            for i in range(args.n_galaxies)
        ])

        assert np.isclose(np.linalg.norm(positions[1] - positions[0]), 80.0)
        assert np.allclose(np.average(positions, axis=0, weights=masses), 0.0)
        assert np.allclose(np.average(velocities, axis=0, weights=masses), 0.0)
        assert velocities[1, 0] < velocities[0, 0]

    def test_generate_tiny_galaxy(self):
        """Test generating a tiny galaxy (fast smoke test)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            yaml_file = Path(tmpdir) / "test_params.yml"

            # Run CLI with minimal parameters
            cmd = [
                "python",
                "-m",
                "swift_spiral_ics.cli.generate",
                "--out-ics",
                str(ic_file),
                "--out-params",
                str(yaml_file),
                "--seed",
                "42",
                "--box-kpc",
                "100",
                "--n-galaxies",
                "1",
                "--dm-mass-msun",
                "1.1e11",
                "--dm-part-mass-msun",
                "1e9",
                "--star-mass-msun",
                "7e9",
                "--bulge-fraction",
                "0.1428571429",
                "--star-part-mass-msun",
                "1e8",
                "--gas-mass-msun",
                "2e9",
                "--gas-part-mass-msun",
                "5e7",
                "--c200",
                "10",
                "--max-timestep-gyr",
                "0.8",
                "--stellar-disk-scale-length-kpc",
                "3",
                "--stellar-disk-scale-height-kpc",
                "0.3",
                "--gas-disk-scale-length-kpc",
                "5",
                "--gas-disk-scale-height-kpc",
                "0.1",
                "--bulge-a-kpc",
                "1",
                "--time-end-gyr",
                "1.0",
                "--snapshot-dt-myr",
                "50",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            # Check that files were created
            assert ic_file.exists(), (
                f"IC file not created. Stdout: {result.stdout}, Stderr: {result.stderr}"
            )
            assert yaml_file.exists()

            # Validate IC file structure
            with h5py.File(ic_file, "r") as f:
                assert "Header" in f
                assert "Units" in f
                assert "PartType0" in f or "PartType1" in f or "PartType4" in f

    def test_multi_galaxy_positions_are_literal_box_coordinates(self):
        """Per-galaxy positions are interpreted as literal coordinates in the box."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            yaml_file = Path(tmpdir) / "test_params.yml"

            cmd = [
                sys.executable,
                "-m",
                "swift_spiral_ics.cli.generate",
                "--out-ics",
                str(ic_file),
                "--out-params",
                str(yaml_file),
                "--seed",
                "42",
                "--box-kpc",
                "100",
                "--n-galaxies",
                "2",
                "--xs",
                "10",
                "90",
                "--ys",
                "50",
                "50",
                "--zs",
                "50",
                "50",
                "--dm-mass-msun",
                "1e9",
                "1e9",
                "--dm-part-mass-msun",
                "1e9",
                "--star-mass-msun",
                "1e8",
                "1e8",
                "--bulge-fraction",
                "0.0",
                "0.0",
                "--star-part-mass-msun",
                "1e8",
                "--gas-mass-msun",
                "1e8",
                "1e8",
                "--gas-part-mass-msun",
                "1e8",
                "--c200",
                "10",
                "10",
                "--stellar-disk-scale-length-kpc",
                "1.0",
                "1.0",
                "--stellar-disk-scale-height-kpc",
                "0.1",
                "0.1",
                "--gas-disk-scale-length-kpc",
                "1.0",
                "1.0",
                "--gas-disk-scale-height-kpc",
                "0.1",
                "0.1",
                "--bulge-a-kpc",
                "0.5",
                "0.5",
                "--nR-grid",
                "16",
                "--nz-grid",
                "16",
                "--eps-grid",
                "0.5",
                "--time-end-gyr",
                "0.01",
                "--snapshot-dt-myr",
                "5",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, result.stderr
            with h5py.File(ic_file, "r") as f:
                dm_x_kpc = f["PartType1/Coordinates"][:, 0] * 1000.0
                assert dm_x_kpc.min() < 20.0
                assert dm_x_kpc.max() > 80.0

    def test_generate_tiny_parabolic_merger(self):
        """The CLI can generate a two-galaxy parabolic merger without manual COM inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            yaml_file = Path(tmpdir) / "test_params.yml"

            cmd = [
                sys.executable,
                "-m",
                "swift_spiral_ics.cli.generate",
                "--out-ics",
                str(ic_file),
                "--out-params",
                str(yaml_file),
                "--seed",
                "42",
                "--box-kpc",
                "200",
                "--n-galaxies",
                "2",
                "--orbit",
                "parabolic",
                "--orbit-r-init-kpc",
                "40",
                "--orbit-r-peri-kpc",
                "5",
                "--dm-mass-msun",
                "1e9",
                "2e9",
                "--dm-part-mass-msun",
                "1e9",
                "--star-mass-msun",
                "1e8",
                "2e8",
                "--bulge-fraction",
                "0.0",
                "0.0",
                "--star-part-mass-msun",
                "1e8",
                "--gas-mass-msun",
                "1e8",
                "2e8",
                "--gas-part-mass-msun",
                "1e8",
                "--c200",
                "10",
                "10",
                "--stellar-disk-scale-length-kpc",
                "1.0",
                "1.0",
                "--stellar-disk-scale-height-kpc",
                "0.1",
                "0.1",
                "--gas-disk-scale-length-kpc",
                "1.0",
                "1.0",
                "--gas-disk-scale-height-kpc",
                "0.1",
                "0.1",
                "--bulge-a-kpc",
                "0.5",
                "0.5",
                "--nR-grid",
                "16",
                "--nz-grid",
                "16",
                "--eps-grid",
                "0.5",
                "--time-end-gyr",
                "0.01",
                "--snapshot-dt-myr",
                "5",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, result.stderr
            assert ic_file.exists()
            assert yaml_file.exists()
            with h5py.File(ic_file, "r") as f:
                assert f["Header"].attrs["NumPart_Total"][1] == 3

    def test_multi_galaxy_positions_must_lie_inside_box(self):
        """Out-of-box galaxy coordinates are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            yaml_file = Path(tmpdir) / "test_params.yml"

            cmd = [
                sys.executable,
                "-m",
                "swift_spiral_ics.cli.generate",
                "--out-ics",
                str(ic_file),
                "--out-params",
                str(yaml_file),
                "--box-kpc",
                "100",
                "--n-galaxies",
                "2",
                "--xs",
                "10",
                "110",
                "--ys",
                "50",
                "50",
                "--zs",
                "50",
                "50",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode != 0
            assert "must lie within 0 and --box-kpc" in result.stderr

    def test_random_background_radius_limits_gas_extent(self):
        """Random background gas can be restricted to a sphere around the box centre."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            yaml_file = Path(tmpdir) / "test_params.yml"

            cmd = [
                sys.executable,
                "-m",
                "swift_spiral_ics.cli.generate",
                "--out-ics",
                str(ic_file),
                "--out-params",
                str(yaml_file),
                "--seed",
                "42",
                "--box-kpc",
                "100",
                "--n-galaxies",
                "1",
                "--dm-mass-msun",
                "1e9",
                "--dm-part-mass-msun",
                "1e9",
                "--star-mass-msun",
                "1e8",
                "--bulge-fraction",
                "0.0",
                "--star-part-mass-msun",
                "1e8",
                "--gas-mass-msun",
                "1e8",
                "--gas-part-mass-msun",
                "1e7",
                "--c200",
                "10",
                "--stellar-disk-scale-length-kpc",
                "1.0",
                "--stellar-disk-scale-height-kpc",
                "0.1",
                "--gas-disk-scale-length-kpc",
                "1.0",
                "--gas-disk-scale-height-kpc",
                "0.1",
                "--bulge-a-kpc",
                "0.5",
                "--nR-grid",
                "16",
                "--nz-grid",
                "16",
                "--eps-grid",
                "0.5",
                "--time-end-gyr",
                "0.01",
                "--snapshot-dt-myr",
                "5",
                "--bg-gas-density-msun-kpc3",
                "1e4",
                "--bg-grid-kpc",
                "0",
                "--bg-radius-kpc",
                "20",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, result.stderr
            with h5py.File(ic_file, "r") as f:
                coords_kpc = f["PartType0/Coordinates"][:] * 1000.0
                centered = coords_kpc - 50.0
                radius = (centered**2).sum(axis=1) ** 0.5
                assert radius.max() <= 20.1

    def test_grid_background_radius_limits_gas_extent(self):
        """Grid background gas can be restricted to a sphere around the box centre."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            yaml_file = Path(tmpdir) / "test_params.yml"

            cmd = [
                sys.executable,
                "-m",
                "swift_spiral_ics.cli.generate",
                "--out-ics",
                str(ic_file),
                "--out-params",
                str(yaml_file),
                "--seed",
                "42",
                "--box-kpc",
                "100",
                "--n-galaxies",
                "1",
                "--dm-mass-msun",
                "1e9",
                "--dm-part-mass-msun",
                "1e9",
                "--star-mass-msun",
                "1e8",
                "--bulge-fraction",
                "0.0",
                "--star-part-mass-msun",
                "1e8",
                "--gas-mass-msun",
                "1e8",
                "--gas-part-mass-msun",
                "1e7",
                "--c200",
                "10",
                "--stellar-disk-scale-length-kpc",
                "1.0",
                "--stellar-disk-scale-height-kpc",
                "0.1",
                "--gas-disk-scale-length-kpc",
                "1.0",
                "--gas-disk-scale-height-kpc",
                "0.1",
                "--bulge-a-kpc",
                "0.5",
                "--nR-grid",
                "16",
                "--nz-grid",
                "16",
                "--eps-grid",
                "0.5",
                "--time-end-gyr",
                "0.01",
                "--snapshot-dt-myr",
                "5",
                "--bg-gas-density-msun-kpc3",
                "1e4",
                "--bg-grid-kpc",
                "10",
                "--bg-radius-kpc",
                "20",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, result.stderr
            with h5py.File(ic_file, "r") as f:
                coords_kpc = f["PartType0/Coordinates"][:] * 1000.0
                centered = coords_kpc - 50.0
                radius = (centered**2).sum(axis=1) ** 0.5
                assert radius.max() <= 21.5


class TestSampling:
    """Test particle sampling functions."""

    def test_sample_nfw_halo_count(self):
        """Test NFW halo sampling produces correct particle count."""
        rng = get_rng(42)
        N = 100
        m200 = 1e12
        c200 = 10.0
        r_max = 100.0

        x, y, z = sample_nfw_halo(N, m200, c200, r_max, rng)

        assert len(x) == N
        assert len(y) == N
        assert len(z) == N

    def test_sample_hernquist_bulge_count(self):
        """Test Hernquist bulge sampling produces correct particle count."""
        rng = get_rng(42)
        N = 50
        m_bulge = 1e10
        a = 1.0

        x, y, z = sample_hernquist_bulge(N, m_bulge, a, rng)

        assert len(x) == N
        assert len(y) == N
        assert len(z) == N

    def test_sample_exponential_disc_count(self):
        """Test exponential disc sampling produces correct particle count."""
        rng = get_rng(42)
        N = 100
        M_disc = 1e10
        R_d = 3.0
        z_d = 0.3

        x, y, z = sample_exponential_disc(N, M_disc, R_d, z_d, rng)

        assert len(x) == N
        assert len(y) == N
        assert len(z) == N

    def test_sample_disc_with_spirals(self):
        """Test disc sampling with spiral arms."""
        rng = get_rng(42)
        N = 100
        M_disc = 1e10
        R_d = 3.0
        z_d = 0.3

        spiral_params = {
            "n_arms": 2,
            "pitch_deg": 15.0,
            "arm_strength": 0.3,
        }

        x, y, z = sample_exponential_disc(N, M_disc, R_d, z_d, rng, spiral_params=spiral_params)

        assert len(x) == N
        assert len(y) == N
        assert len(z) == N

    def test_sample_disc_with_bar(self):
        """Test disc sampling with bar."""
        rng = get_rng(42)
        N = 100
        M_disc = 1e10
        R_d = 3.0
        z_d = 0.3

        bar_params = {
            "enabled": True,
            "radius": 3.0,
            "q": 0.3,
            "strength": 0.5,
            "angle": 0.0,
        }

        x, y, z = sample_exponential_disc(N, M_disc, R_d, z_d, rng, bar_params=bar_params)

        assert len(x) == N
        assert len(y) == N
        assert len(z) == N


class TestMassConservation:
    """Test mass conservation in IC generation."""

    def test_requested_vs_achieved_mass(self):
        """Test that achieved mass is close to requested mass."""
        M_requested = 1e10  # Msun
        m_part = 1e7  # Msun

        N = int(round(M_requested / m_part))
        M_achieved = N * m_part

        # Should be within half a particle mass
        assert abs(M_achieved - M_requested) <= 0.5 * m_part
