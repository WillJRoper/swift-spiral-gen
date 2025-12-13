"""Tests for I/O modules."""

import tempfile
from pathlib import Path

import h5py
import numpy as np

from swift_spiral_ics.io.swift_writer import write_swift_ic
from swift_spiral_ics.io.yaml_writer import generate_swift_params, write_yaml_file


class TestSwiftWriter:
    """Test SWIFT HDF5 writer."""

    def test_write_swift_ic_structure(self):
        """Test SWIFT IC file has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"

            # Create minimal particle data
            N = 100
            particle_data = {
                "dm": {
                    "pos": np.random.uniform(0, 10, (N, 3)),
                    "vel": np.random.uniform(-100, 100, (N, 3)),
                },
                "gas": {
                    "pos": np.random.uniform(0, 10, (N, 3)),
                    "vel": np.random.uniform(-100, 100, (N, 3)),
                },
                "stars": {
                    "pos": np.random.uniform(0, 10, (N, 3)),
                    "vel": np.random.uniform(-100, 100, (N, 3)),
                },
            }

            m_part = 1e6
            box_size = 100.0

            # Write IC file
            write_swift_ic(str(ic_file), box_size, particle_data, m_part)

            # Verify structure
            with h5py.File(ic_file, "r") as f:
                # Check Header
                assert "Header" in f
                assert "Dimension" in f["Header"].attrs
                assert "BoxSize" in f["Header"].attrs
                assert "NumPart_Total" in f["Header"].attrs
                assert "Flag_Entropy_ICs" in f["Header"].attrs

                # Check Units
                assert "Units" in f

                # Check particle groups
                assert "PartType0" in f  # Gas
                assert "PartType1" in f  # DM
                assert "PartType4" in f  # Stars

                # Check required datasets for gas
                assert "Coordinates" in f["PartType0"]
                assert "Velocities" in f["PartType0"]
                assert "Masses" in f["PartType0"]
                assert "ParticleIDs" in f["PartType0"]
                assert "InternalEnergy" in f["PartType0"]
                assert "SmoothingLength" in f["PartType0"]

                # Check dimensions
                assert f["PartType0/Coordinates"].shape == (N, 3)
                assert f["PartType0/Velocities"].shape == (N, 3)
                assert f["PartType0/Masses"].shape == (N,)

    def test_particle_ids_unique(self):
        """Test particle IDs are unique across all types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"

            N = 50
            particle_data = {
                "dm": {
                    "pos": np.random.uniform(0, 10, (N, 3)),
                    "vel": np.random.uniform(-100, 100, (N, 3)),
                },
                "gas": {
                    "pos": np.random.uniform(0, 10, (N, 3)),
                    "vel": np.random.uniform(-100, 100, (N, 3)),
                },
                "stars": {
                    "pos": np.random.uniform(0, 10, (N, 3)),
                    "vel": np.random.uniform(-100, 100, (N, 3)),
                },
            }

            write_swift_ic(str(ic_file), 100.0, particle_data, 1e6)

            # Check IDs are unique
            with h5py.File(ic_file, "r") as f:
                all_ids = []
                all_ids.extend(f["PartType0/ParticleIDs"][:])
                all_ids.extend(f["PartType1/ParticleIDs"][:])
                all_ids.extend(f["PartType4/ParticleIDs"][:])

                assert len(all_ids) == len(set(all_ids))  # All unique

    def test_coordinates_in_box(self):
        """Test all coordinates are within box bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"

            N = 100
            box_size = 50.0
            particle_data = {
                "gas": {
                    "pos": np.random.uniform(-10, 60, (N, 3)),  # Some outside box
                    "vel": np.random.uniform(-100, 100, (N, 3)),
                },
            }

            write_swift_ic(str(ic_file), box_size, particle_data, 1e6)

            # Check wrapping
            with h5py.File(ic_file, "r") as f:
                coords = f["PartType0/Coordinates"][:]
                assert np.all(coords >= 0)
                assert np.all(coords < box_size)

    def test_masses_consistent(self):
        """Test all particles have correct mass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"

            N = 100
            m_part = 1e7
            particle_data = {
                "dm": {
                    "pos": np.random.uniform(0, 10, (N, 3)),
                    "vel": np.random.uniform(-100, 100, (N, 3)),
                },
            }

            write_swift_ic(str(ic_file), 100.0, particle_data, m_part)

            with h5py.File(ic_file, "r") as f:
                masses = f["PartType1/Masses"][:]
                assert np.allclose(masses, m_part)

    def test_gas_has_positive_internal_energy(self):
        """Test gas particles have positive internal energy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"

            N = 100
            particle_data = {
                "gas": {
                    "pos": np.random.uniform(0, 10, (N, 3)),
                    "vel": np.random.uniform(-100, 100, (N, 3)),
                },
            }

            write_swift_ic(str(ic_file), 100.0, particle_data, 1e6)

            with h5py.File(ic_file, "r") as f:
                u = f["PartType0/InternalEnergy"][:]
                assert np.all(u > 0)

    def test_gas_has_positive_smoothing_length(self):
        """Test gas particles have positive smoothing length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"

            N = 100
            particle_data = {
                "gas": {
                    "pos": np.random.uniform(0, 10, (N, 3)),
                    "vel": np.random.uniform(-100, 100, (N, 3)),
                },
            }

            write_swift_ic(str(ic_file), 100.0, particle_data, 1e6)

            with h5py.File(ic_file, "r") as f:
                h = f["PartType0/SmoothingLength"][:]
                assert np.all(h > 0)

    def test_stars_have_positive_smoothing_length(self):
        """Test star particles have positive smoothing length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"

            N = 50
            particle_data = {
                "stars": {
                    "pos": np.random.uniform(0, 10, (N, 3)),
                    "vel": np.random.uniform(-100, 100, (N, 3)),
                },
            }

            write_swift_ic(str(ic_file), 100.0, particle_data, 1e6)

            with h5py.File(ic_file, "r") as f:
                h = f["PartType4/SmoothingLength"][:]
                assert np.all(h > 0)


class TestYamlWriter:
    """Test YAML parameter file writer."""

    def test_generate_swift_params(self):
        """Test YAML parameter generation."""
        params = generate_swift_params(
            ic_filename="test.hdf5",
            box_size=100.0,
            time_end_gyr=2.0,
            snapshot_dt_myr=10.0,
            output_basename="snap",
        )

        assert isinstance(params, str)
        assert "__RUN_NAME__" not in params
        assert "run_name:" in params
        assert "file_name:  test.hdf5" in params
        assert "basename:            snap" in params

    def test_generate_swift_params_run_name_override(self):
        """Run name and template overrides are applied."""
        params = generate_swift_params(
            ic_filename="test.hdf5",
            box_size=50.0,
            time_end_gyr=1.0,
            snapshot_dt_myr=5.0,
            run_name="custom-run",
            param_template="eagle_ref_cosmo",
        )

        assert "run_name:   custom-run" in params
        assert "delta_time:          0.005" in params

    def test_write_yaml_file(self):
        """Test YAML file writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test_params.yml"

            params = generate_swift_params(
                ic_filename="test.hdf5",
                box_size=100.0,
                time_end_gyr=2.0,
                snapshot_dt_myr=10.0,
            )

            write_yaml_file(str(yaml_file), params)

            assert yaml_file.exists()
            assert yaml_file.stat().st_size > 0
            text = yaml_file.read_text()
            assert "file_name:  test.hdf5" in text
            assert "basename:            snapshot" in text
