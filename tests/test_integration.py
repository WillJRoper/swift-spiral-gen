"""Integration tests for full IC generation pipeline."""

import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml

from swift_spiral_ics.cli.generate import (
    _apply_config_file,
    _default_generator_args,
    _hydrostatic_disc_internal_energy,
    _normalise_per_galaxy_args,
    _remove_disc_streaming_modes,
    _resolve_galaxy_placement,
    _rotate_disc_orientation,
)
from swift_spiral_ics.physics.sampling import (
    sample_exponential_disc,
    sample_hernquist_bulge,
    sample_nfw_halo,
)
from swift_spiral_ics.utils.random import get_rng


def _tiny_galaxy_config(ic_file: Path, params_file: Path) -> dict:
    return {
        "output": {"ics": str(ic_file), "params": str(params_file)},
        "simulation": {
            "box_kpc": 100,
            "seed": 42,
            "time_end_gyr": 0.01,
            "snapshot_dt_myr": 5,
        },
        "particle_masses": {"dm_msun": 1e9, "stars_msun": 1e8, "gas_msun": 1e8},
        "grid": {"nR": 16, "nz": 16, "eps_kpc": 0.5},
        "galaxies": [
            {
                "masses": {
                    "dm_msun": 1e9,
                    "stars_msun": 1e8,
                    "gas_msun": 1e8,
                    "bulge_fraction": 0.0,
                },
                "halo": {"c200": 10},
                "bulge": {"a_kpc": 0.5},
                "stellar_disk": {"scale_length_kpc": 1.0, "scale_height_kpc": 0.1},
                "gas_disk": {"scale_length_kpc": 1.0, "scale_height_kpc": 0.1},
            }
        ],
    }


def _run_generator(config: dict, tmpdir: str) -> subprocess.CompletedProcess:
    config_file = Path(tmpdir) / "generator.yml"
    with open(config_file, "w") as handle:
        yaml.safe_dump(config, handle)
    return subprocess.run(
        [sys.executable, "-m", "swift_spiral_ics.cli.generate", str(config_file)],
        capture_output=True,
        text=True,
    )


class TestFullPipeline:
    """Test complete IC generation pipeline."""

    def test_node_angle_orientation_rotates_disc_frame(self):
        """Disc orientation supports both node angle and inclination rotations."""
        vector = np.array([[1.0, 0.0, 0.0]])

        rotated = _rotate_disc_orientation(vector, inclination=90.0, node_angle=90.0)

        assert np.allclose(rotated, [[0.0, 0.0, 1.0]], atol=1e-12)

    def test_yaml_config_maps_to_com_balanced_parabolic_placement(self):
        """Parabolic orbit configs compute two COM-balanced galaxy centres."""
        config = {
            "simulation": {"box_kpc": 200.0},
            "particle_masses": {"dm_msun": 1e9, "stars_msun": 1e8, "gas_msun": 1e8},
            "orbit": {"type": "parabolic", "r_init_kpc": 80.0, "r_peri_kpc": 10.0},
            "galaxies": [
                {
                    "masses": {
                        "dm_msun": 1.0e12,
                        "stars_msun": 6.0e10,
                        "gas_msun": 1.0e10,
                        "bulge_fraction": 0.2,
                    },
                    "halo": {"c200": 10.0},
                    "bulge": {"a_kpc": 0.8},
                    "stellar_disk": {"scale_length_kpc": 3.5, "scale_height_kpc": 0.35, "Q": 2.0},
                    "gas_disk": {"scale_length_kpc": 7.0, "scale_height_kpc": 0.1, "Q": 1.5},
                },
                {
                    "masses": {
                        "dm_msun": 2.0e12,
                        "stars_msun": 1.0e11,
                        "gas_msun": 2.0e10,
                        "bulge_fraction": 0.3,
                    },
                    "halo": {"c200": 10.0},
                    "bulge": {"a_kpc": 1.0},
                    "stellar_disk": {"scale_length_kpc": 5.0, "scale_height_kpc": 0.5, "Q": 2.0},
                    "gas_disk": {"scale_length_kpc": 10.0, "scale_height_kpc": 0.15, "Q": 1.5},
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "generator.yml"
            with open(config_file, "w") as handle:
                yaml.safe_dump(config, handle)

            args = _apply_config_file(_default_generator_args(), str(config_file))
            _normalise_per_galaxy_args(args)
            positions, velocities = _resolve_galaxy_placement(args)
            masses = np.asarray([
                args.m200_msun[i]
                + args.m_star_msun[i]
                + args.m_bulge_msun[i]
                + args.m_gas_msun[i]
                for i in range(args.n_galaxies)
            ])

        assert np.isclose(np.linalg.norm(positions[1] - positions[0]), 80.0)
        assert np.allclose(np.average(positions, axis=0, weights=masses), 0.0)
        assert np.allclose(np.average(velocities, axis=0, weights=masses), 0.0)
        assert velocities[1, 0] < velocities[0, 0]

    def test_yaml_config_maps_to_relative_velocity_orbit(self):
        """Relative velocity orbit configs preserve requested separation and velocity."""
        config = {
            "simulation": {"box_kpc": 200.0},
            "particle_masses": {"dm_msun": 1e9, "stars_msun": 1e8, "gas_msun": 1e8},
            "orbit": {
                "type": "relative_velocity",
                "separation_kpc": 80.0,
                "radial_velocity_kms": -110.0,
                "tangential_velocity_kms": 30.0,
            },
            "galaxies": [
                {
                    "masses": {
                        "dm_msun": 1.0e12,
                        "stars_msun": 6.0e10,
                        "gas_msun": 1.0e10,
                        "bulge_fraction": 0.2,
                    },
                    "black_hole": {"mass_msun": 4.3e6},
                },
                {
                    "masses": {
                        "dm_msun": 2.0e12,
                        "stars_msun": 1.0e11,
                        "gas_msun": 2.0e10,
                        "bulge_fraction": 0.3,
                    },
                    "black_hole": {"mass_msun": 1.4e8},
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "generator.yml"
            with open(config_file, "w") as handle:
                yaml.safe_dump(config, handle)

            args = _apply_config_file(_default_generator_args(), str(config_file))
            _normalise_per_galaxy_args(args)
            positions, velocities = _resolve_galaxy_placement(args)
            masses = np.asarray([
                args.m200_msun[i]
                + args.m_star_msun[i]
                + args.m_bulge_msun[i]
                + args.m_gas_msun[i]
                + args.black_hole_mass_msun[i]
                for i in range(args.n_galaxies)
            ])

        assert np.allclose(positions[1] - positions[0], [80.0, 0.0, 0.0])
        assert np.allclose(velocities[1] - velocities[0], [-110.0, 30.0, 0.0])
        assert np.allclose(np.average(positions, axis=0, weights=masses), 0.0)
        assert np.allclose(np.average(velocities, axis=0, weights=masses), 0.0)

    def test_generate_tiny_galaxy_from_yaml(self):
        """Test generating a tiny galaxy from a YAML config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            params_file = Path(tmpdir) / "test_params.yml"
            config = _tiny_galaxy_config(ic_file, params_file)

            result = _run_generator(config, tmpdir)

            assert ic_file.exists(), (
                f"IC file not created. Stdout: {result.stdout}, Stderr: {result.stderr}"
            )
            assert params_file.exists()
            with h5py.File(ic_file, "r") as f:
                assert "Header" in f
                assert "Units" in f
                assert "PartType0" in f or "PartType1" in f or "PartType4" in f

    def test_gas_disc_internal_energy_supports_requested_scale_height(self):
        """Thicker gas discs receive stronger generated pressure support."""

        class FakeGridSolver:
            eps = 0.01

            def get_potential_and_forces(self, radius, z):
                return {"FZ": np.full_like(radius, 1.0e4, dtype=float)}

        pos = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.1]])

        thin = _hydrostatic_disc_internal_energy(pos, 0.05, FakeGridSolver())
        thick = _hydrostatic_disc_internal_energy(pos, 0.5, FakeGridSolver())

        assert np.median(thick) > np.median(thin)

    def test_stellar_disc_ignores_unsupported_spiral_overdensity(self, monkeypatch):
        """Stellar spirals are not imposed without a matching non-axisymmetric potential."""

        import swift_spiral_ics.cli.generate as generate_module

        seen_spiral_params = []
        original_sampler = generate_module.sample_exponential_disc

        def wrapped_sampler(*args, **kwargs):
            seen_spiral_params.append(kwargs.get("spiral_params"))
            return original_sampler(*args, **kwargs)

        monkeypatch.setattr(generate_module, "sample_exponential_disc", wrapped_sampler)

        args = _default_generator_args()
        args.n_halo = [2]
        args.n_bulge = [0]
        args.n_star = [2]
        args.n_gas = [2]
        args.nR_grid = 16
        args.nz_grid = 16
        args.box_kpc = 100.0
        args.arm_strength = [0.5]
        args.arm_stream_frac = [0.2]
        _normalise_per_galaxy_args(args)

        generate_module.generate_galaxy_particles(0, args, get_rng(1))

        assert seen_spiral_params[0] is None
        assert seen_spiral_params[1] is not None

    def test_stellar_disc_streaming_modes_are_removed_by_annulus(self):
        """Collisionless stabilization removes coherent radial expansion bands."""

        radius = np.linspace(1.0, 10.0, 200)
        phi = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
        pos = np.column_stack([radius * np.cos(phi), radius * np.sin(phi), np.zeros_like(radius)])
        vel = np.column_stack([20.0 * np.cos(phi), 20.0 * np.sin(phi), np.full_like(radius, 5.0)])

        stabilized = _remove_disc_streaming_modes(pos, vel)
        v_radial = (
            stabilized[:, 0] * np.cos(phi)
            + stabilized[:, 1] * np.sin(phi)
        )

        assert abs(float(np.mean(v_radial))) < 1.0e-12
        assert abs(float(np.mean(stabilized[:, 2]))) < 1.0e-12

    def test_multi_galaxy_positions_are_literal_box_coordinates(self):
        """Per-galaxy YAML positions are interpreted as literal coordinates in the box."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            params_file = Path(tmpdir) / "test_params.yml"
            galaxy = _tiny_galaxy_config(ic_file, params_file)["galaxies"][0]
            config = _tiny_galaxy_config(ic_file, params_file)
            config["galaxies"] = [
                {**galaxy, "placement": {"position_kpc": [10, 50, 50]}},
                {**galaxy, "placement": {"position_kpc": [90, 50, 50]}},
            ]

            result = _run_generator(config, tmpdir)

            assert result.returncode == 0, result.stderr
            with h5py.File(ic_file, "r") as f:
                dm_x_kpc = f["PartType1/Coordinates"][:, 0] * 1000.0
                assert dm_x_kpc.min() < 20.0
                assert dm_x_kpc.max() > 80.0

    def test_generate_tiny_parabolic_merger_from_yaml(self):
        """The CLI can generate a two-galaxy parabolic merger from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            params_file = Path(tmpdir) / "test_params.yml"
            galaxy = _tiny_galaxy_config(ic_file, params_file)["galaxies"][0]
            config = _tiny_galaxy_config(ic_file, params_file)
            config["simulation"]["box_kpc"] = 200
            config["orbit"] = {"type": "parabolic", "r_init_kpc": 40, "r_peri_kpc": 5}
            config["galaxies"] = [
                galaxy,
                {
                    **galaxy,
                    "masses": {
                        "dm_msun": 2e9,
                        "stars_msun": 2e8,
                        "gas_msun": 2e8,
                        "bulge_fraction": 0.0,
                    },
                },
            ]

            result = _run_generator(config, tmpdir)

            assert result.returncode == 0, result.stderr
            assert ic_file.exists()
            assert params_file.exists()
            with h5py.File(ic_file, "r") as f:
                assert f["Header"].attrs["NumPart_Total"][1] == 3

    def test_generate_central_black_hole_from_yaml(self):
        """A configured central black hole is written as PartType5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            params_file = Path(tmpdir) / "test_params.yml"
            config = _tiny_galaxy_config(ic_file, params_file)
            config["galaxies"][0]["black_hole"] = {"mass_msun": 4.3e6}

            result = _run_generator(config, tmpdir)

            assert result.returncode == 0, result.stderr
            with h5py.File(ic_file, "r") as f:
                assert f["Header"].attrs["NumPart_Total"][5] == 1
                assert "PartType5" in f
                assert np.isclose(f["PartType5/Masses"][0], 4.3e6 / 1e10)
                assert "DynamicalMasses" in f["PartType5"]
                assert "SubgridMasses" in f["PartType5"]
                assert "SmoothingLength" in f["PartType5"]

    def test_generate_hot_cgm_from_yaml(self):
        """A configured CGM adds hot gas particles around the galaxy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            params_file = Path(tmpdir) / "test_params.yml"
            config = _tiny_galaxy_config(ic_file, params_file)
            config["simulation"]["box_kpc"] = 200
            config["galaxies"][0]["cgm"] = {
                "enabled": True,
                "mass_msun": 2e8,
                "r_min_kpc": 20.0,
                "r_max_kpc": 50.0,
                "core_radius_kpc": 20.0,
                "beta": 0.5,
                "temperature_floor_K": 1e4,
                "temperature_ceiling_K": 1e8,
            }

            result = _run_generator(config, tmpdir)

            assert result.returncode == 0, result.stderr
            with h5py.File(ic_file, "r") as f:
                assert f["Header"].attrs["NumPart_Total"][0] == 3
                internal_energy = f["PartType0/InternalEnergy"][:]
                assert np.all(internal_energy > 0.0)

    def test_mw_m31_example_generates_reduced_stable_cgm(self):
        """The shipped MW-M31 CGM parameters pass generation-time stability checks."""

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(__file__).parents[1]
            with open(repo_root / "examples" / "mw_m31_merger.yml") as handle:
                config = yaml.safe_load(handle)

            ic_file = Path(tmpdir) / "mw_m31_smoke.hdf5"
            params_file = Path(tmpdir) / "mw_m31_smoke.yml"
            config["output"] = {
                "ics": str(ic_file),
                "params": str(params_file),
                "run_name": "mw_m31_smoke",
            }
            config["simulation"]["time_end_gyr"] = 0.01
            config["simulation"]["snapshot_dt_myr"] = 5.0
            config["particle_masses"] = {
                "dm_msun": 5.0e12,
                "stars_msun": 5.0e11,
                "gas_msun": 5.0e10,
            }
            config["grid"] = {**config["grid"], "nR": 32, "nz": 32}

            result = _run_generator(config, tmpdir)

            assert result.returncode == 0, result.stderr
            assert ic_file.exists()

    def test_mw_m31_example_config_parses(self):
        """The shipped MW-M31 example stays in sync with the generator schema."""
        config_file = Path(__file__).parents[1] / "examples" / "mw_m31_merger.yml"

        args = _apply_config_file(_default_generator_args(), str(config_file))
        _normalise_per_galaxy_args(args)
        positions, velocities = _resolve_galaxy_placement(args)

        assert args.n_galaxies == 4
        assert args.galaxy_names == ["Milky Way", "Andromeda", "LMC", "SMC"]
        assert args.orbit == "relative_velocity"
        assert args.black_hole_mass_msun == [4.3e6, 1.4e8, 0.0, 0.0]
        assert np.isclose(np.linalg.norm(positions[1] - positions[0]), 780.0)
        assert np.allclose(velocities[1] - velocities[0], [-110.0, 17.0, 0.0])
        assert np.allclose(positions[2] - positions[0], [-1.0, -41.0, -28.0])
        assert np.allclose(positions[3] - positions[0], [15.0, -38.0, -44.0])
        assert np.all(np.isfinite(velocities))

    def test_mw_m31_resolution_example_configs_parse(self):
        """Resolution variants keep the same setup with finer particle masses."""
        examples = {
            "mw_m31_merger_10x.yml": (1.0e6, 1.0e5, 1.0e5),
            "mw_m31_merger_100x.yml": (1.0e5, 1.0e4, 1.0e4),
            "mw_m31_merger_1000x.yml": (1.0e4, 1.0e3, 1.0e3),
        }

        for filename, particle_masses in examples.items():
            config_file = Path(__file__).parents[1] / "examples" / filename
            args = _apply_config_file(_default_generator_args(), str(config_file))
            _normalise_per_galaxy_args(args)
            positions, velocities = _resolve_galaxy_placement(args)

            assert (
                args.dm_part_mass_msun,
                args.star_part_mass_msun,
                args.gas_part_mass_msun,
            ) == particle_masses
            assert args.orbit == "relative_velocity"
            assert np.isclose(np.linalg.norm(positions[1] - positions[0]), 780.0)
            assert np.allclose(velocities[1] - velocities[0], [-110.0, 17.0, 0.0])

    def test_multi_galaxy_positions_must_lie_inside_box(self):
        """Out-of-box galaxy coordinates are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            params_file = Path(tmpdir) / "test_params.yml"
            galaxy = _tiny_galaxy_config(ic_file, params_file)["galaxies"][0]
            config = _tiny_galaxy_config(ic_file, params_file)
            config["galaxies"] = [
                {**galaxy, "placement": {"position_kpc": [10, 50, 50]}},
                {**galaxy, "placement": {"position_kpc": [110, 50, 50]}},
            ]

            result = _run_generator(config, tmpdir)

            assert result.returncode != 0
            assert "must lie within 0 and --box-kpc" in result.stderr

    def test_random_background_radius_limits_gas_extent(self):
        """Random background gas can be restricted to a sphere around the box centre."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ic_file = Path(tmpdir) / "test_ic.hdf5"
            params_file = Path(tmpdir) / "test_params.yml"
            config = _tiny_galaxy_config(ic_file, params_file)
            config["particle_masses"]["gas_msun"] = 1e7
            config["background"] = {
                "gas_density_msun_kpc3": 1e4,
                "grid_kpc": 0,
                "radius_kpc": 20,
            }

            result = _run_generator(config, tmpdir)

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
            params_file = Path(tmpdir) / "test_params.yml"
            config = _tiny_galaxy_config(ic_file, params_file)
            config["particle_masses"]["gas_msun"] = 1e7
            config["background"] = {
                "gas_density_msun_kpc3": 1e4,
                "grid_kpc": 10,
                "radius_kpc": 20,
            }

            result = _run_generator(config, tmpdir)

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

        assert abs(M_achieved - M_requested) <= 0.5 * m_part
