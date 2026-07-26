"""Tests for evolved-snapshot stability validation."""

import numpy as np

from swift_spiral_ics.cli.validate import compare_snapshots
from swift_spiral_ics.io.swift_writer import write_swift_ic


def test_identical_snapshot_is_stable(tmp_path):
    filename = tmp_path / "snapshot.hdf5"
    rng = np.random.default_rng(3)
    pos = rng.normal(scale=2.0, size=(100, 3)) + 50.0
    vel = rng.normal(scale=20.0, size=(100, 3))
    particle_data = {
        "dm": {"pos": np.empty((0, 3)), "vel": np.empty((0, 3)), "mass": np.empty(0)},
        "gas": {
            "pos": np.empty((0, 3)),
            "vel": np.empty((0, 3)),
            "mass": np.empty(0),
            "internal_energy": np.empty(0),
        },
        "stars": {"pos": pos, "vel": vel, "mass": np.full(100, 1.0e6)},
        "black_holes": {
            "pos": np.empty((0, 3)),
            "vel": np.empty((0, 3)),
            "mass": np.empty(0),
        },
    }
    write_swift_ic(str(filename), 100.0, particle_data)

    report = compare_snapshots(
        str(filename),
        str(filename),
        "stars",
        np.array([50.0, 50.0, 50.0]),
        np.array([50.0, 50.0, 50.0]),
        20.0,
    )

    assert all(value == 0.0 for value in report["fractional_changes"].values())
