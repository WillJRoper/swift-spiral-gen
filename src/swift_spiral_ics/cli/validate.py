"""Validate structural stability between an IC file and an evolved SWIFT snapshot."""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

_PARTICLE_GROUPS = {"gas": "PartType0", "dm": "PartType1", "stars": "PartType4"}
_KPC_IN_CGS = 3.08567758e21
_MSUN_IN_CGS = 1.98841e33


def _load_component(filename: str, component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(filename, "r") as handle:
        group = handle[_PARTICLE_GROUPS[component]]
        units = handle["Units"].attrs

        def unit_value(key: str) -> float:
            return float(np.asarray(units[key]).reshape(-1)[0])

        unit_length_cgs = unit_value("Unit length in cgs (U_L)")
        unit_time_cgs = unit_value("Unit time in cgs (U_t)")
        length_factor = unit_length_cgs / _KPC_IN_CGS
        mass_factor = unit_value("Unit mass in cgs (U_M)") / _MSUN_IN_CGS
        velocity_factor = (unit_length_cgs / unit_time_cgs) / 1.0e5
        pos = np.asarray(group["Coordinates"], dtype=float) * length_factor
        vel = np.asarray(group["Velocities"], dtype=float) * velocity_factor
        mass = np.asarray(group["Masses"], dtype=float) * mass_factor
    return pos, vel, mass


def _select_aperture(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    center: np.ndarray,
    aperture_kpc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.linalg.norm(pos - center, axis=1) <= aperture_kpc
    if np.count_nonzero(selected) < 20:
        raise ValueError("Stability aperture contains fewer than 20 particles")
    return pos[selected], vel[selected], mass[selected]


def _component_metrics(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> dict[str, float]:
    total_mass = float(np.sum(mass))
    center = np.average(pos, axis=0, weights=mass)
    bulk_velocity = np.average(vel, axis=0, weights=mass)
    relative_pos = pos - center
    relative_vel = vel - bulk_velocity
    radius = np.linalg.norm(relative_pos, axis=1)
    order = np.argsort(radius)
    cumulative_mass = np.cumsum(mass[order])
    half_mass_radius = float(radius[order][np.searchsorted(cumulative_mass, 0.5 * total_mass)])

    angular_momentum = np.sum(
        mass[:, None] * np.cross(relative_pos, relative_vel), axis=0
    )
    angular_momentum_norm = float(np.linalg.norm(angular_momentum))
    if angular_momentum_norm > 0.0:
        normal = angular_momentum / angular_momentum_norm
        height = relative_pos @ normal
        scale_height = float(np.sqrt(np.average(height**2, weights=mass)))
    else:
        scale_height = 0.0

    return {
        "mass_msun": total_mass,
        "half_mass_radius_kpc": half_mass_radius,
        "rms_scale_height_kpc": scale_height,
        "specific_angular_momentum_kpc_kms": angular_momentum_norm / total_mass,
    }


def compare_snapshots(
    initial_file: str,
    evolved_file: str,
    component: str,
    initial_center_kpc: np.ndarray,
    final_center_kpc: np.ndarray,
    aperture_kpc: float,
) -> dict[str, object]:
    initial = _select_aperture(
        *_load_component(initial_file, component), initial_center_kpc, aperture_kpc
    )
    evolved = _select_aperture(
        *_load_component(evolved_file, component), final_center_kpc, aperture_kpc
    )
    initial_metrics = _component_metrics(*initial)
    evolved_metrics = _component_metrics(*evolved)

    def fractional_change(key: str) -> float:
        baseline = initial_metrics[key]
        return abs(evolved_metrics[key] - baseline) / max(abs(baseline), 1.0e-30)

    return {
        "initial": initial_metrics,
        "evolved": evolved_metrics,
        "fractional_changes": {
            "mass": fractional_change("mass_msun"),
            "half_mass_radius": fractional_change("half_mass_radius_kpc"),
            "scale_height": fractional_change("rms_scale_height_kpc"),
            "specific_angular_momentum": fractional_change(
                "specific_angular_momentum_kpc_kms"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial_file")
    parser.add_argument("evolved_file")
    parser.add_argument("--component", choices=sorted(_PARTICLE_GROUPS), default="stars")
    parser.add_argument("--initial-center-kpc", nargs=3, type=float, required=True)
    parser.add_argument("--final-center-kpc", nargs=3, type=float)
    parser.add_argument("--aperture-kpc", type=float, required=True)
    parser.add_argument("--max-mass-change", type=float, default=0.05)
    parser.add_argument("--max-half-mass-radius-change", type=float, default=0.05)
    parser.add_argument("--max-scale-height-change", type=float, default=0.10)
    parser.add_argument("--max-angular-momentum-change", type=float, default=0.05)
    args = parser.parse_args()

    initial_center = np.asarray(args.initial_center_kpc, dtype=float)
    final_center = np.asarray(
        args.final_center_kpc if args.final_center_kpc is not None else args.initial_center_kpc,
        dtype=float,
    )
    report = compare_snapshots(
        args.initial_file,
        args.evolved_file,
        args.component,
        initial_center,
        final_center,
        args.aperture_kpc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    changes = report["fractional_changes"]
    failed = (
        changes["mass"] > args.max_mass_change
        or changes["half_mass_radius"] > args.max_half_mass_radius_change
        or changes["scale_height"] > args.max_scale_height_change
        or changes["specific_angular_momentum"] > args.max_angular_momentum_change
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
