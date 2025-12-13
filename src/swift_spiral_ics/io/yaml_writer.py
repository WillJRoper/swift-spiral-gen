"""SWIFT YAML parameter file generator using packaged templates."""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
from typing import Any, Dict, Iterable, List

import yaml

_TEMPLATE_PACKAGE = "swift_spiral_ics.templates"
_TEMPLATE_SUFFIXES = (".yml", ".yaml")


def available_param_templates() -> List[str]:
    """Return available packaged parameter templates."""
    template_dir = resources.files(_TEMPLATE_PACKAGE)
    return sorted(
        path.stem for path in template_dir.iterdir() if path.suffix in _TEMPLATE_SUFFIXES
    )


def _load_template(template_name: str) -> Dict[str, Any]:
    """Load a YAML template bundled with the package."""
    template_path = resources.files(_TEMPLATE_PACKAGE) / f"{template_name}.yml"
    if not template_path.exists():
        raise ValueError(
            f"Unknown parameter template '{template_name}'. "
            f"Available: {', '.join(available_param_templates())}"
        )
    with template_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _set_nested(params: Dict[str, Any], keys: Iterable[str], value: Any) -> None:
    """Set a nested key, creating dictionaries as needed."""
    current = params
    keys = list(keys)
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def generate_swift_params(
    ic_filename: str,
    box_size: float,
    time_end_gyr: float,
    snapshot_dt_myr: float,
    output_basename: str = "snapshot",
    run_name: str | None = None,
    param_template: str = "eagle_isolated",
) -> dict:
    """Generate SWIFT parameter file content from a packaged template.

    Args:
        ic_filename: Path to IC file.
        box_size: Box size (kpc).
        time_end_gyr: End time (Gyr).
        snapshot_dt_myr: Snapshot time spacing (Myr).
        output_basename: Basename for output snapshots.
        run_name: Optional run name override.
        param_template: Name of packaged template to start from.

    Returns:
        Dict containing SWIFT parameters.
    """
    template = _load_template(param_template)
    params: Dict[str, Any] = deepcopy(template)

    # Required dynamic fields
    _set_nested(params, ["MetaData", "run_name"], run_name or "swift_spiral_run")
    _set_nested(params, ["InitialConditions", "file_name"], ic_filename)

    # Time and outputs (respect template units)
    _set_nested(params, ["TimeIntegration", "time_end"], time_end_gyr)
    _set_nested(params, ["Snapshots", "basename"], output_basename)
    _set_nested(params, ["Snapshots", "delta_time"], snapshot_dt_myr / 1000.0)
    _set_nested(params, ["Snapshots", "time_first"], 0.0)
    _set_nested(params, ["Statistics", "delta_time"], snapshot_dt_myr / 1000.0)
    _set_nested(params, ["Statistics", "time_first"], 0.0)

    # Keep box size available for downstream consumers (currently unused in template)
    _set_nested(params, ["MetaData", "box_size_kpc"], box_size)

    return params


def write_yaml_file(filename: str, params: dict) -> None:
    """Write parameters to YAML file.

    Args:
        filename: Output YAML filename.
        params: Parameter dictionary.
    """
    with open(filename, "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False, width=100)


def print_yaml_summary(filename: str, time_end_gyr: float, snapshot_dt_myr: float) -> None:
    """Print summary of YAML parameters.

    Args:
        filename: YAML filename.
        time_end_gyr: End time (Gyr).
        snapshot_dt_myr: Snapshot spacing (Myr).
    """
    n_snapshots = int(time_end_gyr * 1000 / snapshot_dt_myr) + 1

    print(f"\nGenerated SWIFT parameter file: {filename}")
    print(f"  Simulation end time: {time_end_gyr:.2f} Gyr")
    print(f"  Snapshot spacing: {snapshot_dt_myr:.2f} Myr")
    print(f"  Expected number of snapshots: ~{n_snapshots}")
    print("  Physics: Gravity + Hydro + EAGLE + SPINJETAGN (BH seeding enabled)")
