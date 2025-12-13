"""Minimal YAML writer that performs token replacement on a text template.

We keep the parameter file identical to the shipped EAGLE example and only
substitute a few fields: run name, IC filename, snapshot basename, and snapshot
cadence. Everything else stays verbatim to avoid SWIFT parser issues.
"""

from __future__ import annotations

from importlib import resources
from typing import List


_TEMPLATE_PACKAGE = "swift_spiral_ics.templates"
_TEMPLATE_SUFFIXES = (".yml", ".yaml")


def available_param_templates() -> List[str]:
    template_dir = resources.files(_TEMPLATE_PACKAGE)
    return sorted(
        path.stem for path in template_dir.iterdir() if path.suffix in _TEMPLATE_SUFFIXES
    )


def _load_template_text(template_name: str) -> str:
    template_path = resources.files(_TEMPLATE_PACKAGE) / f"{template_name}.yml"
    if not template_path.exists():
        raise ValueError(
            f"Unknown parameter template '{template_name}'. "
            f"Available: {', '.join(available_param_templates())}"
        )
    return template_path.read_text(encoding="utf-8")


def generate_swift_params(
    ic_filename: str,
    box_size: float,
    time_end_gyr: float,
    snapshot_dt_myr: float,
    dt_min_gyr: float,
    output_basename: str = "snapshot",
    run_name: str | None = None,
    param_template: str = "eagle_ref_cosmo",
) -> str:
    """Generate a parameter file by substituting tokens in the template text."""
    template_text = _load_template_text(param_template)
    snapshot_dt_gyr = snapshot_dt_myr / 1000.0

    replacements = {
        "__RUN_NAME__": run_name or "swift_spiral_run",
        "__IC_FILE__": ic_filename,
        "__SNAP_BASENAME__": output_basename,
        "__SNAP_DT__": f"{snapshot_dt_gyr}",
        "__STAT_DT__": f"{snapshot_dt_gyr}",
        "__DT_MIN_GYR__": f"{dt_min_gyr}",
    }

    for token, value in replacements.items():
        template_text = template_text.replace(token, value)

    return template_text


def write_yaml_file(filename: str, params_text: str) -> None:
    """Write the substituted parameter text to disk."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(params_text)


def print_yaml_summary(filename: str, time_end_gyr: float, snapshot_dt_myr: float, dt_min_gyr: float) -> None:
    """Print summary of YAML parameters."""
    n_snapshots = int(time_end_gyr * 1000 / snapshot_dt_myr) + 1

    print(f"\nGenerated SWIFT parameter file: {filename}")
    print(f"  Simulation end time: {time_end_gyr:.2f} Gyr")
    print(f"  Snapshot spacing: {snapshot_dt_myr:.2f} Myr")
    print(f"  Minimum timestep: {dt_min_gyr:.2e} Gyr")
    print(f"  Expected number of snapshots: ~{n_snapshots}")
    print("  Template: EAGLE_50 example with IC/run/snapshot/dt_min fields substituted")
