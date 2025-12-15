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


import re # Add import

def generate_swift_params(
    ic_filename: str,
    box_size: float,
    time_end_gyr: float,
    snapshot_dt_myr: float,
    dt_min_gyr: float,
    softening_kpc: float,
    output_basename: str = "snapshot",
    run_name: str | None = None,
    param_template: str = "eagle_ref_cosmo",
    min_gas_mass_msun: float | None = None,
) -> str:
    """Generate a parameter file by substituting tokens in the template text."""
    template_text = _load_template_text(param_template)
    snapshot_dt_gyr = snapshot_dt_myr / 1000.0
    # Softening now in Mpc
    softening_mpc_val = softening_kpc / 1000.0

    # Remove Cosmology section to ensure non-cosmological run
    template_text = re.sub(r'(?m)^# Cosmological parameters\nCosmology:.*?(?=^# Parameters)', '', template_text, flags=re.DOTALL)
    
    # Set periodic to 0
    template_text = re.sub(r'periodic:\s*1', 'periodic:   0', template_text)

    # Update SPH Parameters for Unit System (Mpc, 1e10 Msun)
    
    # 1. h_max: Set to half box size, converted to Mpc
    h_max_val = box_size / 2.0 / 1000.0 # box_size is in kpc, convert to Mpc
    template_text = re.sub(r'h_max:\s*[\d.eE+-]+', f'h_max:                             {h_max_val}', template_text)
    
    # 2. Particle Splitting Threshold
    # Default to a huge number if mass unknown (effectively disable)
    splitting_threshold_internal_units = 1e5 
    if min_gas_mass_msun is not None and min_gas_mass_msun > 0:
        # min_gas_mass_msun is in Msun. Convert to 1e10 Msun units.
        splitting_threshold_internal_units = 4.0 * min_gas_mass_msun / 1e10
    
    template_text = re.sub(r'particle_splitting_mass_threshold:\s*[\d.eE+-]+', f'particle_splitting_mass_threshold: {splitting_threshold_internal_units:.4e}', template_text)

    # Inject InternalUnitSystem (Mpc, 1e10 Msun, km/s)
    new_units = """InternalUnitSystem:
  UnitMass_in_cgs:     1.98841e43    # 10^10 M_sun in grams
  UnitLength_in_cgs:   3.08567758e24 # Mpc in centimeters
  UnitVelocity_in_cgs: 1e5           # km/s in centimeters per second
  UnitCurrent_in_cgs:  1.0           # Amperes
  UnitTemp_in_cgs:     1.0           # Kelvin"""
    
    if "InternalUnitSystem:" in template_text:
        template_text = re.sub(r'(?m)^InternalUnitSystem:.*?(\n\S|\Z)', f'{new_units}\n\\1', template_text, flags=re.DOTALL)
    else:
        # Prepend if not found (unlikely)
        template_text = new_units + "\n\n" + template_text

    replacements = {
        "__RUN_NAME__": run_name or "swift_spiral_run",
        "__IC_FILE__": ic_filename,
        "__SNAP_BASENAME__": output_basename,
        "__SNAP_DT__": f"{snapshot_dt_gyr}",
        "__STAT_DT__": f"{snapshot_dt_gyr}",
        "__DT_MIN_GYR__": f"{dt_min_gyr}",
        "__SOFTENING__": f"{softening_mpc_val}",
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
