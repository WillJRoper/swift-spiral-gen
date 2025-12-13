"""SWIFT YAML parameter file generator."""

import yaml


def generate_swift_params(
    ic_filename: str,
    box_size: float,
    time_end_gyr: float,
    snapshot_dt_myr: float,
    output_basename: str = "snapshot",
) -> dict:
    """Generate SWIFT parameter file content.

    Args:
        ic_filename: Path to IC file.
        box_size: Box size (kpc).
        time_end_gyr: End time (Gyr).
        snapshot_dt_myr: Snapshot time spacing (Myr).
        output_basename: Basename for output snapshots.

    Returns:
        Dict containing SWIFT parameters.
    """
    # Convert times to internal units (assume default SWIFT units)
    # SWIFT uses 978.5 Myr as internal time unit for galaxy simulations

    params = {
        "MetaData": {"run_name": "spiral_galaxy_simulation"},
        "InternalUnitSystem": {
            "UnitMass_in_cgs": 1.98841e43,  # Msun
            "UnitLength_in_cgs": 3.085678e21,  # kpc
            "UnitVelocity_in_cgs": 1.0e5,  # km/s
            "UnitCurrent_in_cgs": 1.0,
            "UnitTemp_in_cgs": 1.0,
        },
        "Cosmology": {
            "Omega_cdm": 0.0,
            "Omega_lambda": 0.0,
            "Omega_b": 0.0,
            "h": 0.7,
            "a_begin": 1.0,
            "a_end": 1.0,
        },
        "TimeIntegration": {
            "time_begin": 0.0,
            "time_end": time_end_gyr,
            "dt_min": 1e-6,
            "dt_max": 0.01,
        },
        "Snapshots": {
            "basename": output_basename,
            "scale_factor_first": 1.0,
            "time_first": 0.0,
            "delta_time": snapshot_dt_myr / 1000.0,  # Convert to Gyr
            "invoke_stf": 0,
        },
        "Statistics": {
            "delta_time": snapshot_dt_myr / 1000.0,  # Convert to Gyr
            "scale_factor_first": 1.0,
            "time_first": 0.0,
        },
        "Gravity": {
            "eta": 0.025,
            "theta": 0.7,
            "comoving_DM_softening": 0.1,
            "max_physical_DM_softening": 0.1,
            "comoving_baryon_softening": 0.1,
            "max_physical_baryon_softening": 0.1,
            "mesh_side_length": 64,
        },
        "SPH": {
            "resolution_eta": 1.2348,
            "h_min_ratio": 0.1,
            "h_max": 10.0,
            "CFL_condition": 0.1,
            "minimal_temperature": 100.0,
            "initial_temperature": 1e4,
            "particle_splitting": 1,
            "particle_splitting_mass_threshold": 2.0,
        },
        "InitialConditions": {
            "file_name": ic_filename,
            "periodic": 1,
            "cleanup_h_factors": 0,
            "cleanup_velocity_factors": 0,
            "generate_gas_in_ics": 0,
            "cleanup_smoothing_lengths": 1,
        },
        "EAGLEChemistry": {
            "init_abundance_metal": 0.01,
            "init_abundance_Hydrogen": 0.752,
            "init_abundance_Helium": 0.248,
            "init_abundance_Carbon": 0.0,
            "init_abundance_Nitrogen": 0.0,
            "init_abundance_Oxygen": 0.0,
            "init_abundance_Neon": 0.0,
            "init_abundance_Magnesium": 0.0,
            "init_abundance_Silicon": 0.0,
            "init_abundance_Iron": 0.0,
        },
        "EAGLECooling": {
            "dir_name": "./coolingtables",
            "H_reion_z": 11.5,
            "He_reion_z_centre": 3.5,
            "He_reion_z_sigma": 0.5,
            "He_reion_eV_p_H": 2.0,
        },
        "EAGLEStarFormation": {
            "SF_threshold": "Subgrid",
            "SF_model": "PressureLaw",
            "KS_normalisation": 1.515e-4,
            "KS_exponent": 1.4,
            "min_over_density": 100.0,
            "KS_high_density_threshold_H_p_cm3": 1e3,
            "EOS_entropy_margin_dex": 0.3,
            "threshold_norm_H_p_cm3": 0.1,
            "threshold_Z0": 0.002,
            "threshold_slope": -0.64,
            "threshold_max_density_H_p_cm3": 10.0,
        },
        "EAGLEFeedback": {
            "use_SNII_feedback": 1,
            "use_SNIa_feedback": 1,
            "use_AGB_enrichment": 1,
            "use_SNII_enrichment": 1,
            "use_SNIa_enrichment": 1,
            "filename": "./yieldtables/",
            "IMF_min_mass_Msun": 0.1,
            "IMF_max_mass_Msun": 100.0,
            "SNII_min_mass_Msun": 8.0,
            "SNII_max_mass_Msun": 100.0,
            "SNII_energy_erg": 1.0e51,
            "SNII_energy_fraction_min": 0.5,
            "SNII_energy_fraction_max": 3.0,
            "SNII_energy_fraction_Z_0": 0.0012663729,
            "SNII_energy_fraction_n_0_H_p_cm3": 0.67,
            "SNII_energy_fraction_n_Z": 0.8686,
            "SNII_energy_fraction_n_n": 0.8686,
        },
        "SPINJETAGN": {
            "use_jets": 1,
            "use_agn_feedback": 1,
            "use_bh_seeding": 1,
            "BH_seed_mass_Msun": 1e5,
            "BH_seed_halo_mass_Msun": 1e10,
            "viscous_alpha": 1e6,
            "jet_efficiency": 0.1,
            "radiative_efficiency": 0.1,
            "AGN_delta_T_K": 1e8,
            "AGN_heating_temperature_K": 1e8,
        },
        "Restarts": {
            "enable": 1,
            "save": 1,
            "delta_hours": 6.0,
            "max_run_time": 71.5,
            "resubmit_on_exit": 0,
            "resubmit_command": "bash resubmit.sh",
        },
    }

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
