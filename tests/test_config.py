from pathlib import Path

import yaml


def test_plant_params_exists_and_valid():
    config_dir = Path(__file__).parent.parent / "config"
    params_file = config_dir / "plant_params.yaml"
    assert params_file.exists(), f"Arquivo não encontrado: {params_file}"
    
    with open(params_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    assert data["plant"]["nominal_flow_lps"] == 25.0
    assert data["units"]["floc_decanters"]["num_parallel_units"] == 2
    assert data["units"]["disinfection_and_storage"]["total_effective_contact_volume_m3"] == 800.0
    assert data["actuators"]["dosing_pumps"]["coagulant_max_mg_l"] == 80.0
    assert data["portaria_888"]["turbidity_target_ntu"] == 0.5

def test_simulation_config_exists_and_valid():
    config_dir = Path(__file__).parent.parent / "config"
    sim_file = config_dir / "simulation_config.yaml"
    assert sim_file.exists(), f"Arquivo não encontrado: {sim_file}"
    
    with open(sim_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    assert data["simulation"]["time_step_hours"] == 2.0
    assert 1 in data["seasons"]["chuvoso_months"]
    assert 7 in data["seasons"]["seca_months"]
    assert data["estuary_boundary_defaults"]["tide_amplitude_m"] == 2.5
