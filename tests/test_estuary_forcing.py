import numpy as np

from models.estuary_forcing import (
    EstuaryBoundaryConditions,
    EstuaryTideModel,
    HydraulicPumpAffinity,
    RainfallRunoffModel,
)


def test_tide_model_harmonics_and_shear():
    tide = EstuaryTideModel(amplitude_m=2.5, m2_period_h=12.42, s2_period_h=12.0)
    
    # Nível no tempo 0 e no pico
    level_0 = tide.get_water_level(0.0)
    assert -2.5 <= level_0 <= 2.5
    
    # Pulso de turbidez por cisalhamento de maré (ressuspensão)
    shear_pulse = tide.get_shear_turbidity_pulse(3.1) # quarto de período
    assert shear_pulse >= 0.0

def test_rainfall_runoff_convolution():
    rain = RainfallRunoffModel(washoff_gain=1.2, tau_recession_h=12.0)
    
    # Pulso de chuva instantâneo
    time_series_h = np.linspace(0, 48, 49)
    rain_series_mm = np.zeros_like(time_series_h)
    rain_series_mm[10] = 50.0 # tempestade de 50mm na hora 10
    
    turb_rain = rain.simulate_turbidity_response(time_series_h, rain_series_mm)
    assert len(turb_rain) == len(time_series_h)
    assert turb_rain[10] > 0.0
    assert turb_rain[12] > turb_rain[24] # decaimento exponencial de recessão

def test_pump_affinity_laws_and_energy():
    pump = HydraulicPumpAffinity(nominal_power_kw=15.0, nominal_freq_hz=60.0, nominal_flow_lps=25.0)
    
    # Modulação de frequência por VFD
    power_60 = pump.calculate_power(60.0)
    power_50 = pump.calculate_power(50.0)
    flow_50 = pump.calculate_flow(50.0)
    
    assert power_60 == 15.0
    assert np.isclose(power_50, 15.0 * (50.0 / 60.0) ** 3, atol=1e-2)
    assert np.isclose(flow_50, 25.0 * (50.0 / 60.0), atol=1e-2)

def test_full_estuary_boundary_conditions():
    boundary = EstuaryBoundaryConditions()
    state = boundary.get_boundary_state(t_hours=6.0, season="chuvoso", rainfall_mm=25.0)
    
    assert "ab_turbidez" in state
    assert "ab_cor" in state
    assert "tide_level_m" in state
    assert state["ab_turbidez"] > 0.0
    assert state["ab_cor"] > 0.0
