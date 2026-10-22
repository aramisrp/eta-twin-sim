from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import ETADatasetLoader
from evaluation.metrics import (
    calculate_metrics,
    calculate_opex_savings,
    calculate_portaria_888_compliance,
)
from evaluation.scenarios import ScenarioRunner


def test_metrics_calculation():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([10.2, 19.8, 30.1, 39.7, 50.3])
    
    m = calculate_metrics(y_true, y_pred)
    assert m["r2"] > 0.99
    assert m["nse"] > 0.99
    assert m["rmse"] < 0.5
    assert m["mape"] < 3.0

def test_portaria_888_compliance_calculation():
    df = pd.DataFrame({
        "SAI_Turbidez": [0.3, 0.4, 0.48, 0.55, 6.0], # 3 <= 0.5 (60%), 4 <= 5.0 (80%)
        "SAI_Cloro": [1.5, 1.2, 0.8, 0.1, 2.5],       # 3 em [0.2, 2.0] (60%)
        "SAI_pH": [6.8, 7.0, 7.2, 5.8, 9.2],          # 3 em [6.0, 9.0] (60%)
    })
    
    comp = calculate_portaria_888_compliance(df)
    assert comp["turbidity_0_5_compliance_pct"] == 60.0
    assert comp["turbidity_5_0_compliance_pct"] == 80.0
    assert comp["chlorine_compliance_pct"] == 60.0
    assert comp["ph_compliance_pct"] == 60.0

def test_portaria_888_compliance_empty_and_lowercase_columns():
    # DataFrame vazio retorna dicionário vazio
    assert calculate_portaria_888_compliance(pd.DataFrame()) == {}

    # Colunas em minúsculo (formato de saída do simulador)
    df = pd.DataFrame({
        "sai_turb": [0.3, 6.0],
        "sai_cloro": [1.0, 0.1],
        "sai_ph": [7.0, 5.0],
    })
    comp = calculate_portaria_888_compliance(df)
    assert comp["turbidity_0_5_compliance_pct"] == 50.0
    assert comp["chlorine_compliance_pct"] == 50.0
    assert comp["ph_compliance_pct"] == 50.0

def test_metrics_constant_series():
    # Série constante: R²/NSE indefinidos não devem explodir (guard ss_tot)
    y_true = np.full(10, 5.0)
    y_pred = np.full(10, 5.2)
    m = calculate_metrics(y_true, y_pred)
    assert np.isfinite(m["r2"])
    assert np.isfinite(m["nse"])
    assert m["rmse"] > 0.0

def test_opex_savings():
    q_series = np.full(10, 25.0)
    dos_real = np.full(10, 45.0) # 45 mg/L manual
    dos_twin = np.full(10, 35.0) # 35 mg/L automatizado
    
    savings = calculate_opex_savings(dos_real, dos_twin, q_series)
    assert np.isclose(savings["relative_saving_pct"], (45.0 - 35.0) / 45.0 * 100.0)
    assert savings["total_saved_kg"] > 0.0

def test_scenario_runner():
    artigo_dir = Path(__file__).parents[3]
    data_path = artigo_dir / "02_Series_Historicas_e_Estatistica" / "Dados" / "3_Ouro" / "ouro_wide.csv"
    loader = ETADatasetLoader(data_path)
    df = loader.load_clean_data().iloc[:100] # Subamostra para teste rápido
    
    runner = ScenarioRunner()
    results = runner.run_scenarios(df)
    
    assert "baseline_historical" in results
    assert "digital_twin_routine" in results
    assert "extreme_storm_tide" in results
    assert "summary_table" in results
    assert isinstance(results["summary_table"], pd.DataFrame)
