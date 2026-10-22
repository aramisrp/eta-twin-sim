"""
Módulo de cálculo de métricas estatísticas, conformidade sanitária e economia de OPEX.
Em conformidade com a Portaria GM/MS nº 888/2021 e padrões de modelagem ambiental.
"""


import numpy as np
import pandas as pd


def calculate_metrics(
    y_true: np.ndarray | list, y_pred: np.ndarray | list
) -> dict[str, float]:
    """
    Calcula R², NSE (Nash-Sutcliffe Efficiency), RMSE e MAPE.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    # Erro Quadrático Médio e RMSE
    mse = np.mean((yt - yp) ** 2)
    rmse = np.sqrt(mse)

    # Coeficiente de Determinação R²
    yt_mean = np.mean(yt)
    ss_tot = np.sum((yt - yt_mean) ** 2)
    ss_res = np.sum((yt - yp) ** 2)
    r2 = 1.0 - (ss_res / max(1e-8, ss_tot))

    # Eficiência de Nash-Sutcliffe (NSE)
    nse = 1.0 - (np.sum((yt - yp) ** 2) / max(1e-8, np.sum((yt - yt_mean) ** 2)))

    # MAPE (Mean Absolute Percentage Error)
    mask = yt > 1e-4
    mape = (
        np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0 if np.any(mask) else 0.0
    )

    return {
        "rmse": float(rmse),
        "r2": float(r2),
        "nse": float(nse),
        "mape": float(mape),
    }


def calculate_portaria_888_compliance(df_results: pd.DataFrame) -> dict[str, float]:
    """
    Calcula as taxas de conformidade (%) com os limites da Portaria GM/MS nº 888/2021:
    - Turbidez Saída <= 0.5 NTU (meta pós-filtração) e <= 5.0 NTU (padrão de aceitação na rede)
    - Cloro Residual Livre em [0.2, 2.0] mg/L
    - pH em [6.0, 9.0]
    """
    n_total = len(df_results)
    if n_total == 0:
        return {}

    # Turbidez
    turb_col = "SAI_Turbidez" if "SAI_Turbidez" in df_results.columns else "sai_turb"
    turb_05_ok = np.sum(df_results[turb_col] <= 0.5)
    turb_50_ok = np.sum(df_results[turb_col] <= 5.0)

    # Cloro
    cl_col = "SAI_Cloro" if "SAI_Cloro" in df_results.columns else "sai_cloro"
    cl_ok = np.sum((df_results[cl_col] >= 0.2) & (df_results[cl_col] <= 2.0))

    # pH
    ph_col = "SAI_pH" if "SAI_pH" in df_results.columns else "sai_ph"
    ph_ok = np.sum((df_results[ph_col] >= 6.0) & (df_results[ph_col] <= 9.0))

    return {
        "turbidity_0_5_compliance_pct": float((turb_05_ok / n_total) * 100.0),
        "turbidity_5_0_compliance_pct": float((turb_50_ok / n_total) * 100.0),
        "chlorine_compliance_pct": float((cl_ok / n_total) * 100.0),
        "ph_compliance_pct": float((ph_ok / n_total) * 100.0),
    }


def calculate_opex_savings(
    dosage_real_mg_l: np.ndarray,
    dosage_twin_mg_l: np.ndarray,
    flow_lps: np.ndarray,
    dt_hours: float = 2.0,
) -> dict[str, float]:
    """
    Calcula a massa total consumida (kg) e a economia percentual de reagentes químicos.
    Massa (kg) = sum( Dosagem (g/m³) * Vazao (m³/s) * dt (s) / 1000 )
    """
    flow_m3_s = np.asarray(flow_lps) / 1000.0
    dt_seconds = dt_hours * 3600.0

    mass_real_kg = np.sum(np.asarray(dosage_real_mg_l) * flow_m3_s * dt_seconds) / 1000.0
    mass_twin_kg = np.sum(np.asarray(dosage_twin_mg_l) * flow_m3_s * dt_seconds) / 1000.0

    diff_kg = mass_real_kg - mass_twin_kg
    saving_pct = (diff_kg / max(1e-4, mass_real_kg)) * 100.0

    return {
        "mass_real_kg": float(mass_real_kg),
        "mass_twin_kg": float(mass_twin_kg),
        "total_saved_kg": float(diff_kg),
        "relative_saving_pct": float(saving_pct),
    }
