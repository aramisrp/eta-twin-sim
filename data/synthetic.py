"""
Synthetic Water Quality Time Series Generator for Demonstrations and CI/CD Testing.
Generates realistic multi-seasonal time series with tidal cycles and rainfall shocks.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_water_quality_data(
    output_path: str | Path | None = None,
    num_records: int = 2400,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic water quality dataset matching the standard schema.
    """
    np.random.seed(seed)

    timestamps = pd.date_range("2025-01-01 01:00", periods=num_records, freq="2h")
    months = timestamps.month

    seasons = []
    for m in months:
        if m in (1, 2, 3, 4, 5):
            seasons.append("chuvoso")
        elif m in (6, 7):
            seasons.append("transicao_chuva_seca")
        elif m in (8, 9, 10, 11):
            seasons.append("seca")
        else:
            seasons.append("transicao_seca_chuva")

    hours = (timestamps - timestamps[0]).total_seconds() / 3600.0

    tide_harmonic = 0.5 * np.sin(2 * np.pi * hours / 12.42) + 0.2 * np.sin(2 * np.pi * hours / 12.00)
    seasonal_base = np.where(np.isin(months, [1, 2, 3, 4, 5]), 55.0, 30.0)

    ab_turb = np.clip(seasonal_base + 15.0 * tide_harmonic + np.random.normal(0, 4.0, num_records), 10.0, 250.0)
    ab_cor = np.clip(ab_turb * 2.2 + np.random.normal(0, 10.0, num_records), 20.0, 400.0)
    ab_ph = np.clip(6.8 + 0.3 * np.sin(2 * np.pi * hours / 24.0) + np.random.normal(0, 0.1, num_records), 5.5, 8.5)
    ab_vazao = np.clip(25.0 + np.random.normal(0, 1.5, num_records), 18.0, 30.0)

    dec_turb = np.clip(ab_turb * 0.18 + np.random.normal(0, 1.0, num_records), 1.5, 30.0)
    dec_cor = np.clip(ab_cor * 0.25 + np.random.normal(0, 3.0, num_records), 5.0, 80.0)
    dec_ph = np.clip(ab_ph - 0.3 + np.random.normal(0, 0.05, num_records), 5.0, 8.0)

    fil_turb = np.clip(dec_turb * 0.15 + np.random.normal(0, 0.2, num_records), 0.1, 8.0)
    fil_cor = np.clip(dec_cor * 0.30 + np.random.normal(0, 1.0, num_records), 1.0, 25.0)

    sai_turb = np.clip(fil_turb * 0.90, 0.05, 5.0)
    sai_cor = np.clip(fil_cor * 0.90, 1.0, 20.0)
    sai_ph = np.clip(dec_ph + 0.2, 6.0, 9.0)
    sai_cloro = np.clip(1.4 + 0.2 * np.sin(2 * np.pi * hours / 48.0) + np.random.normal(0, 0.15, num_records), 0.2, 2.5)

    df = pd.DataFrame(
        {
            "Mes": months,
            "Sazonalidade": seasons,
            "Data": timestamps.strftime("%d/%m/%Y"),
            "Hora": timestamps.strftime("%H:%M"),
            "ETA": "ETA-Piloto",
            "AB_pH": np.round(ab_ph, 2),
            "AB_Cor": np.round(ab_cor, 1),
            "AB_Turbidez": np.round(ab_turb, 2),
            "AB_Vazao": np.round(ab_vazao, 2),
            "DEC_pH": np.round(dec_ph, 2),
            "DEC_Cor": np.round(dec_cor, 1),
            "DEC_Turbidez": np.round(dec_turb, 2),
            "FIL_Cor": np.round(fil_cor, 1),
            "FIL_Turbidez": np.round(fil_turb, 2),
            "SAI_pH": np.round(sai_ph, 2),
            "SAI_Cloro": np.round(sai_cloro, 2),
            "SAI_Cor": np.round(sai_cor, 1),
            "SAI_Turbidez": np.round(sai_turb, 2),
        }
    )

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    return df
