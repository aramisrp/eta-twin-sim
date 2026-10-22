"""
Carregador e validador de séries temporais de qualidade da água para simulação de ETA.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class ETADatasetLoader:
    """
    Carrega, valida e limpa a série temporal de qualidade da água.
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    def load_clean_data(self) -> pd.DataFrame:
        """
        Carrega o arquivo CSV, cria índice temporal Datetime e trata valores nulos/outliers.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Input dataset not found: {self.csv_path}\n"
                "Note: Operational plant datasets are proprietary (see data/README.md). "
                "To run the simulation, please provide a path to your structured CSV dataset."
            )

        # Leitura com pandas
        df = pd.read_csv(self.csv_path, encoding="utf-8")

        # Criação da coluna DataHora datetime
        df["DataHora_str"] = df["Data"] + " " + df["Hora"]
        df["DataHora"] = pd.to_datetime(
            df["DataHora_str"], format="%d/%m/%Y %H:%M", errors="coerce"
        )
        
        # Ordenação cronológica
        df = df.sort_values(by="DataHora").reset_index(drop=True)

        # Colunas numéricas que devem ser estritamente não-negativas
        non_negative_cols = [
            "AB_Turbidez", "AB_Cor", "AB_Vazao",
            "DEC_Turbidez", "DEC_Cor",
            "FIL_Turbidez", "FIL_Cor",
            "SAI_Turbidez", "SAI_Cor", "SAI_Cloro"
        ]

        for col in non_negative_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Trata NaNs por interpolação linear temporal
                df[col] = df[col].interpolate(method="linear").bfill().ffill()
                # Garante não-negatividade
                df[col] = np.maximum(df[col], 0.0)

        # Trata pH
        ph_cols = ["AB_pH", "DEC_pH", "SAI_pH"]
        for col in ph_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].interpolate(method="linear").bfill().ffill()
                df[col] = np.clip(df[col], 3.0, 11.0)

        # Padronização de nomes de sazonalidade
        df["Sazonalidade"] = df["Sazonalidade"].astype(str).str.strip().str.lower()

        return df
