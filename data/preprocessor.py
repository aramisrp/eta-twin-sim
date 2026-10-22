"""
Divisão, estratificação sazonal e injeção de distúrbios controlados da série temporal.
"""


import pandas as pd


class SeasonalSplitter:
    """
    Estratifica a série temporal por regimes sazonais e de transição estuarina/tropical.
    """

    def __init__(
        self,
        chuvoso_months: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        seca_months: tuple[int, ...] = (7, 8, 9, 10, 11, 12),
        transition_rain_to_dry: tuple[int, ...] = (6, 7),
        transition_dry_to_rain: tuple[int, ...] = (12, 1),
    ):
        self.chuvoso_months = chuvoso_months
        self.seca_months = seca_months
        self.trans_rain_dry = transition_rain_to_dry
        self.trans_dry_rain = transition_dry_to_rain

    def split_by_season(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Retorna dicionário com os subsets estacionais.
        """
        if "Mes" not in df.columns and "DataHora" in df.columns:
            df["Mes"] = df["DataHora"].dt.month

        result = {
            "chuvoso": df[df["Mes"].isin(self.chuvoso_months)].copy(),
            "seca": df[df["Mes"].isin(self.seca_months)].copy(),
            "transicao_chuva_seca": df[df["Mes"].isin(self.trans_rain_dry)].copy(),
            "transicao_seca_chuva": df[df["Mes"].isin(self.trans_dry_rain)].copy(),
        }
        return result

    @staticmethod
    def train_test_split(
        df: pd.DataFrame, train_ratio: float = 0.70
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divisão cronológica (sem vazamento de dados futuros) para calibração e teste.
        """
        n = len(df)
        split_idx = int(n * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        return train_df, test_df
