from pathlib import Path

import pandas as pd

from data.loader import ETADatasetLoader
from data.preprocessor import SeasonalSplitter


def test_load_clean_historical_data(default_data_path):
    loader = ETADatasetLoader(default_data_path)
    df = loader.load_clean_data()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) >= 2200

    # Conferência de colunas obrigatórias
    expected_cols = [
        "DataHora", "Mes", "Sazonalidade", "AB_pH", "AB_Cor", "AB_Turbidez", "AB_Vazao",
        "DEC_pH", "DEC_Cor", "DEC_Turbidez", "FIL_Cor", "FIL_Turbidez",
        "SAI_pH", "SAI_Cloro", "SAI_Cor", "SAI_Turbidez"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Coluna ausente: {col}"

    # Teste de não-negatividade de turbidez e cor
    assert (df["AB_Turbidez"] >= 0).all()
    assert (df["DEC_Turbidez"] >= 0).all()
    assert (df["SAI_Turbidez"] >= 0).all()


def test_seasonal_splitter(default_data_path):
    loader = ETADatasetLoader(default_data_path)
    df = loader.load_clean_data()

    splitter = SeasonalSplitter()
    seasons = splitter.split_by_season(df)

    assert "chuvoso" in seasons
    assert "seca" in seasons
    assert "transicao_chuva_seca" in seasons
    assert "transicao_seca_chuva" in seasons

    assert len(seasons["chuvoso"]) > 0
    assert len(seasons["seca"]) > 0
    assert len(seasons["transicao_chuva_seca"]) > 0


def test_seasonal_splitter_train_test_split():
    df = pd.DataFrame({"DataHora": pd.date_range("2025-01-01", periods=100, freq="2h"), "valor": range(100)})

    splitter = SeasonalSplitter()
    train, test = splitter.train_test_split(df, train_ratio=0.70)

    assert len(train) == 70
    assert len(test) == 30
    # Split cronológico sem vazamento
    assert train["DataHora"].max() < test["DataHora"].min()


def test_loader_missing_file_raises():
    loader = ETADatasetLoader(Path("caminho_inexistente.csv"))
    try:
        loader.load_clean_data()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Deveria lançar FileNotFoundError")
