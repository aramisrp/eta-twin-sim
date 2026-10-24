"""
Pytest Fixtures for Self-Contained Test Execution.
Provides fallback synthetic data when proprietary operational datasets are absent.
"""

from pathlib import Path
import pytest
from data.synthetic import generate_synthetic_water_quality_data


@pytest.fixture(scope="session")
def default_data_path(tmp_path_factory):
    artigo_dir = Path(__file__).parents[3]
    candidate = artigo_dir / "02_Series_Historicas_e_Estatistica" / "Dados" / "3_Ouro" / "ouro_wide.csv"
    if candidate.exists():
        return candidate

    temp_dir = tmp_path_factory.mktemp("data")
    synthetic_path = temp_dir / "synthetic_water_quality.csv"
    generate_synthetic_water_quality_data(synthetic_path, num_records=2400)
    return synthetic_path
