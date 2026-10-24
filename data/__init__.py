"""
Módulo de ingestão, limpeza e pré-processamento de séries temporais de qualidade da água.
"""

from .loader import ETADatasetLoader
from .preprocessor import SeasonalSplitter
from .synthetic import generate_synthetic_water_quality_data

__all__ = ["ETADatasetLoader", "SeasonalSplitter", "generate_synthetic_water_quality_data"]
