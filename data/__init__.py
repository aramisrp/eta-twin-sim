"""
Módulo de ingestão, limpeza e pré-processamento de séries temporais de qualidade da água.
"""

from .loader import ETADatasetLoader
from .preprocessor import SeasonalSplitter

__all__ = ["ETADatasetLoader", "SeasonalSplitter"]
