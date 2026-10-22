"""
Módulo de avaliação de métricas estatísticas, conformidade sanitária e cenários do simulador.
"""

from .metrics import (
    calculate_metrics,
    calculate_opex_savings,
    calculate_portaria_888_compliance,
)
from .scenarios import ScenarioRunner

__all__ = [
    "ScenarioRunner",
    "calculate_metrics",
    "calculate_opex_savings",
    "calculate_portaria_888_compliance",
]
