"""
Módulo de controle do Gêmeo Digital (Feedforward + Feedback PI/PID).
"""

from .digital_twin import DigitalTwinClosedLoop
from .feedforward import ChlorineFeedforward, CoagulantFeedforward
from .pid_feedback import DiscretePIDWithAntiWindup

__all__ = [
    "ChlorineFeedforward",
    "CoagulantFeedforward",
    "DigitalTwinClosedLoop",
    "DiscretePIDWithAntiWindup",
]
