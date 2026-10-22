"""
Mapeamento estequiométrico e empírico de controle antecipatório (Feedforward).
Calcula dosagens de sulfato de alumínio e cloro antes que as perturbações entrem no decantador.
"""

import numpy as np


class CoagulantFeedforward:
    """
    Controle Feedforward de Sulfato de Alumínio baseado na turbidez e cor da água bruta e vazão de adução.
    """

    def __init__(
        self,
        a0: float = 6.0,
        a1: float = 3.5,
        a2: float = 0.18,
        ref_flow_lps: float = 25.0,
        max_dosage: float = 80.0,
    ):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.ref_flow = ref_flow_lps
        self.max_dosage = max_dosage

    def compute_dosage(self, raw_turb: float, raw_cor: float, flow_lps: float) -> float:
        """
        D_FF = a0 + a1 * sqrt(Turb) + a2 * Cor * (Q / Q_ref)
        """
        turb = max(0.0, raw_turb)
        cor = max(0.0, raw_cor)
        flow_factor = max(0.1, flow_lps) / self.ref_flow

        dosage = self.a0 + self.a1 * np.sqrt(turb) + self.a2 * cor * flow_factor
        return float(np.clip(dosage, 0.0, self.max_dosage))


class ChlorineFeedforward:
    """
    Controle Feedforward de Hipoclorito de Sódio baseado na demanda da matéria orgânica (Cor) e setpoint.
    """

    def __init__(
        self,
        b0: float = 0.6,
        b1: float = 0.008,
        b2: float = 1.05,
        max_dosage: float = 10.0,
    ):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.max_dosage = max_dosage

    def compute_dosage(self, raw_cor: float, target_cloro: float = 1.5) -> float:
        """
        D_FF_cloro = b0 + b1 * Cor + b2 * Cl_target
        """
        cor = max(0.0, raw_cor)
        dosage = self.b0 + self.b1 * cor + self.b2 * target_cloro
        return float(np.clip(dosage, 0.0, self.max_dosage))
