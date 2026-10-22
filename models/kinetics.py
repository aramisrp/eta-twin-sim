"""
Cinéticas físico-químicas de coagulação/sedimentação, filtração e decaimento de cloro.
Baseado em Crittenden et al. (2022/2023), Edzwald (2011) e Haas et al. (2014).
"""


import numpy as np


class CoagulationKinetics:
    """
    Cinética de coagulação com Sulfato de Alumínio e sedimentação em decantadores lamelares de alta taxa.
    """

    def __init__(
        self,
        k0: float = 6.8,
        d_sat: float = 24.0,
        gamma: float = 0.85,
        nominal_flow_lps: float = 25.0,
    ):
        self.k0 = k0
        self.d_sat = d_sat
        self.gamma = gamma
        self.nominal_flow = nominal_flow_lps

    def calculate_ksed(
        self, dosage_sulfato: float | np.ndarray, flow_lps: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Taxa de remoção de turbidez k_sed (h^-1).
        k_sed = k0 * [1 - exp(-D / D_sat)] * (Q_nom / Q)^gamma
        """
        dos = np.maximum(0.0, dosage_sulfato)
        q = np.maximum(1.0, flow_lps)
        eff_coag = 1.0 - np.exp(-dos / self.d_sat)
        flow_factor = (self.nominal_flow / q) ** self.gamma
        return self.k0 * eff_coag * flow_factor

    def calculate_kcolor(
        self, dosage_sulfato: float | np.ndarray, ph: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Taxa de remoção de cor aparente k_cor (h^-1).
        Ótimo em faixa de pH 5.8 - 6.8 típica de coagulação por sulfato.
        """
        dos = np.maximum(0.0, dosage_sulfato)
        ph_factor = np.exp(-0.5 * ((ph - 6.2) / 0.8) ** 2)
        eff_coag = 1.0 - np.exp(-dos / (self.d_sat * 0.9))
        return self.k0 * 1.1 * eff_coag * ph_factor


class ChlorineDecayKinetics:
    """
    Cinética de decaimento de Cloro Residual Livre e demanda imediata de oxidação.
    """

    def __init__(self, k_base: float = 0.075, immediate_demand_factor: float = 0.015):
        self.k_base = k_base
        self.immediate_demand_factor = immediate_demand_factor

    def calculate_kdecay(
        self, turbidity: float | np.ndarray, ph: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Taxa de decaimento de 1ª ordem k_decay (h^-1).
        """
        turb_effect = 1.0 + 0.25 * np.maximum(0.0, turbidity)
        ph_effect = 1.0 + 0.15 * np.abs(ph - 7.0)
        return self.k_base * turb_effect * ph_effect

    def calculate_immediate_demand(self, color: float | np.ndarray) -> float | np.ndarray:
        """
        Demanda imediata de cloro em função da carga orgânica residual (uC).
        """
        return self.immediate_demand_factor * np.maximum(0.0, color)


class PHBalanceModel:
    """
    Balanço estequiométrico de pH e consumo de alcalinidade.
    """

    def __init__(self, alpha_sulfato: float = 0.018, beta_cloro: float = 0.030):
        self.alpha_sulfato = alpha_sulfato # Queda de pH por mg/L de sulfato de alumínio
        self.beta_cloro = beta_cloro       # Elevação de pH por mg/L de hipoclorito de sódio

    def calculate_ph_out(
        self, ph_raw: float, dos_sulfato: float, dos_cloro: float
    ) -> float:
        """
        Calcula o pH da água tratada na saída da planta.
        """
        delta_ph = -self.alpha_sulfato * dos_sulfato + self.beta_cloro * dos_cloro
        ph_final = np.clip(ph_raw + delta_ph, 4.5, 9.5)
        return float(ph_final)
