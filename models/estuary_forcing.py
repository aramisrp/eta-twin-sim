"""
Módulo de Condições de Contorno de Montante e Forçantes Ambientais Estuarinas.
Modela a dinâmica de macromarés semidiurnas (M2/S2), precipitação/lixiviação,
ressuspensão de sedimentos por cisalhamento de fundo e afinidade hidráulica da captação.
"""

import numpy as np


class EstuaryTideModel:
    """
    Modela o ciclo hidrodinâmico de marés estuarinas em canal afluente.
    Combina as componentes harmônicas semidiurnas lunar (M2) e solar (S2).
    """

    def __init__(
        self,
        amplitude_m: float = 2.5,
        m2_period_h: float = 12.42,
        s2_period_h: float = 12.00,
        m2_weight: float = 0.75,
        s2_weight: float = 0.25,
        shear_gain: float = 15.0,
    ):
        self.amplitude_m = amplitude_m
        self.m2_period_h = m2_period_h
        self.s2_period_h = s2_period_h
        self.m2_weight = m2_weight
        self.s2_weight = s2_weight
        self.shear_gain = shear_gain

        self.omega_m2 = 2.0 * np.pi / self.m2_period_h
        self.omega_s2 = 2.0 * np.pi / self.s2_period_h

    def get_water_level(self, t_hours: float | np.ndarray) -> float | np.ndarray:
        """
        Retorna o nível d'água estuarino relativo ao nível médio (metros).
        """
        eta = self.amplitude_m * (
            self.m2_weight * np.sin(self.omega_m2 * t_hours)
            + self.s2_weight * np.sin(self.omega_s2 * t_hours)
        )
        return eta

    def get_shear_turbidity_pulse(
        self, t_hours: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calcula o pulso de ressuspensão coloidal proporcional ao gradiente temporal de nível (|d(eta)/dt|^1.5).
        """
        d_eta_dt = self.amplitude_m * (
            self.m2_weight * self.omega_m2 * np.cos(self.omega_m2 * t_hours)
            + self.s2_weight * self.omega_s2 * np.cos(self.omega_s2 * t_hours)
        )
        shear_pulse = self.shear_gain * (np.abs(d_eta_dt) ** 1.5)
        return shear_pulse


class RainfallRunoffModel:
    """
    Modela o escoamento superficial e a lixiviação de matéria orgânica na bacia estuarina.
    """

    def __init__(self, washoff_gain: float = 0.95, tau_recession_h: float = 14.0):
        self.washoff_gain = washoff_gain
        self.tau_recession_h = tau_recession_h

    def simulate_turbidity_response(
        self, time_hours: np.ndarray, rainfall_mm: np.ndarray
    ) -> np.ndarray:
        """
        Convolução discreta da resposta de turbidez ao sinal de chuva.
        """
        dt = np.diff(time_hours)[0] if len(time_hours) > 1 else 1.0
        response = np.zeros_like(time_hours, dtype=float)
        current_state = 0.0

        alpha = np.exp(-dt / self.tau_recession_h)
        for i, p in enumerate(rainfall_mm):
            current_state = current_state * alpha + self.washoff_gain * p
            response[i] = current_state

        return response


class HydraulicPumpAffinity:
    """
    Modela o conjunto motobomba de captação e o inversor de frequência (VFD) via Leis de Afinidade.
    Potência: P ∝ n³ | Vazão: Q ∝ n
    """

    def __init__(
        self,
        nominal_power_kw: float = 15.0,
        nominal_freq_hz: float = 60.0,
        nominal_flow_lps: float = 25.0,
    ):
        self.nominal_power_kw = nominal_power_kw
        self.nominal_freq_hz = nominal_freq_hz
        self.nominal_flow_lps = nominal_flow_lps

    def calculate_power(self, freq_hz: float) -> float:
        """
        Potência elétrica demandada (kW).
        """
        freq_clipped = np.clip(freq_hz, 25.0, 65.0)
        ratio = freq_clipped / self.nominal_freq_hz
        return float(self.nominal_power_kw * (ratio**3))

    def calculate_flow(self, freq_hz: float) -> float:
        """
        Vazão aduzida (L/s).
        """
        freq_clipped = np.clip(freq_hz, 25.0, 65.0)
        ratio = freq_clipped / self.nominal_freq_hz
        return float(self.nominal_flow_lps * ratio)


class EstuaryBoundaryConditions:
    """
    Orquestrador das condições de contorno ambientais na captação flutuante da planta.
    """

    def __init__(
        self,
        tide_model: EstuaryTideModel = None,
        rain_model: RainfallRunoffModel = None,
        base_turbidity_wet: float = 55.0,
        base_turbidity_dry: float = 30.0,
        base_color_wet: float = 140.0,
        base_color_dry: float = 75.0,
    ):
        self.tide = tide_model or EstuaryTideModel()
        self.rain = rain_model or RainfallRunoffModel()
        self.base_turb_wet = base_turbidity_wet
        self.base_turb_dry = base_turbidity_dry
        self.base_color_wet = base_color_wet
        self.base_color_dry = base_color_dry

    def get_boundary_state(
        self, t_hours: float, season: str = "chuvoso", rainfall_mm: float = 0.0
    ) -> dict[str, float]:
        """
        Calcula o estado completo da água bruta na captação sob as forçantes estuarinas.
        """
        is_wet = "chuv" in season.lower()
        t_base = self.base_turb_wet if is_wet else self.base_turb_dry
        c_base = self.base_color_wet if is_wet else self.base_color_dry

        tide_level = self.tide.get_water_level(t_hours)
        shear_turb = self.tide.get_shear_turbidity_pulse(t_hours)
        rain_turb = self.rain.washoff_gain * rainfall_mm

        total_turb = max(5.0, t_base + shear_turb + rain_turb)
        total_cor = max(10.0, c_base + 0.8 * shear_turb + 1.2 * rain_turb)

        return {
            "ab_turbidez": float(total_turb),
            "ab_cor": float(total_cor),
            "tide_level_m": float(tide_level),
            "shear_pulse": float(shear_turb),
        }
