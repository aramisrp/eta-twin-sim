"""
Modelos de Reatores Hidráulicos Contínuos em Espaço de Estados (CSTRs e PFRs).
Representa a física dos 2 flocodecantadores em paralelo, filtros rápidos e tanques de contato.
"""

import numpy as np

from .kinetics import ChlorineDecayKinetics, CoagulationKinetics, PHBalanceModel

TURB_SENSOR_FLOOR = 0.01
"""Limite analítico de turbidez (NTU): resolução típica de turbidímetro (ex.: Hach 2100Q)."""


class LamellaDecanterCSTR:
    """
    Decantador de placas paralelas cilíndrico modelado como CSTR dinâmico com termo de sedimentação.
    Suporta N unidades em paralelo (planta padrão = 2 unidades de 35 m³ cada).
    """

    def __init__(
        self,
        unit_volume_m3: float = 35.0,
        num_parallel_units: int = 2,
        kinetics: CoagulationKinetics | None = None,
    ):
        self.unit_volume_m3 = unit_volume_m3
        self.num_parallel_units = num_parallel_units
        self.total_volume_m3 = unit_volume_m3 * num_parallel_units
        self.kinetics = kinetics or CoagulationKinetics()

        # Estados internos (inicializados em valores médios de água decantada)
        self.state_turb = 8.0
        self.state_cor = 35.0
        self.state_ph = 6.0

    def step(
        self,
        turb_in: float,
        color_in: float,
        flow_lps: float,
        dos_sulfato: float,
        dt_hours: float = 2.0,
        ph_in: float = 6.8,
    ) -> dict[str, float]:
        """
        Integração analítica/numérica do balanço de massa no decantador:
        dC/dt = (Q/V)*(C_in - C) - k*C
        Solução no regime quase-estacionário ou discretizado:
        C(t+dt) = C_in / (1 + k * tau) + [C(t) - C_in / (1 + k * tau)] * exp(-(1/tau + k)*dt)
        """
        flow_m3_h = flow_lps * 3.6
        tau = self.total_volume_m3 / max(1.0, flow_m3_h)

        k_sed = self.kinetics.calculate_ksed(dos_sulfato, flow_lps)
        k_cor = self.kinetics.calculate_kcolor(dos_sulfato, ph_in)

        # Atualização Turbidez
        c_ss_turb = turb_in / (1.0 + k_sed * tau)
        eig_turb = (1.0 / tau) + k_sed
        self.state_turb = float(
            c_ss_turb + (self.state_turb - c_ss_turb) * np.exp(-eig_turb * dt_hours)
        )

        # Atualização Cor
        c_ss_cor = color_in / (1.0 + k_cor * tau)
        eig_cor = (1.0 / tau) + k_cor
        self.state_cor = float(
            c_ss_cor + (self.state_cor - c_ss_cor) * np.exp(-eig_cor * dt_hours)
        )

        # pH no decantador
        self.state_ph = float(np.clip(ph_in - 0.02 * dos_sulfato, 4.5, 8.5))

        return {
            "dec_turb": float(np.maximum(TURB_SENSOR_FLOOR, self.state_turb)),
            "dec_cor": float(np.maximum(1.0, self.state_cor)),
            "dec_ph": float(self.state_ph),
            "tau_dec_hours": float(tau),
        }


class RapidSandFilter:
    """
    Filtro granular rápido ascendente/descendente (Equação de Clarificação de Iwasaki).
    """

    def __init__(
        self,
        lambda_fil: float = 1.45,
        bed_height_m: float = 1.2,
        color_removal_ratio: float = 0.45,
        nominal_flow_lps: float = 25.0,
    ):
        self.lambda_fil = lambda_fil
        self.bed_height_m = bed_height_m
        self.color_removal_ratio = color_removal_ratio
        self.nominal_flow = nominal_flow_lps

    def step(
        self, turb_in: float, color_in: float, flow_lps: float
    ) -> dict[str, float]:
        """
        Remoção de turbidez profunda: T_fil = T_dec * exp(-lambda * L * (Q_nom / Q)^0.5)
        """
        q_ratio = (self.nominal_flow / max(1.0, flow_lps)) ** 0.5
        turb_out = turb_in * np.exp(-self.lambda_fil * self.bed_height_m * q_ratio)
        cor_out = color_in * (1.0 - self.color_removal_ratio)

        return {
            "fil_turb": float(np.maximum(TURB_SENSOR_FLOOR, turb_out)),
            "fil_cor": float(np.maximum(1.0, cor_out)),
        }


class ChlorineContactReservoir:
    """
    Tanque de contato e reservatório de água tratada (600 m³ semienterrado + 200 m³ apoiado = 800 m³).
    Balanço dinâmico de Cloro Residual Livre e pH de distribuição.
    """

    def __init__(
        self,
        total_volume_m3: float = 800.0,
        kinetics: ChlorineDecayKinetics | None = None,
        ph_model: PHBalanceModel | None = None,
    ):
        self.total_volume_m3 = total_volume_m3
        self.kinetics = kinetics or ChlorineDecayKinetics()
        self.ph_model = ph_model or PHBalanceModel()

        self.state_cloro = 1.5
        self.state_ph = 6.2

    def step(
        self,
        dos_cloro: float,
        fil_turb: float,
        fil_cor: float,
        flow_lps: float,
        dt_hours: float = 2.0,
        ph_raw: float = 6.8,
        dos_sulfato: float = 30.0,
    ) -> dict[str, float]:
        """
        Balanço de massa de Cloro Residual Livre no reservatório de contato:
        dCl/dt = (Q/V)*(D_cloro - Demanda_imediata - Cl) - k_decay * Cl
        """
        flow_m3_h = flow_lps * 3.6
        tau = self.total_volume_m3 / max(1.0, flow_m3_h)

        imm_demand = self.kinetics.calculate_immediate_demand(fil_cor)
        eff_cl_in = max(0.0, dos_cloro - imm_demand)

        k_dec = self.kinetics.calculate_kdecay(fil_turb, self.state_ph)

        cl_ss = eff_cl_in / (1.0 + k_dec * tau)
        eig_cl = (1.0 / tau) + k_dec
        self.state_cloro = float(
            cl_ss + (self.state_cloro - cl_ss) * np.exp(-eig_cl * dt_hours)
        )

        # pH de saída
        self.state_ph = self.ph_model.calculate_ph_out(
            ph_raw=ph_raw, dos_sulfato=dos_sulfato, dos_cloro=dos_cloro
        )

        return {
            "sai_cloro": float(np.clip(self.state_cloro, 0.0, 5.0)),
            "sai_ph": float(self.state_ph),
            "sai_turb": float(np.maximum(TURB_SENSOR_FLOOR, fil_turb * 0.95)),
            "sai_cor": float(np.maximum(1.0, fil_cor * 0.90)),
        }


class ETAPlantStateSpace:
    """
    Planta completa de tratamento de água acoplando Decantador -> Filtro -> Tanque de Contato/Saída.
    """

    def __init__(
        self,
        v_dec: float = 70.0,
        v_fil: float = 30.0,
        v_res: float = 800.0,
        k0_sed: float = 6.8,
        lambda_fil: float = 1.45,
        k_decay: float = 0.075,
    ):
        self.decanter = LamellaDecanterCSTR(
            unit_volume_m3=v_dec / 2.0,
            num_parallel_units=2,
            kinetics=CoagulationKinetics(k0=k0_sed),
        )
        self.filter = RapidSandFilter(lambda_fil=lambda_fil)
        self.reservoir = ChlorineContactReservoir(
            total_volume_m3=v_res,
            kinetics=ChlorineDecayKinetics(k_base=k_decay),
        )

    def step(self, inputs: dict[str, float], dt_hours: float = 2.0) -> dict[str, float]:
        """
        Avança o estado físico-químico da planta por um passo de tempo dt_hours.
        """
        ab_turb = inputs.get("ab_turb", 50.0)
        ab_cor = inputs.get("ab_cor", 130.0)
        ab_ph = inputs.get("ab_ph", 6.8)
        flow = inputs.get("ab_flow", 25.0)
        dos_sulf = inputs.get("dos_sulfato", 35.0)
        dos_cl = inputs.get("dos_cloro", 2.0)

        # 1. Decantadores em Paralelo
        dec_out = self.decanter.step(
            turb_in=ab_turb,
            color_in=ab_cor,
            flow_lps=flow,
            dos_sulfato=dos_sulf,
            dt_hours=dt_hours,
            ph_in=ab_ph,
        )

        # 2. Bateria de Filtros
        fil_out = self.filter.step(
            turb_in=dec_out["dec_turb"],
            color_in=dec_out["dec_cor"],
            flow_lps=flow,
        )

        # 3. Desinfecção e Saída para Distribuição
        res_out = self.reservoir.step(
            dos_cloro=dos_cl,
            fil_turb=fil_out["fil_turb"],
            fil_cor=fil_out["fil_cor"],
            flow_lps=flow,
            dt_hours=dt_hours,
            ph_raw=ab_ph,
            dos_sulfato=dos_sulf,
        )

        result = {**dec_out, **fil_out, **res_out}
        return result
