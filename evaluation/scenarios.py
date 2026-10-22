"""
Executor dos Cenários de Simulação Sazonais e Comparativos da ETA Piloto.
Cenário 1: Baseline Real Histórico (Manual)
Cenário 2: Digital Twin Rotina (Automação Feedforward + Feedback)
Cenário 3: Eventos Extremos de Tempestade e Macromaré Estuarina
"""


import numpy as np
import pandas as pd

from control.digital_twin import DigitalTwinClosedLoop
from models.estuary_forcing import EstuaryBoundaryConditions
from models.reactors import ETAPlantStateSpace

from .metrics import calculate_opex_savings, calculate_portaria_888_compliance


class ScenarioRunner:
    """
    Executa os 3 cenários comparativos e gera as tabelas consolidadas de resultados.
    Recebe parâmetros cinéticos calibrados (k0_sed, lambda_fil, k_decay) que são
    repassados às plantas dos Cenários 1 e 2; defaults físicos quando ausentes.
    """

    def __init__(self, plant_params: dict | None = None):
        self.params = plant_params or {}

    def _build_plant(self) -> ETAPlantStateSpace:
        return ETAPlantStateSpace(
            k0_sed=self.params.get("k0_sed", 6.8),
            lambda_fil=self.params.get("lambda_fil", 1.45),
            k_decay=self.params.get("k_decay_cloro", self.params.get("k_decay", 0.075)),
        )

    def run_scenarios(
        self, df_data: pd.DataFrame, plant_params: dict | None = None
    ) -> dict[str, object]:
        """
        Executa os cenários sobre a série temporal fornecida.
        plant_params (k0_sed, lambda_fil, k_decay[_cloro]) sobrepõem os defaults.
        """
        if plant_params:
            self.params.update(plant_params)

        # =====================================================================
        # CENÁRIO 1: Baseline Histórico (Open-Loop com dosagem estimada real)
        # =====================================================================
        plant_baseline = self._build_plant()
        dos_sulf_manual = np.clip(
            0.35 * df_data["AB_Turbidez"].values + 0.15 * df_data["AB_Cor"].values,
            0.0,
            80.0,
        )
        dos_cl_manual = np.full(len(df_data), 2.0)

        s1_records = []
        for i in range(len(df_data)):
            row = df_data.iloc[i]
            inp = {
                "ab_turb": float(row["AB_Turbidez"]),
                "ab_cor": float(row["AB_Cor"]),
                "ab_ph": float(row["AB_pH"]),
                "ab_flow": float(row["AB_Vazao"]),
                "dos_sulfato": float(dos_sulf_manual[i]),
                "dos_cloro": float(dos_cl_manual[i]),
            }
            out = plant_baseline.step(inp, dt_hours=2.0)
            out["dos_sulfato_applied"] = dos_sulf_manual[i]
            out["dos_cloro_applied"] = dos_cl_manual[i]
            s1_records.append(out)

        df_s1 = pd.DataFrame(s1_records)

        # =====================================================================
        # CENÁRIO 2: Digital Twin Rotina (Closed-Loop Feedforward + Feedback)
        # =====================================================================
        plant_twin = self._build_plant()
        twin = DigitalTwinClosedLoop(plant=plant_twin)

        s2_records = []
        for i in range(len(df_data)):
            row = df_data.iloc[i]
            inp = {
                "ab_turb": float(row["AB_Turbidez"]),
                "ab_cor": float(row["AB_Cor"]),
                "ab_ph": float(row["AB_pH"]),
                "ab_flow": float(row["AB_Vazao"]),
            }
            out = twin.step(inp, dt_hours=2.0)
            s2_records.append(out)

        df_s2 = pd.DataFrame(s2_records)

        # =====================================================================
        # CENÁRIO 3: Eventos Extremos de Tempestade e Macromaré (Choques)
        # =====================================================================
        boundary = EstuaryBoundaryConditions()
        plant_storm = ETAPlantStateSpace()
        twin_storm = DigitalTwinClosedLoop(plant=plant_storm)

        n_steps = min(len(df_data), 120) # 10 dias a cada 2h
        time_hours = np.linspace(0, n_steps * 2.0, n_steps)
        # Injeção de chuva severa de 70mm nas horas 24 a 36
        rain_pulse = np.zeros(n_steps)
        rain_pulse[12:18] = 70.0

        s3_records = []
        for i, t in enumerate(time_hours):
            b_state = boundary.get_boundary_state(
                t_hours=t, season="chuvoso", rainfall_mm=rain_pulse[i]
            )
            inp = {
                "ab_turb": b_state["ab_turbidez"],
                "ab_cor": b_state["ab_cor"],
                "ab_ph": 6.6,
                "ab_flow": 28.0 if b_state["tide_level_m"] > 0 else 18.0, # Variação de maré na bomba
            }
            out = twin_storm.step(inp, dt_hours=2.0)
            out["time_hours"] = t
            out["ab_turb_injected"] = inp["ab_turb"]
            out["ab_cor_injected"] = inp["ab_cor"]
            out["tide_level_m"] = b_state["tide_level_m"]
            s3_records.append(out)

        df_s3 = pd.DataFrame(s3_records)

        # =====================================================================
        # Consolidação de Métricas e Tabela Comparativa
        # =====================================================================
        comp_s1 = calculate_portaria_888_compliance(df_s1)
        comp_s2 = calculate_portaria_888_compliance(df_s2)
        comp_s3 = calculate_portaria_888_compliance(df_s3)

        opex_sulf = calculate_opex_savings(
            dosage_real_mg_l=df_s1["dos_sulfato_applied"].values,
            dosage_twin_mg_l=df_s2["dos_sulfato_applied"].values,
            flow_lps=df_data["AB_Vazao"].values,
        )

        opex_cloro = calculate_opex_savings(
            dosage_real_mg_l=df_s1["dos_cloro_applied"].values,
            dosage_twin_mg_l=df_s2["dos_cloro_applied"].values,
            flow_lps=df_data["AB_Vazao"].values,
        )

        summary_data = [
            {
                "Cenario": "1. Baseline Historico (Manual)",
                "Conformidade_Turb_0_5_pct": comp_s1.get("turbidity_0_5_compliance_pct", 0.0),
                "Conformidade_Turb_5_0_pct": comp_s1.get("turbidity_5_0_compliance_pct", 0.0),
                "Conformidade_Cloro_pct": comp_s1.get("chlorine_compliance_pct", 0.0),
                "Conformidade_pH_pct": comp_s1.get("ph_compliance_pct", 0.0),
                "Economia_Sulfato_vs_Manual_pct": 0.0,
                "Economia_Cloro_vs_Manual_pct": 0.0,
            },
            {
                "Cenario": "2. Digital Twin (Feedforward + Feedback)",
                "Conformidade_Turb_0_5_pct": comp_s2.get("turbidity_0_5_compliance_pct", 0.0),
                "Conformidade_Turb_5_0_pct": comp_s2.get("turbidity_5_0_compliance_pct", 0.0),
                "Conformidade_Cloro_pct": comp_s2.get("chlorine_compliance_pct", 0.0),
                "Conformidade_pH_pct": comp_s2.get("ph_compliance_pct", 0.0),
                "Economia_Sulfato_vs_Manual_pct": opex_sulf.get("relative_saving_pct", 0.0),
                "Economia_Cloro_vs_Manual_pct": opex_cloro.get("relative_saving_pct", 0.0),
            },
            {
                "Cenario": "3. Choque Extremo (Tempestade + Mare)",
                "Conformidade_Turb_0_5_pct": comp_s3.get("turbidity_0_5_compliance_pct", 0.0),
                "Conformidade_Turb_5_0_pct": comp_s3.get("turbidity_5_0_compliance_pct", 0.0),
                "Conformidade_Cloro_pct": comp_s3.get("chlorine_compliance_pct", 0.0),
                "Conformidade_pH_pct": comp_s3.get("ph_compliance_pct", 0.0),
                "Economia_Sulfato_vs_Manual_pct": 0.0,
                "Economia_Cloro_vs_Manual_pct": 0.0,
            },
        ]
        df_summary = pd.DataFrame(summary_data)

        return {
            "baseline_historical": df_s1,
            "digital_twin_routine": df_s2,
            "extreme_storm_tide": df_s3,
            "opex_sulfato": opex_sulf,
            "opex_cloro": opex_cloro,
            "summary_table": df_summary,
        }
