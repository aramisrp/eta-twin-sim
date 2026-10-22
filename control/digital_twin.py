"""
Orquestrador em Malha Fechada do Gêmeo Digital (Digital Twin) de ETA Compacta.
Integra Feedforward antecipatório + Feedback PI nos decantadores e desinfecção.
"""


import numpy as np

from models.reactors import ETAPlantStateSpace

from .feedforward import ChlorineFeedforward, CoagulantFeedforward
from .pid_feedback import DiscretePIDWithAntiWindup


class DigitalTwinClosedLoop:
    """
    Simulador em malha fechada que atua sobre a planta ETA através do controle composto.
    """

    def __init__(
        self,
        plant: ETAPlantStateSpace | None = None,
        target_turbidity_ntu: float = 0.5,
        target_chlorine_mg_l: float = 1.5,
        max_sulfato_mg_l: float = 80.0,
        max_cloro_mg_l: float = 10.0,
    ):
        self.plant = plant or ETAPlantStateSpace()
        self.target_turbidity = target_turbidity_ntu
        self.target_chlorine = target_chlorine_mg_l

        # Controladores Feedforward
        self.ff_sulfato = CoagulantFeedforward(max_dosage=max_sulfato_mg_l)
        self.ff_cloro = ChlorineFeedforward(max_dosage=max_cloro_mg_l)

        # Controladores Feedback (PI)
        # Para turbidez, ação reversa: erro = (Turb_medida - Turb_alvo) -> erro > 0
        # exige aumento de coagulante (u_fb > 0). Anti-windup preservado via update().
        self.pid_turb = DiscretePIDWithAntiWindup(
            kp=1.2, ki=0.10, kd=0.02, u_min=-30.0, u_max=40.0, reverse_action=True
        )
        # Para cloro: erro = (Cl_alvo - Cl_medido) -> erro > 0 exige aumento de dosador de cloro
        self.pid_cloro = DiscretePIDWithAntiWindup(
            kp=0.9, ki=0.12, kd=0.01, u_min=-3.0, u_max=5.0
        )

        self.max_sulfato = max_sulfato_mg_l
        self.max_cloro = max_cloro_mg_l

    def step(self, raw_water_inputs: dict[str, float], dt_hours: float = 2.0) -> dict[str, float]:
        """
        Executa um passo completo de controle e simulação da planta.
        """
        ab_turb = raw_water_inputs.get("ab_turb", 50.0)
        ab_cor = raw_water_inputs.get("ab_cor", 130.0)
        ab_ph = raw_water_inputs.get("ab_ph", 6.8)
        flow = raw_water_inputs.get("ab_flow", 25.0)

        # 1. Ação Feedforward (Antecipatória)
        u_ff_sulf = self.ff_sulfato.compute_dosage(ab_turb, ab_cor, flow)
        u_ff_cl = self.ff_cloro.compute_dosage(ab_cor, self.target_chlorine)

        # 2. Leitura dos estados anteriores da planta para Feedback
        prev_dec_turb = getattr(self.plant.decanter, "state_turb", 8.0)
        prev_sai_cl = getattr(self.plant.reservoir, "state_cloro", 1.5)

        # 3. Ação Feedback (Correção por Erro Residual)
        # Decantador target = 3.0 NTU (para garantir que filtro entregue < 0.5 NTU)
        u_fb_sulf = self.pid_turb.update(
            setpoint=3.0, measured=prev_dec_turb, dt_hours=dt_hours
        )

        u_fb_cl = self.pid_cloro.update(
            setpoint=self.target_chlorine, measured=prev_sai_cl, dt_hours=dt_hours
        )

        # 4. Dosagem Composta Final Saturada nos Limites dos Atuadores
        dos_sulf_final = float(np.clip(u_ff_sulf + u_fb_sulf, 0.0, self.max_sulfato))
        dos_cl_final = float(np.clip(u_ff_cl + u_fb_cl, 0.0, self.max_cloro))

        # 5. Aplicação das Dosagens na Planta Físico-Química
        plant_inputs = {
            "ab_turb": ab_turb,
            "ab_cor": ab_cor,
            "ab_ph": ab_ph,
            "ab_flow": flow,
            "dos_sulfato": dos_sulf_final,
            "dos_cloro": dos_cl_final,
        }
        plant_state = self.plant.step(plant_inputs, dt_hours=dt_hours)

        output = {
            **plant_state,
            "dos_sulfato_applied": dos_sulf_final,
            "dos_cloro_applied": dos_cl_final,
            "u_ff_sulfato": u_ff_sulf,
            "u_fb_sulfato": u_fb_sulf,
            "u_ff_cloro": u_ff_cl,
            "u_fb_cloro": u_fb_cl,
        }
        return output
