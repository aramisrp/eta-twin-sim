"""
Controlador PID Discreto com Anti-Windup (Clamping) e Saturação de Atuador.
"""

import numpy as np


class DiscretePIDWithAntiWindup:
    """
    Controlador PID com amostragem discreta e proteção anti-windup para bombas dosadoras.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.0,
        u_min: float = 0.0,
        u_max: float = 80.0,
        reverse_action: bool = False,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.u_min = u_min
        self.u_max = u_max
        self.reverse_action = reverse_action

        self.integral = 0.0
        self.prev_error = 0.0
        self.is_saturated = False

    def reset(self):
        """Reinicializa os estados do integrador e derivada."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.is_saturated = False

    def update(self, setpoint: float, measured: float, dt_hours: float = 1.0) -> float:
        """
        Calcula a ação de controle u(t) com clamping do integrador.
        Por padrão, ação direta (erro = setpoint - measured).
        Com reverse_action=True, ação reversa (erro = measured - setpoint),
        preservando a lógica de anti-windup sobre o erro interno.
        """
        error = (measured - setpoint) if self.reverse_action else (setpoint - measured)
        dt = max(1e-4, dt_hours)

        # Proporcional
        p_term = self.kp * error

        # Derivativo
        d_term = self.kd * (error - self.prev_error) / dt

        # Cálculo tentativo do integrador
        integral_candidate = self.integral + error * dt
        i_term = self.ki * integral_candidate

        # Sinal não saturado
        u_unsaturated = p_term + i_term + d_term

        # Saturação estrita
        u_saturated = float(np.clip(u_unsaturated, self.u_min, self.u_max))

        # Clamping Anti-Windup: só atualiza a integral se não estiver saturado
        # ou se o erro estiver no sentido de desaturar
        is_saturating_high = u_unsaturated > self.u_max and error > 0
        is_saturating_low = u_unsaturated < self.u_min and error < 0

        if not (is_saturating_high or is_saturating_low):
            self.integral = integral_candidate

        self.prev_error = error
        return u_saturated
