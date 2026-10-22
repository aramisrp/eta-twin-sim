"""
Módulo de calibração não-linear de parâmetros cinéticos do modelo Gray-Box.
Ajusta k0_sed, lambda_fil, k_decay a partir da série real ouro_wide.csv via scipy.optimize.
"""


import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .reactors import ETAPlantStateSpace


class GrayBoxCalibrator:
    """
    Calibra os parâmetros cinéticos k0_sed, lambda_fil e k_decay_cloro minimizando o RMSE
    entre as medições históricas de decantador, filtro e saída e as predições do simulador.
    """

    def __init__(self, v_dec: float = 70.0, v_fil: float = 30.0, v_res: float = 800.0):
        self.v_dec = v_dec
        self.v_fil = v_fil
        self.v_res = v_res

    def calibrate(self, df_historical: pd.DataFrame) -> dict[str, float]:
        """
        Executa a otimização não-linear por mínimos quadrados (L-BFGS-B),
        com múltiplos pontos de partida, penalidade de fronteira e o estágio
        SAI_Turbidez incluído no objetivo.
        """
        # Estimativas de dosagem com base na carga inicial se não houver coluna explícita
        dos_sulf_est = np.clip(
            0.35 * df_historical["AB_Turbidez"].values + 0.15 * df_historical["AB_Cor"].values,
            0.0,
            80.0,
        )
        dos_cl_est = np.full(len(df_historical), 2.0)

        ab_turb = df_historical["AB_Turbidez"].values
        ab_cor = df_historical["AB_Cor"].values
        ab_ph = df_historical["AB_pH"].values
        flow = df_historical["AB_Vazao"].values

        target_dec_turb = df_historical["DEC_Turbidez"].values
        target_fil_turb = df_historical["FIL_Turbidez"].values
        target_sai_turb = df_historical["SAI_Turbidez"].values
        target_sai_cl = df_historical["SAI_Cloro"].values

        # Limites físicos amplos (revisados): a solução não deve encostar na fronteira
        bounds = [(0.5, 12.0), (0.1, 4.0), (0.005, 0.50)]
        eps_frac = 0.02  # fração da amplitude considerada "fronteira"

        def boundary_penalty(params):
            pen = 0.0
            for p, (lo, hi) in zip(params, bounds):
                width = hi - lo
                if p <= lo + eps_frac * width:
                    pen += 1e3 * ((lo + eps_frac * width - p) / width) ** 2
                if p >= hi - eps_frac * width:
                    pen += 1e3 * ((p - (hi - eps_frac * width)) / width) ** 2
            return pen

        def objective(params):
            k0_sed, lambda_fil, k_decay = params
            plant = ETAPlantStateSpace(
                v_dec=self.v_dec,
                v_fil=self.v_fil,
                v_res=self.v_res,
                k0_sed=k0_sed,
                lambda_fil=lambda_fil,
                k_decay=k_decay,
            )

            sim_dec_turb = []
            sim_fil_turb = []
            sim_sai_turb = []
            sim_sai_cl = []

            for i in range(len(ab_turb)):
                inp = {
                    "ab_turb": float(ab_turb[i]),
                    "ab_cor": float(ab_cor[i]),
                    "ab_ph": float(ab_ph[i]),
                    "ab_flow": float(flow[i]),
                    "dos_sulfato": float(dos_sulf_est[i]),
                    "dos_cloro": float(dos_cl_est[i]),
                }
                out = plant.step(inp, dt_hours=2.0)
                sim_dec_turb.append(out["dec_turb"])
                sim_fil_turb.append(out["fil_turb"])
                sim_sai_turb.append(out["sai_turb"])
                sim_sai_cl.append(out["sai_cloro"])

            err_dec = np.mean((np.array(sim_dec_turb) - target_dec_turb) ** 2)
            err_fil = np.mean((np.array(sim_fil_turb) - target_fil_turb) ** 2)
            err_sai = np.mean((np.array(sim_sai_turb) - target_sai_turb) ** 2)
            err_cl = np.mean((np.array(sim_sai_cl) - target_sai_cl) ** 2)

            return err_dec + 2.0 * err_fil + 1.0 * err_sai + err_cl + boundary_penalty(params)

        # Múltiplos chutes iniciais (multistart) e seleção do melhor
        starts = [
            [2.2, 1.45, 0.075],
            [6.0, 0.8, 0.05],
            [10.0, 0.4, 0.02],
        ]
        best = None
        for x0 in starts:
            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-10},
            )
            if best is None or res.fun < best.fun:
                best = res

        calibrated_params = {
            "k0_sed": float(best.x[0]),
            "lambda_fil": float(best.x[1]),
            "k_decay_cloro": float(best.x[2]),
            "success": bool(best.success),
            "final_cost": float(best.fun),
        }
        return calibrated_params
