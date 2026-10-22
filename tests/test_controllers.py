from control.digital_twin import DigitalTwinClosedLoop
from control.feedforward import ChlorineFeedforward, CoagulantFeedforward
from control.pid_feedback import DiscretePIDWithAntiWindup
from models.reactors import ETAPlantStateSpace


def test_feedforward_mapping():
    ff_sulf = CoagulantFeedforward(a0=5.0, a1=3.2, a2=0.15)
    dos_sulf = ff_sulf.compute_dosage(raw_turb=60.0, raw_cor=140.0, flow_lps=25.0)
    assert 10.0 < dos_sulf < 60.0

    ff_cl = ChlorineFeedforward(b0=0.8, b1=0.008, b2=1.0)
    dos_cl = ff_cl.compute_dosage(raw_cor=140.0, target_cloro=1.5)
    assert 1.0 < dos_cl < 4.0

def test_pid_feedback_anti_windup_and_saturation():
    # PID para Cloro: Setpoint 1.5 mg/L, medido 0.5 mg/L -> erro positivo -> aumenta dosagem
    pid_cl = DiscretePIDWithAntiWindup(
        kp=1.2, ki=0.15, kd=0.05, u_min=0.0, u_max=10.0
    )
    
    # 20 passos com erro constante grande
    for _ in range(20):
        u = pid_cl.update(setpoint=1.5, measured=0.2, dt_hours=0.5)
    assert u <= 10.0 # Saturação estrita
    
    # Agora a medição sobe repentinamente acima do setpoint
    u_after = pid_cl.update(setpoint=1.5, measured=2.2, dt_hours=0.5)
    assert u_after < 10.0 # Resposta descola imediatamente sem windup atrasado

def test_pid_reset_and_reverse_action():
    pid = DiscretePIDWithAntiWindup(kp=1.0, ki=0.5, kd=0.0, u_min=-40.0, u_max=40.0)
    for _ in range(10):
        pid.update(setpoint=1.0, measured=0.2, dt_hours=1.0)

    pid.reset()
    assert pid.integral == 0.0
    assert pid.prev_error == 0.0
    assert not pid.is_saturated

    # Ação reversa: erro = measured - setpoint
    pid_rev = DiscretePIDWithAntiWindup(kp=1.0, ki=0.0, kd=0.0, reverse_action=True)
    u = pid_rev.update(setpoint=3.0, measured=6.0, dt_hours=1.0)
    assert u == 3.0

def test_digital_twin_closed_loop_simulation():
    plant = ETAPlantStateSpace()
    twin = DigitalTwinClosedLoop(
        plant=plant,
        target_turbidity_ntu=0.5,
        target_chlorine_mg_l=1.5,
    )
    
    # Rodar um ciclo de 10 passos com perturbação
    raw_inputs = [
        {"ab_turb": 40.0, "ab_cor": 110.0, "ab_ph": 6.8, "ab_flow": 25.0},
        {"ab_turb": 80.0, "ab_cor": 160.0, "ab_ph": 6.7, "ab_flow": 25.0}, # Pico de chuva
        {"ab_turb": 120.0, "ab_cor": 200.0, "ab_ph": 6.6, "ab_flow": 28.0}, # Pico de maré
        {"ab_turb": 60.0, "ab_cor": 130.0, "ab_ph": 6.8, "ab_flow": 25.0},
    ]
    
    history = []
    for inp in raw_inputs:
        res = twin.step(inp, dt_hours=2.0)
        history.append(res)
        assert 0.0 < res["sai_turb"] < 6.0
        assert 0.2 <= res["sai_cloro"] <= 2.5
        assert res["dos_sulfato_applied"] <= 80.0
        assert res["dos_cloro_applied"] <= 10.0

def test_digital_twin_turbidity_pid_no_windup():
    # Regressão: o feedback de turbidez deve respeitar os limites declarados
    # u_min=-30 / u_max=40 (anti-windup via update() com ação reversa)
    plant = ETAPlantStateSpace()
    twin = DigitalTwinClosedLoop(plant=plant)

    # Perturbação persistente de alta turbidez por 120 passos
    for _ in range(120):
        res = twin.step(
            {"ab_turb": 90.0, "ab_cor": 160.0, "ab_ph": 6.8, "ab_flow": 25.0},
            dt_hours=2.0,
        )
        assert -30.0 - 1e-6 <= res["u_fb_sulfato"] <= 40.0 + 1e-6
    assert twin.pid_turb.integral < 1e6
