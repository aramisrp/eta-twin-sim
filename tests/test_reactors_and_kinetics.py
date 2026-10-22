from pathlib import Path

from data.loader import ETADatasetLoader
from models.calibrator import GrayBoxCalibrator
from models.kinetics import ChlorineDecayKinetics, CoagulationKinetics, PHBalanceModel
from models.reactors import (
    ChlorineContactReservoir,
    LamellaDecanterCSTR,
    RapidSandFilter,
)


def test_coagulation_kinetics():
    kin = CoagulationKinetics(k0=1.8, d_sat=25.0, gamma=0.8)
    
    # Taxa de sedimentação aumenta com a dosagem de coagulante
    k_low = kin.calculate_ksed(dosage_sulfato=10.0, flow_lps=25.0)
    k_high = kin.calculate_ksed(dosage_sulfato=40.0, flow_lps=25.0)
    assert k_high > k_low
    
    # Efeito da vazão na taxa
    k_high_flow = kin.calculate_ksed(dosage_sulfato=40.0, flow_lps=35.0)
    assert k_high_flow < k_high

def test_chlorine_decay_and_ph_balance():
    cl_kin = ChlorineDecayKinetics(k_base=0.08)
    decay_rate = cl_kin.calculate_kdecay(turbidity=0.4, ph=6.8)
    assert decay_rate > 0.0
    
    ph_model = PHBalanceModel(alpha_sulfato=0.015, beta_cloro=0.025)
    ph_out = ph_model.calculate_ph_out(ph_raw=6.8, dos_sulfato=30.0, dos_cloro=2.0)
    assert 5.0 < ph_out < 9.0

def test_reactors_mass_balance_and_stability():
    # 2 Decantadores em paralelo (35m³ cada = 70m³ total)
    decanter = LamellaDecanterCSTR(unit_volume_m3=35.0, num_parallel_units=2)
    state = decanter.step(
        turb_in=60.0, color_in=140.0, flow_lps=25.0,
        dos_sulfato=35.0, dt_hours=2.0
    )
    assert 0.0 < state["dec_turb"] < 60.0
    assert 0.0 < state["dec_cor"] < 140.0
    
    # Filtro de areia
    filter_unit = RapidSandFilter(lambda_fil=1.6, bed_height_m=1.2)
    fil_state = filter_unit.step(turb_in=state["dec_turb"], color_in=state["dec_cor"], flow_lps=25.0)
    assert 0.0 < fil_state["fil_turb"] < state["dec_turb"]
    assert 0.0 < fil_state["fil_cor"] < state["dec_cor"]
    
    # Reservatório e contato de cloro (800m³)
    contact = ChlorineContactReservoir(total_volume_m3=800.0)
    res_state = contact.step(
        dos_cloro=2.0, fil_turb=fil_state["fil_turb"], fil_cor=fil_state["fil_cor"],
        flow_lps=25.0, dt_hours=2.0
    )
    assert 0.0 < res_state["sai_cloro"] <= 2.0

def test_full_state_space_and_calibrator():
    artigo_dir = Path(__file__).parents[3]
    data_path = artigo_dir / "02_Series_Historicas_e_Estatistica" / "Dados" / "3_Ouro" / "ouro_wide.csv"
    loader = ETADatasetLoader(data_path)
    df = loader.load_clean_data()
    
    calibrator = GrayBoxCalibrator()
    params = calibrator.calibrate(df.iloc[:200]) # Calibração rápida em subset
    
    assert "k0_sed" in params
    assert "lambda_fil" in params
    assert "k_decay_cloro" in params
    assert params["k0_sed"] > 0.0
