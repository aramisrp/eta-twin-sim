import pandas as pd

from run_simulation import execute_simulation_study


def test_full_pipeline_execution(tmp_path):
    output_dir = tmp_path / "output_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = execute_simulation_study(
        output_dir=output_dir,
        sample_size=150, # Modo rápido para teste
        generate_plots=True,
    )
    
    assert "summary_table" in results
    assert (output_dir / "tabela_resumo_cenarios.csv").exists()
    assert (output_dir / "figura_1_comparacao_temporal_cenarios.png").exists()
    assert (output_dir / "figura_2_dispersao_e_portaria_888.png").exists()
    assert (output_dir / "figura_3_resposta_choque_estuarino.png").exists()
    
    df_resumo = pd.read_csv(output_dir / "tabela_resumo_cenarios.csv")
    assert len(df_resumo) == 3
