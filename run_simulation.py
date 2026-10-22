"""
Master Execution Script for Water Treatment Plant Digital Twin Simulation (WTP Twin Sim).
"""

import argparse
from pathlib import Path

import numpy as np

from data.loader import ETADatasetLoader
from evaluation.scenarios import ScenarioRunner
from models.calibrator import GrayBoxCalibrator
from visualization.plots import PublicationPlotter


def execute_simulation_study(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    sample_size: int | None = None,
    generate_plots: bool = True,
) -> dict[str, object]:
    """
    Executes the full digital twin simulation pipeline:
    1. Ingestion of input water quality time series (CSV)
    2. Non-linear calibration of the Gray-Box mechanistic model
    3. Scenario execution (Historical Baseline, Closed-Loop Digital Twin, Extreme Shocks)
    4. Compliance assessment against potability standards and OPEX chemical balance
    5. High-resolution figure rendering (300 DPI) and summary table export
    """
    current_dir = Path(__file__).parent

    # Reproducibility seed
    np.random.seed(42)

    # Default data path
    if data_path is None:
        local_data = current_dir / "data" / "ouro_wide.csv"
        if local_data.exists():
            data_path = local_data
        else:
            data_path = (
                current_dir.parents[1]
                / "02_Series_Historicas_e_Estatistica"
                / "Dados"
                / "3_Ouro"
                / "ouro_wide.csv"
            )
    else:
        data_path = Path(data_path)

    # Default output directory
    if output_dir is None:
        output_dir = current_dir / "resultados_simulacao"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== STARTING WATER TREATMENT PLANT DIGITAL TWIN SIMULATION ===")
    print(f"Data Source: {data_path}")
    print(f"Output Directory: {output_dir}\n")

    # 1. Loading
    loader = ETADatasetLoader(data_path)
    df_raw = loader.load_clean_data()
    if sample_size is not None and sample_size < len(df_raw):
        df_raw = df_raw.iloc[:sample_size].copy()

    print(f"-> {len(df_raw)} temporal records successfully loaded.")

    # 2. Calibration
    print("-> Calibrating Gray-Box model kinetic parameters...")
    calibrator = GrayBoxCalibrator()
    train_size = int(len(df_raw) * 0.70)
    calib_params = calibrator.calibrate(df_raw.iloc[:train_size])
    print(f"   k0_sed = {calib_params['k0_sed']:.3f} h^-1")
    print(f"   lambda_fil = {calib_params['lambda_fil']:.3f}")
    print(f"   k_decay_cloro = {calib_params['k_decay_cloro']:.4f} h^-1\n")

    # 3. Scenario Execution
    print("-> Executing simulation scenarios (Baseline, Digital Twin, and Shocks)...")
    runner = ScenarioRunner(plant_params=calib_params)
    results = runner.run_scenarios(df_raw)

    # 4. Summary Export
    df_summary = results["summary_table"]
    csv_summary_path = output_dir / "tabela_resumo_cenarios.csv"
    df_summary.to_csv(csv_summary_path, index=False, encoding="utf-8-sig")
    print(f"-> Summary table exported to: {csv_summary_path}")

    # 5. Scientific Plots Generation (300 DPI)
    if generate_plots:
        print("-> Generating high-resolution publication figures (300 DPI)...")
        plotter = PublicationPlotter(output_dir)

        p1 = plotter.plot_time_series_comparison(
            df_real=df_raw,
            df_baseline=results["baseline_historical"],
            df_twin=results["digital_twin_routine"],
        )
        print(f"   [OK] {p1.name}")

        p2 = plotter.plot_compliance_and_scatter(
            df_real=df_raw,
            df_twin=results["digital_twin_routine"],
        )
        print(f"   [OK] {p2.name}")

        p3 = plotter.plot_shock_response(
            df_storm=results["extreme_storm_tide"],
        )
        print(f"   [OK] {p3.name}")

    print("\n=== SIMULATION COMPLETED SUCCESSFULLY! ===")
    return results


def main():
    parser = argparse.ArgumentParser(description="Water Treatment Plant Digital Twin Simulator")
    parser.add_argument("--data", type=str, default=None, help="Path to input water quality time series CSV")
    parser.add_argument("--output", type=str, default=None, help="Output directory to save simulation figures and tables")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for fast test execution")
    args = parser.parse_args()

    execute_simulation_study(
        data_path=args.data,
        output_dir=args.output,
        sample_size=args.sample,
        generate_plots=True,
    )


if __name__ == "__main__":
    main()
