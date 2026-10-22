"""
Scientific Figure Rendering Module in High Resolution (300 DPI) for Publication.
Generates comparative figures and charts for technical reports and scientific papers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Global publication style settings
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


class PublicationPlotter:
    """
    Renders and exports 300 DPI publication figures.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_time_series_comparison(
        self,
        df_real: pd.DataFrame,
        df_baseline: pd.DataFrame,
        df_twin: pd.DataFrame,
        filename: str = "figura_1_comparacao_temporal_cenarios.png",
    ) -> Path:
        """
        Figure 1: Comparative time series of Turbidity and Free Chlorine at WTP effluent.
        """
        _, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        n = min(len(df_real), len(df_twin), 240)  # 20-day window (2-hour step)
        time_steps = np.arange(n) * 2.0  # Hours

        # 1. Raw Water vs Effluent Turbidity
        axes[0].plot(
            time_steps, df_real["AB_Turbidez"].iloc[:n], color="#7f7f7f", alpha=0.6, label="Raw Water Influent"
        )
        axes[0].plot(
            time_steps, df_baseline["sai_turb"].iloc[:n], color="#d62728", linestyle="--", label="Manual Baseline Effluent"
        )
        axes[0].plot(
            time_steps, df_twin["sai_turb"].iloc[:n], color="#1f77b4", linewidth=1.8, label="Closed-Loop Digital Twin"
        )
        axes[0].axhline(0.5, color="#2ca02c", linestyle=":", linewidth=1.5, label="Target Standard (<= 0.5 NTU)")
        axes[0].axhline(5.0, color="#d62728", linestyle=":", linewidth=1.2, label="Maximum Limit (5.0 NTU)")
        axes[0].set_ylabel("Turbidity (NTU)")
        axes[0].set_title("A. Clarification Dynamics: Influent vs. Treated Effluent Turbidity")
        axes[0].legend(loc="upper right", framealpha=0.9, ncol=2)

        # 2. Free Residual Chlorine
        axes[1].plot(
            time_steps, df_real["SAI_Cloro"].iloc[:n], color="#7f7f7f", alpha=0.7, label="Historical Measured Chlorine"
        )
        axes[1].plot(
            time_steps, df_twin["sai_cloro"].iloc[:n], color="#1f77b4", linewidth=1.8, label="Controlled Digital Twin Chlorine"
        )
        axes[1].axhspan(0.2, 2.0, color="#2ca02c", alpha=0.15, label="Standard Range (0.2 - 2.0 mg/L)")
        axes[1].axhline(1.5, color="#1f77b4", linestyle="--", linewidth=1.2, label="Digital Twin Setpoint (1.5 mg/L)")
        axes[1].set_ylabel("Free Chlorine (mg/L)")
        axes[1].set_title("B. Disinfection Control: Free Residual Chlorine Stability")
        axes[1].legend(loc="upper right", framealpha=0.9, ncol=2)

        # 3. Actuated Coagulant and Chlorine Dosages
        ax3_twin = axes[2].twinx()
        l1 = axes[2].plot(
            time_steps, df_twin["dos_sulfato_applied"].iloc[:n], color="#ff7f0e", label="Coagulant Dosing Rate (mg/L)"
        )
        l2 = ax3_twin.plot(
            time_steps, df_twin["dos_cloro_applied"].iloc[:n], color="#9467bd", linestyle="-.", label="Sodium Hypochlorite Rate (mg/L)"
        )
        axes[2].set_ylabel("Coagulant (mg/L)", color="#ff7f0e")
        ax3_twin.set_ylabel("Chlorine (mg/L)", color="#9467bd")
        axes[2].set_xlabel("Continuous Operating Time (hours)")
        axes[2].set_title("C. Peristaltic Dosing Pump Actuation (Feedforward + Feedback)")

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        axes[2].legend(lines, labels, loc="upper right", framealpha=0.9)

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_compliance_and_scatter(
        self,
        df_real: pd.DataFrame,
        df_twin: pd.DataFrame,
        filename: str = "figura_2_dispersao_e_portaria_888.png",
    ) -> Path:
        """
        Figure 2: Scatter diagram and compliance distributions against standards.
        """
        _, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # 1. Raw vs Treated Turbidity Scatter
        n = min(len(df_real), len(df_twin))
        sns.scatterplot(
            x=df_real["AB_Turbidez"].iloc[:n],
            y=df_real["SAI_Turbidez"].iloc[:n],
            ax=axes[0],
            color="#d62728",
            alpha=0.5,
            label="Manual Historical Baseline",
        )
        sns.scatterplot(
            x=df_real["AB_Turbidez"].iloc[:n],
            y=df_twin["sai_turb"].iloc[:n],
            ax=axes[0],
            color="#1f77b4",
            alpha=0.5,
            label="Automated Digital Twin",
        )
        axes[0].axhline(0.5, color="#2ca02c", linestyle="--", linewidth=1.5, label="Target Standard (<= 0.5 NTU)")
        axes[0].set_xlabel("Raw Water Influent Turbidity (NTU)")
        axes[0].set_ylabel("Treated Effluent Turbidity (NTU)")
        axes[0].set_title("A. Turbidity Removal Efficiency Under Load Variations")
        axes[0].legend(loc="upper left")

        # 2. Residual Chlorine Distribution by Season
        plot_data = []
        for i in range(n):
            season = df_real["Sazonalidade"].iloc[i]
            plot_data.append({"Season": season, "Mode": "Historical", "Chlorine": df_real["SAI_Cloro"].iloc[i]})
            plot_data.append({"Season": season, "Mode": "Digital Twin", "Chlorine": df_twin["sai_cloro"].iloc[i]})

        df_box = pd.DataFrame(plot_data)
        sns.boxplot(
            x="Season",
            y="Chlorine",
            hue="Mode",
            data=df_box,
            ax=axes[1],
            palette=["#ff9896", "#aec7e8"],
        )
        axes[1].axhspan(0.2, 2.0, color="#2ca02c", alpha=0.15, label="Standard Range (0.2 - 2.0 mg/L)")
        axes[1].set_ylabel("Free Residual Chlorine (mg/L)")
        axes[1].set_title("B. Disinfection Stability Across Seasonal Regimes")
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_shock_response(
        self,
        df_storm: pd.DataFrame,
        filename: str = "figura_3_resposta_choque_estuarino.png",
    ) -> Path:
        """
        Figure 3: Transient response under severe storm and tidal shock.
        """
        _, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
        t = df_storm["time_hours"].values

        # 1. Boundary Condition: Tidal Level and Shock Turbidity
        ax1_twin = axes[0].twinx()
        l1 = axes[0].plot(t, df_storm["tide_level_m"], color="#17becf", linestyle="--", label="Tidal Level (m)")
        l2 = ax1_twin.plot(t, df_storm["ab_turb_injected"], color="#8c564b", linewidth=1.8, label="Raw Turbidity with Storm Shock (NTU)")
        axes[0].set_ylabel("Tidal Level (m)", color="#17becf")
        ax1_twin.set_ylabel("Raw Turbidity (NTU)", color="#8c564b")
        axes[0].set_title("A. Upstream Environmental Forcings: Tidal Oscillation and Rainfall Peak")
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        axes[0].legend(lines, labels, loc="upper right")

        # 2. Control Response
        axes[1].plot(t, df_storm["u_ff_sulfato"], color="#ff7f0e", linestyle=":", label="Feedforward Predictive Action")
        axes[1].plot(t, df_storm["dos_sulfato_applied"], color="#d62728", linewidth=1.8, label="Total Applied Coagulant Dosage (mg/L)")
        axes[1].set_ylabel("Coagulant Dosage (mg/L)")
        axes[1].set_title("B. Digital Twin Composite Control Action During Transient Shock")
        axes[1].legend(loc="upper right")

        # 3. Effluent Turbidity and Chlorine
        ax3_twin = axes[2].twinx()
        l3 = axes[2].plot(t, df_storm["sai_turb"], color="#1f77b4", linewidth=1.8, label="Effluent Turbidity (NTU)")
        l4 = ax3_twin.plot(t, df_storm["sai_cloro"], color="#2ca02c", linewidth=1.8, label="Effluent Chlorine (mg/L)")
        axes[2].axhline(0.5, color="#1f77b4", linestyle="--", alpha=0.7, label="Target 0.5 NTU")
        ax3_twin.axhline(1.5, color="#2ca02c", linestyle="--", alpha=0.7, label="Chlorine Setpoint 1.5 mg/L")
        axes[2].set_ylabel("Effluent Turbidity (NTU)", color="#1f77b4")
        ax3_twin.set_ylabel("Effluent Chlorine (mg/L)", color="#2ca02c")
        axes[2].set_xlabel("Elapsed Time (hours)")
        axes[2].set_title("C. Treated Water Quality Response During Extreme Transient Shock")

        lines2 = l3 + l4
        labels2 = [l.get_label() for l in lines2]
        axes[2].legend(lines2, labels2, loc="upper right")

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path)
        plt.close()
        return out_path
