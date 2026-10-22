# Input Data Guidelines (`data/`)

This directory is intended for input time series datasets used to feed the water treatment plant digital twin.

## 📋 Input Data Schema Specification
To execute the simulator with empirical time series or synthetic datasets, the CSV file must contain the following columns:

| Column | Data Type | Units | Description |
| :--- | :---: | :---: | :--- |
| `DataHora` | Datetime | `DD/MM/YYYY HH:MM` | Timestamp of sampling or simulation step |
| `Mes` | Integer | 1 to 12 | Month of the year (for seasonal stratification) |
| `Sazonalidade` | String | Categorical | Seasonal regime: `"chuvoso"`, `"seca"`, `"transicao_chuva_seca"`, `"transicao_seca_chuva"` |
| `AB_pH` | Float | dimensionless | Raw water pH at the intake |
| `AB_Cor` | Float | Pt-Co (uH) | Raw water apparent color |
| `AB_Turbidez` | Float | NTU | Raw water influent turbidity |
| `AB_Vazao` | Float | L/s or m³/h | Volumetric raw water intake flow rate |
| `DEC_pH` | Float | dimensionless | pH after lamella clarification |
| `DEC_Cor` | Float | Pt-Co | Apparent color in clarified water |
| `DEC_Turbidez` | Float | NTU | Clarified water turbidity |
| `FIL_Cor` | Float | Pt-Co | Filtered water apparent color |
| `FIL_Turbidez` | Float | NTU | Filtered water turbidity |
| `SAI_pH` | Float | dimensionless | Treated effluent pH at contact tank outlet |
| `SAI_Cloro` | Float | mg/L | Free residual chlorine concentration at effluent |
| `SAI_Cor` | Float | Pt-Co | Treated water final apparent color |
| `SAI_Turbidez` | Float | NTU | Treated water final turbidity |

## 🚀 Execution with Custom Datasets
To run the simulation pointing to an external CSV file:

```bash
python run_simulation.py --data "path/to/your_dataset.csv"
```
