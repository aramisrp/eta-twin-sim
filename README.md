# Water Treatment Plant Digital Twin Simulator (`eta_twin_sim`)

A computational Python framework for physicochemical **Gray-Box modeling** and **Digital Twin** closed-loop simulation (incorporating stoichiometric *Feedforward* + Discrete *Feedback PI with Clamping Anti-Windup*) for conventional and compact water treatment plants (WTP).

---

## 🚀 Package Architecture

- **`config/`**: Geometric plant parameters (parallel inclined-plate lamella clarifiers, rapid granular dual-media filters, contact/storage reservoir) and physical actuator operating limits.
- **`data/`**: Ingestion pipelines (`loader.py`) and seasonal stratification tools (`preprocessor.py`). See [`data/README.md`](data/README.md) for the required CSV input schema.
- **`models/`**:
  - `reactors.py`: State-space Continuous Stirred-Tank Reactor (CSTR) cascades coupled with Iwasaki's deep-bed granular filtration model.
  - `kinetics.py`: Coagulant-assisted sedimentation laws and free chlorine decay kinetics (bulk liquid reaction, wall consumption, and organic demand).
  - `estuary_forcing.py`: Upstream hydrodynamic boundary conditions (semidiurnal $M_2/S_2$ tidal cycles, rainfall-runoff convolution, and pump affinity laws).
  - `calibrator.py`: Non-linear kinetic parameter estimation via numerical optimization (`scipy.optimize`).
- **`control/`**: Feedforward predictive dosing and PI feedback loops with integrator desaturation (*clamping anti-windup*).
- **`evaluation/`**: Water quality compliance metrics ($R^2$, $NSE$, $RMSE$, $MAPE$), potability standards tracking, and chemical consumption balance (OPEX).
- **`visualization/`**: Publication-grade scientific figure generator (300 DPI) for technical reports and journal articles.
- **`tests/`**: Comprehensive automated test suite executed with `pytest`.

---

## 💻 Getting Started

### 1. Installation
Python 3.10 or higher is recommended:

```bash
git clone https://github.com/aramisrp/eta-twin-sim.git
cd eta-twin-sim
pip install -e .
```

### 2. Running the Simulation
To execute the end-to-end digital twin pipeline with your water quality time series dataset:

```bash
python run_simulation.py --data "path/to/your_dataset.csv"
```

### 3. Running Automated Tests (Pytest)
```bash
pytest tests/ -v
```

---

## 📄 License

Distributed under an open-source license for academic, scientific, and engineering research in water sanitation and process control automation.
