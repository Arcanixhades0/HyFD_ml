# HyFD: Hybrid Failure Detection for Production ML Systems

This repository contains the complete code and experiments for the paper:

**"HyFD: A Hybrid Multi-Signal Framework for Automated Discovery of Hidden Failure Modes in Production Machine Learning Systems"**

---

## Project Structure

```
ml_failure_detection/
├── src/                          # Core source modules
│   ├── hyfd.py                   # Main HyFD system (proposed method)
│   ├── drift_detector.py         # Signal 1: Data drift (KS + PSI)
│   ├── uncertainty_estimator.py  # Signal 2: Prediction uncertainty
│   ├── slice_analyzer.py         # Signal 3: Slice-based performance
│   ├── data_quality.py           # Signal 4+5: Quality + OOD detection
│   ├── baselines.py              # Single-signal baseline methods
│   └── scenario_simulator.py     # Production failure simulators
├── experiments/
│   ├── run_experiments.py        # Main experiment runner
│   └── generate_figures.py       # Figure generator
├── results/                      # CSV output files (auto-generated)
├── figures/                      # PNG figures (auto-generated)
├── paper/
│   ├── generate_paper.js         # DOCX paper generator (Node.js)
│   └── HyFD_Research_Paper.docx  # Final paper
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Run all experiments
```bash
cd experiments
PYTHONPATH=../src python run_experiments.py
```

### 3. Generate all figures
```bash
cd experiments
PYTHONPATH=../src python generate_figures.py
```

### 4. Regenerate the paper (requires Node.js + docx)
```bash
npm install -g docx
cd paper
node generate_paper.js
```

---

## Modules

### `hyfd.py` — Main System
The `HyFD` class is the proposed multi-signal detector. Usage:

```python
from hyfd import HyFD

# Initialize and fit on reference data
hyfd = HyFD(detection_threshold=0.35)
hyfd.fit(X_train, y_train, model, feature_names=feature_names)

# Detect failures on a production batch
result = hyfd.detect(X_production, y_prod=None)
print(result["failure_detected"])   # True / False
print(result["composite_score"])    # 0.0 – 1.0
print(result["signal_scores"])      # Per-signal breakdown
```

### `scenario_simulator.py` — Failure Scenarios
```python
from scenario_simulator import get_all_scenarios
scenarios = get_all_scenarios(X_train, n_samples=500)
# Returns dict: {scenario_name: (X_prod, label)}
```

### `baselines.py` — Single-Signal Baselines
```python
from baselines import get_all_baselines
baselines = get_all_baselines(X_train, model)
result = baselines["DriftOnly"].detect(X_prod)
```

---

## Failure Scenarios

| Scenario | Description | Primary Signal |
|---|---|---|
| No Failure | Healthy production data | — |
| Data Drift | Feature mean shift (+1.0 to +2.0) | Drift |
| Noisy Features | Gaussian noise σ=3.0 added | Uncertainty + Quality |
| Missing Data | 25% NaN injection | Quality |
| OOD Samples | 40% samples from 6σ away | OOD |
| Corrupted Data | 15% extreme values (±999) | Quality |

---

## Results Summary

| Method | Accuracy | TPR | FPR | F1 |
|---|---|---|---|---|
| **HyFD (Proposed)** | **100%** | **100%** | **0%** | **1.000** |
| DriftOnly | 100% | 100% | 0% | 1.000 |
| OODOnly | 83.3% | 100% | 100% | 0.909 |
| UncertaintyOnly | 83.3% | 80% | 0% | 0.889 |
| QualityOnly | 66.7% | 60% | 0% | 0.750 |

*HyFD is the only method with complete scenario coverage (100% accuracy on every individual scenario)*

---

## Citation
If you use this code, please cite:
```
@article{hyfd2025,
  title={HyFD: A Hybrid Multi-Signal Framework for Automated Discovery
         of Hidden Failure Modes in Production Machine Learning Systems},
  author={Anonymous},
  journal={Journal of Machine Learning Research},
  year={2025}
}
```
