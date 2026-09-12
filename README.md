# AI-Based Urban Water-Quality Monitoring Framework

An integrated AI-based framework for urban water-quality monitoring that combines:

- Water-quality prediction
- Risk-level classification using SMOTE
- Model explainability
- Spatial hotspot identification
- Temporal hotspot evolution
- Anomaly detection
- Anomaly stability analysis

The project integrates the prediction/explainability, spatial/temporal, and anomaly/uncertainty components into a single end-to-end Python pipeline.

---

## Project Overview

The system processes water-quality measurements and generates a River Pollution Index (RPI) for each site/date observation.

The framework provides two machine-learning tasks:

1. **Regression** - predicts the continuous River Pollution Index.
2. **Classification** - classifies observations into LOW, MEDIUM, or HIGH pollution-risk levels.

The classification branch follows the required machine-learning workflow:

**Data preprocessing -> Data leakage checks -> Train/test split -> 10-fold cross-validation -> SMOTE on training data only -> Initial model training -> Overfitting/underfitting detection -> Model correction and retraining -> Final test evaluation**

---
## Run Command
python run_pipeline.py

## Dataset

The pipeline works with the water-quality CSV dataset:

```text
data/raw/water_quality.csv


