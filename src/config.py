"""Shared paths, constants, and thresholds.

Every branch (prediction, spatial, temporal, anomaly) imports from here so
that thresholds and column names never drift apart between modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw" / "water_quality.csv"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET = "River Pollution Index"

# Columns that identify a location/time rather than being direct model
# features (kept out of X).
METADATA_COLUMNS = [
    "siteid",
    "siteengname",
    "countyen",
    "townshipen",
    "basinen",
    "riveren",
    "twd97lon",
    "twd97lat",
    "sampledate",
]

# The shared prediction-output schema every downstream module expects.
# (see project-plan.md, section 5 "Integration Notes")
PREDICTION_SCHEMA = [
    "site_id",
    "lat",
    "lon",
    "timestamp",
    "observed",
    "predicted",
    "residual",
    "model_version",
]

# ---------------------------------------------------------------------------
# Anomaly detection (Member 3)
# ---------------------------------------------------------------------------
ANOMALY_METHOD = "std"       # "mad" needs >1 distinct residual value to be
ANOMALY_MULTIPLIER = 2.0     # meaningful; small demo datasets use "std" instead.

# ---------------------------------------------------------------------------
# Spatial risk classification (Member 2)
# ---------------------------------------------------------------------------
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]
