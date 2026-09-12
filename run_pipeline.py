"""End-to-end pipeline: prediction -> explainability -> spatial -> temporal -> anomaly.

Run with:  python run_pipeline.py

This wires together the three team members' branches into one working demo:
  Member 1 (prediction + explainability) -> Member 2 (spatial + temporal)
                                          -> Member 3 (anomaly + uncertainty)

The bundled dataset (data/raw/water_quality.csv) is a tiny hackathon sample
(only 2 site/date readings have complete, non-placeholder values), so model
accuracy numbers here are not meaningful - the point of this script is to
prove the whole pipeline runs cleanly end to end on real data of this shape.
Drop a bigger CSV in data/raw/water_quality.csv and it will scale up
automatically (10-fold CV, real holdout split, etc.).
"""

from __future__ import annotations

import json
import warnings

import pandas as pd

from src.config import (
    ANOMALY_METHOD,
    ANOMALY_MULTIPLIER,
    DATA_RAW,
    MODELS_DIR,
    OUTPUTS_DIR,
    TARGET,
)
from src.data.preprocessing import build_features_and_target, load_raw, pivot_to_model_ready
from src.data.split import safe_train_test_split
from src.models.train import (
    cross_validate_models,
    evaluate_holdout,
    select_best_model_name,
    train_final_model,
)
from src.models.predict import build_prediction_output
from src.explainability.feature_ranking import rank_features
from src.explainability.shap_analysis import compute_shap_values
from src.spatial.hotspot_classifier import classify_risk
from src.classification.smote_classifier import run_smote_classification
from src.spatial.mapping import make_hotspot_map
from src.temporal.time_grouping import add_time_groups
from src.temporal.trend_tracking import classify_hotspot_evolution
from src.anomaly.residuals import error_metrics
from src.anomaly.threshold import calculate_threshold
from src.anomaly.flagging import build_event_report, anomaly_stability

import joblib

warnings.filterwarnings("ignore")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = ["# Pipeline Run Report\n"]

    # ------------------------------------------------------------------
    # Member 1: Preprocessing + Prediction
    # ------------------------------------------------------------------
    section("1. LOAD + PREPROCESS")
    raw = load_raw(DATA_RAW)
    quality_df = pivot_to_model_ready(raw)
    X, y, feature_columns, labeled_df = build_features_and_target(quality_df)
    print(f"Raw readings: {raw.shape}  ->  pivoted site/date rows: {quality_df.shape}")
    print(f"Rows with a usable '{TARGET}' target: {len(y)}")
    report_lines.append(
        f"- Raw readings: {raw.shape[0]} rows -> pivoted to {quality_df.shape[0]} "
        f"site/date rows -> {len(y)} rows with a usable target.\n"
    )

    # ------------------------------------------------------------------
    # Evaluator-required classification branch: SMOTE
    # ------------------------------------------------------------------
    section("2. CLASSIFICATION: SMOTE + RISK PREDICTION")
    report_lines.append(
    "- Classification procedure: Data preprocessing -> Data leakage checks -> "
    "Train/test split -> 10-fold cross-validation -> SMOTE on training data only -> "
    "Initial model training -> Overfitting/underfitting detection -> "
    "Correction and retraining -> Final test evaluation.\n"
    )
    classification_result = run_smote_classification(X, y)

    classification_metrics = classification_result["metrics"]

    report_lines.append(
        f"- SMOTE classification final accuracy: "
        f"{classification_metrics['final_test_accuracy'] * 100:.2f}%\n"
    )
    report_lines.append(
        f"- SMOTE classification precision: "
        f"{classification_metrics['final_test_precision']:.4f}\n"
    )
    report_lines.append(
        f"- SMOTE classification recall: "
        f"{classification_metrics['final_test_recall']:.4f}\n"
    )
    report_lines.append(
        f"- SMOTE classification F1: "
        f"{classification_metrics['final_test_f1']:.4f}\n"
    )

    section("3. MODEL COMPARISON (CV)")
    cv_results_df, cv_note = cross_validate_models(X, y)
    print(cv_note)
    print(cv_results_df.to_string(index=False))
    report_lines.append(f"- {cv_note}\n\n{cv_results_df.to_markdown(index=False)}\n")

    best_model_name = select_best_model_name(cv_results_df)
    print(f"\nSelected model: {best_model_name}")
    report_lines.append(f"\n- Selected model: **{best_model_name}**\n")

    section("3. TRAIN + HOLDOUT EVALUATION")
    X_train, X_test, y_train, y_test, split_note = safe_train_test_split(X, y)
    print(split_note)
    final_model = train_final_model(X_train, y_train, best_model_name)
    holdout_metrics, _ = evaluate_holdout(final_model, X_test, y_test)
    print("Holdout metrics:", holdout_metrics)
    report_lines.append(f"- {split_note}\n- Holdout metrics: `{holdout_metrics}`\n")

    # Refit on ALL labeled rows for the model we actually ship/predict with.
    final_model = train_final_model(X, y, best_model_name)
    joblib.dump(final_model, MODELS_DIR / "final_model.pkl")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump({"model": best_model_name, "holdout": holdout_metrics}, f, indent=2)
    print("Saved: models/final_model.pkl, feature_columns.json, model_metrics.json")

    # ------------------------------------------------------------------
    # Member 1: Explainability
    # ------------------------------------------------------------------
    section("4. EXPLAINABILITY")
    importance_df = rank_features(final_model, feature_columns)
    print("Top features (model importance):")
    print(importance_df.head(10).to_string(index=False))
    importance_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

    shap_values, X_processed, shap_note = compute_shap_values(final_model, X)
    print(shap_note)
    report_lines.append(f"- Explainability: {shap_note}\n")

    # ------------------------------------------------------------------
    # Standardized prediction output (the contract every other branch uses)
    # ------------------------------------------------------------------
    section("5. STANDARDIZED PREDICTION OUTPUT")
    predicted = final_model.predict(X)
    prediction_output = build_prediction_output(
        quality_df, y.index, X, y, predicted, model_version=best_model_name
    )
    prediction_output.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)
    print(prediction_output[["site_id", "lat", "lon", "timestamp", "observed",
                              "predicted", "residual", "model_version"]].to_string(index=False))
    print("Saved: outputs/predictions.csv")

    # ------------------------------------------------------------------
    # Member 2: Spatial risk classification + map
    # ------------------------------------------------------------------
    section("6. SPATIAL: RISK CLASSIFICATION + MAP")
    spatial_input = prediction_output.rename(
        columns={"site_id": "siteid", "lat": "twd97lat", "lon": "twd97lon"}
    )
    risk_df = classify_risk(spatial_input)
    print(risk_df[["siteid", "risk_score", "risk_level", "risk_basis"]].to_string(index=False))
    make_hotspot_map(risk_df, output_html=str(OUTPUTS_DIR / "hotspot_map.html"))
    print("Saved: outputs/hotspot_map.html")

    # ------------------------------------------------------------------
    # Member 2: Temporal trend tracking
    # ------------------------------------------------------------------
    section("7. TEMPORAL: TIME GROUPING + HOTSPOT EVOLUTION")
    temporal_df = risk_df.rename(columns={"timestamp": "sampledate"})
    if "sampledate" not in temporal_df.columns:
        temporal_df["sampledate"] = prediction_output["timestamp"]
    temporal_df = add_time_groups(temporal_df)
    print(temporal_df[["siteid", "sampledate", "year", "season"]].to_string(index=False))

    evolution_input = temporal_df.rename(columns={"siteid": "siteengname"}) \
        if "siteengname" not in temporal_df.columns else temporal_df
    evolution_df = classify_hotspot_evolution(evolution_input)
    evolution_df.to_csv(OUTPUTS_DIR / "hotspot_evolution.csv", index=False)
    print(evolution_df.to_string(index=False))
    print("Saved: outputs/hotspot_evolution.csv")

    # ------------------------------------------------------------------
    # Member 3: Anomaly detection + uncertainty
    # ------------------------------------------------------------------
    section("8. ANOMALY DETECTION + UNCERTAINTY")
    residuals = prediction_output["residual"].tolist()
    threshold = calculate_threshold(residuals, method=ANOMALY_METHOD, multiplier=ANOMALY_MULTIPLIER)
    print(f"Anomaly threshold ({ANOMALY_METHOD}, x{ANOMALY_MULTIPLIER}): {threshold:.4f}")

    metrics = error_metrics(residuals)
    print("Residual error metrics:", metrics)

    event_records = prediction_output.assign(year=temporal_df["year"].values).to_dict("records")
    event_report = build_event_report(event_records, threshold=threshold)
    event_report_df = pd.DataFrame(event_report)
    event_report_df.to_csv(OUTPUTS_DIR / "anomaly_report.csv", index=False)
    print(event_report_df[["site_id", "observed", "predicted", "residual",
                            "is_anomaly", "direction"]].to_string(index=False))
    print("Saved: outputs/anomaly_report.csv")

    stability = anomaly_stability(event_report)
    stability_df = pd.DataFrame(stability)
    stability_df.to_csv(OUTPUTS_DIR / "anomaly_stability.csv", index=False)
    print(stability_df.to_string(index=False))
    print("Saved: outputs/anomaly_stability.csv")

    report_lines.append(f"- Anomaly threshold ({ANOMALY_METHOD}, x{ANOMALY_MULTIPLIER}): {threshold:.4f}\n")
    report_lines.append(f"- Residual error metrics: `{metrics}`\n")
    report_lines.append(
        f"- Flagged anomalies: {int(event_report_df['is_anomaly'].sum())} / {len(event_report_df)}\n"
    )

    with open(OUTPUTS_DIR / "final_report.md", "w") as f:
        f.write("\n".join(report_lines))
    print("\nSaved: outputs/final_report.md")

    section("PIPELINE COMPLETE")
    print("All stages ran successfully. See outputs/ for predictions, map, and reports.")


if __name__ == "__main__":
    main()
