"""
SMOTE-based classification branch for evaluator requirements.

This branch classifies River Pollution Index (RPI) into
LOW, MEDIUM, and HIGH risk classes.

SMOTE is applied only to training data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import RANDOM_STATE, OUTPUTS_DIR


def create_risk_labels(y, q1, q2):
    """Convert continuous RPI values into LOW/MEDIUM/HIGH classes."""
    return pd.Series(
        np.select(
            [y <= q1, y <= q2],
            ["LOW", "MEDIUM"],
            default="HIGH",
        ),
        index=y.index,
        name="risk_level",
    )


def check_data_leakage(X, y):
    """Run basic target/duplicate/suspicious-feature leakage checks."""
    checks = {}

    checks["target_in_features"] = "River Pollution Index" in X.columns

    duplicate_rows = X.duplicated().sum()
    checks["duplicate_feature_rows"] = int(duplicate_rows)

    suspicious = [
        col
        for col in X.columns
        if "river pollution index" in str(col).lower()
        or str(col).lower() == "rpi"
    ]
    checks["suspicious_features"] = suspicious

    return checks


def build_initial_model():
    """Initial classifier."""
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    class_weight=None,
                ),
            ),
        ]
    )


def build_corrected_model():
    """
    Corrected model after checking for overfitting.

    Stronger regularization is used to reduce tree complexity.
    """
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    class_weight=None,
                ),
            ),
        ]
    )


def run_smote_classification(X, y):
    """
    Execute the evaluator-required classification sequence.

    Returns a dictionary containing metrics and trained model.
    """

    print("\n" + "=" * 70)
    print("SMOTE CLASSIFICATION BRANCH")
    print("=" * 70)

    # --------------------------------------------------------------
    # 1. Create train/test split FIRST
    # --------------------------------------------------------------
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train_rpi, y_test_rpi = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print("\n1. TRAIN / TEST SPLIT")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows:     {len(X_test)}")

    # --------------------------------------------------------------
    # 2. Risk thresholds calculated from TRAINING data only
    # --------------------------------------------------------------
    q1, q2 = y_train_rpi.quantile([0.33, 0.67])

    y_train = create_risk_labels(y_train_rpi, q1, q2)
    y_test = create_risk_labels(y_test_rpi, q1, q2)

    print("\nRisk thresholds from TRAINING data:")
    print(f"LOW / MEDIUM boundary:  {q1:.4f}")
    print(f"MEDIUM / HIGH boundary: {q2:.4f}")

    print("\nTraining class distribution BEFORE SMOTE:")
    print(y_train.value_counts().to_string())

    # --------------------------------------------------------------
    # 3. Data leakage checks
    # --------------------------------------------------------------
    print("\n2. DATA LEAKAGE CHECKS")

    leakage = check_data_leakage(X_train, y_train)

    print(
        "Target leakage:",
        "DETECTED" if leakage["target_in_features"] else "NOT DETECTED",
    )

    print(
        "Duplicate feature rows:",
        leakage["duplicate_feature_rows"],
    )

    print(
        "Suspicious features:",
        leakage["suspicious_features"]
        if leakage["suspicious_features"]
        else "None",
    )

    # --------------------------------------------------------------
    # 4. 10-fold cross-validation
    #    SMOTE happens inside each training fold.
    # --------------------------------------------------------------
    print("\n3. 10-FOLD CROSS-VALIDATION")

    initial_model = build_initial_model()

    cv_results = cross_validate(
        initial_model,
        X_train,
        y_train,
        cv=10,
        scoring=[
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ],
        return_train_score=True,
    )

    cv_metrics = {
        "accuracy": float(cv_results["test_accuracy"].mean()),
        "precision": float(cv_results["test_precision_macro"].mean()),
        "recall": float(cv_results["test_recall_macro"].mean()),
        "f1": float(cv_results["test_f1_macro"].mean()),
    }

    print(
        f"CV Accuracy:  {cv_metrics['accuracy']:.4f} "
        f"({cv_metrics['accuracy'] * 100:.2f}%)"
    )
    print(f"CV Precision: {cv_metrics['precision']:.4f}")
    print(f"CV Recall:    {cv_metrics['recall']:.4f}")
    print(f"CV F1:        {cv_metrics['f1']:.4f}")

    # --------------------------------------------------------------
    # 5. SMOTE + initial model training
    # --------------------------------------------------------------
    print("\n4. SMOTE + INITIAL MODEL TRAINING")

    # Fit the pipeline on TRAINING DATA ONLY.
    initial_model.fit(X_train, y_train)

    # Show resulting training distribution after SMOTE.
    X_train_imputed = SimpleImputer(strategy="median").fit_transform(X_train)
    X_train_smote, y_train_smote = SMOTE(
        random_state=RANDOM_STATE
    ).fit_resample(X_train_imputed, y_train)

    print("Training class distribution AFTER SMOTE:")
    print(pd.Series(y_train_smote).value_counts().to_string())

    train_pred_initial = initial_model.predict(X_train)
    test_pred_initial = initial_model.predict(X_test)

    train_accuracy_initial = accuracy_score(y_train, train_pred_initial)
    test_accuracy_initial = accuracy_score(y_test, test_pred_initial)

    print(f"\nInitial training accuracy: {train_accuracy_initial:.4f}")
    print(f"Initial test accuracy:     {test_accuracy_initial:.4f}")

    # --------------------------------------------------------------
    # 6. Overfitting / underfitting detection
    # --------------------------------------------------------------
    print("\n5. OVERFITTING / UNDERFITTING CHECK")

    accuracy_gap = train_accuracy_initial - test_accuracy_initial

    if accuracy_gap >= .05:
        fit_status = "OVERFITTING DETECTED"
        print(f"{fit_status} (train-test gap = {accuracy_gap:.4f})")
    elif train_accuracy_initial < 0.60:
        fit_status = "POSSIBLE UNDERFITTING"
        print(f"{fit_status}")
    else:
        fit_status = "NO SIGNIFICANT OVERFITTING DETECTED"
        print(fit_status)

    # --------------------------------------------------------------
    # 7. Correction + retraining
    # --------------------------------------------------------------
    print("\n6. CORRECTION TECHNIQUE + RETRAINING")

    corrected_model = build_corrected_model()
    corrected_model.fit(X_train, y_train)

    train_pred = corrected_model.predict(X_train)
    test_pred = corrected_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"Corrected training accuracy: {train_accuracy:.4f}")
    print(f"Corrected test accuracy:     {test_accuracy:.4f}")

    # --------------------------------------------------------------
    # 8. Final test evaluation
    # --------------------------------------------------------------
    print("\n7. FINAL TEST EVALUATION")

    accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(
        y_test, test_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, test_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, test_pred, average="macro", zero_division=0
    )

    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_pred,
            labels=["LOW", "MEDIUM", "HIGH"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            test_pred,
            labels=["LOW", "MEDIUM", "HIGH"],
        )
    )

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "cv_accuracy": cv_metrics["accuracy"],
        "cv_precision": cv_metrics["precision"],
        "cv_recall": cv_metrics["recall"],
        "cv_f1": cv_metrics["f1"],
        "initial_train_accuracy": train_accuracy_initial,
        "initial_test_accuracy": test_accuracy_initial,
        "corrected_train_accuracy": train_accuracy,
        "final_test_accuracy": accuracy,
        "final_test_precision": precision,
        "final_test_recall": recall,
        "final_test_f1": f1,
        "fit_status": fit_status,
        "risk_threshold_low_medium": float(q1),
        "risk_threshold_medium_high": float(q2),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
    }

    pd.DataFrame([metrics]).to_csv(
        OUTPUTS_DIR / "classification_metrics.csv",
        index=False,
    )

    print("\nSaved: outputs/classification_metrics.csv")

    return {
        "model": corrected_model,
        "metrics": metrics,
        "leakage_checks": leakage,
        "thresholds": (q1, q2),
    }