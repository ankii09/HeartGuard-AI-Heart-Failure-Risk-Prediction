from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "heart_failure_clinical_records_dataset.csv"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"


def load_dataset() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
        df = pd.read_csv(url)
        df.to_csv(DATASET_PATH, index=False)
    return df


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict:
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": 0.5,
    }
    if y_proba is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_proba)), 4)
    return metrics


def save_roc_curve(y_true, y_score, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#3b82f6", lw=2)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_pr_curve(y_true, y_score, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="#10b981", lw=2)
    plt.axhline(y=np.mean(y_true), color="gray", linestyle="--", lw=1, label="Baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_feature_importance_plot(model, feature_names: list[str], output_path: Path) -> None:
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).ravel()
    else:
        return

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance,
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x="Importance", y="Feature", palette="Blues_d")
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_local_explanation_plot(feature_names: list[str], importances: list[float], output_path: Path) -> None:
    local_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": importances,
    }).sort_values("Contribution", key=np.abs, ascending=False)

    plt.figure(figsize=(9, 6))
    colors = ["#10b981" if value >= 0 else "#ef4444" for value in local_df["Contribution"]]
    plt.barh(local_df["Feature"], local_df["Contribution"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Local Feature Contributions")
    plt.xlabel("Approximate contribution to positive-class log-odds")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_local_contributions(pipeline: Pipeline, sample: pd.DataFrame) -> dict:
    model = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]
    feature_names = sample.columns.tolist()
    scaled = scaler.transform(sample)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_.ravel())
    else:
        return {"feature_names": feature_names, "contributions": [0.0] * len(feature_names), "summary": "No local contribution information is available for this model."}

    contribution_scores = (scaled[0] * importances) / (np.std(scaled[0]) + 1e-8)
    contribution_scores = contribution_scores.tolist()
    summary = []
    contribution_pairs = list(zip(feature_names, contribution_scores))
    for feature, value in sorted(contribution_pairs, key=lambda kv: abs(kv[1]), reverse=True)[:5]:
        direction = "increases the positive class score" if value > 0 else "reduces the positive class score"
        summary.append(f"{feature}: {value:.3f} ({direction})")

    return {
        "feature_names": feature_names,
        "contributions": contribution_scores,
        "summary": "\n".join(summary),
    }


def main() -> None:
    df = load_dataset()
    print("\nDataset Shape:", df.shape)
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nMissing Values:\n", df.isnull().sum())

    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    models = {
        "DummyClassifier": DummyClassifier(strategy="stratified"),
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=300, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    validation_results = []
    best_pipeline = None
    best_name = None
    best_val_roc_auc = -1.0

    for model_name, estimator in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)

        val_pred = pipeline.predict(X_val)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, val_pred, val_proba)

        validation_results.append({
            "model": model_name,
            "metrics": val_metrics,
        })

        if val_metrics["roc_auc"] > best_val_roc_auc:
            best_val_roc_auc = val_metrics["roc_auc"]
            best_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No valid model was trained.")

    final_X = pd.concat([X_train, X_val], axis=0)
    final_y = pd.concat([y_train, y_val], axis=0)
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", clone(best_pipeline.named_steps["model"])),
    ])
    final_pipeline.fit(final_X, final_y)

    final_pred = final_pipeline.predict(X_test)
    final_proba = final_pipeline.predict_proba(X_test)[:, 1]
    final_metrics = compute_metrics(y_test, final_pred, final_proba)

    best_model_name = best_name
    feature_names = list(X.columns)
    model = final_pipeline.named_steps["model"]

    save_feature_importance_plot(model, feature_names, REPORTS_DIR / "feature_importance.png")
    save_roc_curve(y_test, final_proba, REPORTS_DIR / "roc_curve.png")
    save_pr_curve(y_test, final_proba, REPORTS_DIR / "pr_curve.png")

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(model.coef_).ravel(),
    }).sort_values("Importance", ascending=False)
    save_local_explanation_plot(
        feature_importance["Feature"].tolist(),
        feature_importance["Importance"].tolist(),
        REPORTS_DIR / "local_explanation.png",
    )

    plt.figure(figsize=(8, 6))
    sns.countplot(x="DEATH_EVENT", data=df)
    plt.title("Distribution of Death Events")
    plt.xlabel("Death Event (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "death_event_distribution.png")
    plt.close()

    joblib.dump(final_pipeline, MODELS_DIR / "heart_failure_pipeline.pkl")
    joblib.dump(final_pipeline.named_steps["model"], MODELS_DIR / "heart_failure_model.pkl")
    joblib.dump(final_pipeline.named_steps["scaler"], MODELS_DIR / "scaler.pkl")

    metadata = {
        "dataset_source": "UCI Heart Failure Clinical Records, archive.ics.uci.edu",
        "target_definition": {
            "field": "DEATH_EVENT",
            "positive_class": 1,
            "negative_class": 0,
            "description": "1 = death event occurred during follow-up; 0 = no death event was recorded."
        },
        "training_samples": int(len(final_X)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "num_features": int(len(feature_names)),
        "training_methodology": "Stratified train/validation/test split (70/15/15) with feature scaling applied only on the training split. The best validation set model was selected via ROC-AUC before final retraining on train+validation and evaluation on the untouched test set.",
        "cross_validation": {
            "strategy": "Hold-out validation",
            "selected_model": best_model_name,
            "validation_auc": round(best_val_roc_auc, 4)
        },
        "classification_threshold": 0.5,
        "model_version": "v2.0.0",
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "educational_notice": "This is an educational machine learning model for research and demonstration. It is not clinically validated and must not be used for diagnosis or treatment.",
        "model_comparison": validation_results,
        "test_metrics": final_metrics,
        "selected_model": best_model_name,
    }
    with MODEL_METADATA_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print("\nSelected model:", best_model_name)
    print("\nValidation results:")
    for row in validation_results:
        print(row["model"], row["metrics"])
    print("\nFinal test metrics:")
    print(final_metrics)

    print("\nModel training complete. Files saved to:")
    print(f"- {MODELS_DIR}")
    print(f"- {REPORTS_DIR}")


if __name__ == "__main__":
    main()
