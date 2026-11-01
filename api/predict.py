import json
import math
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "heart_failure_pipeline.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
FEATURES = [
    ("age", "Age", 40.0, 95.0, "years", "Patient age in years."),
    ("anaemia", "Anaemia", 0.0, 1.0, "yes/no", "Whether anaemia was present (1 = yes, 0 = no)."),
    (
        "creatinine_phosphokinase",
        "Creatinine phosphokinase",
        23.0,
        7861.0,
        "mcg/L",
        "Creatinine phosphokinase enzyme level in the blood.",
    ),
    ("diabetes", "Diabetes", 0.0, 1.0, "yes/no", "Whether diabetes was present (1 = yes, 0 = no)."),
    ("ejection_fraction", "Ejection fraction", 14.0, 80.0, "%", "Percentage of blood leaving the heart at each contraction."),
    (
        "high_blood_pressure",
        "High blood pressure",
        0.0,
        1.0,
        "yes/no",
        "Whether high blood pressure was present (1 = yes, 0 = no).",
    ),
    ("platelets", "Platelets", 25100.0, 850000.0, "kiloplatelets/mL", "Platelet count in the blood."),
    ("serum_creatinine", "Serum creatinine", 0.5, 9.4, "mg/dL", "Serum creatinine level."),
    ("serum_sodium", "Serum sodium", 113.0, 148.0, "mEq/L", "Serum sodium level."),
    ("sex", "Sex", 0.0, 1.0, "0/1", "Dataset encoding: 1 = male, 0 = female."),
    ("smoking", "Smoking", 0.0, 1.0, "yes/no", "Smoking status (1 = yes, 0 = no)."),
    ("time", "Follow-up time", 4.0, 285.0, "days", "Follow-up period in the original dataset."),
]
FEATURE_NAMES = [feature[0] for feature in FEATURES]
THRESHOLD = 0.5
OPTIONS = {
    "anaemia": [{"value": 0, "label": "0 — No"}, {"value": 1, "label": "1 — Yes"}],
    "diabetes": [{"value": 0, "label": "0 — No"}, {"value": 1, "label": "1 — Yes"}],
    "high_blood_pressure": [{"value": 0, "label": "0 — No"}, {"value": 1, "label": "1 — Yes"}],
    "sex": [{"value": 0, "label": "0 — Female"}, {"value": 1, "label": "1 — Male"}],
    "smoking": [{"value": 0, "label": "0 — No"}, {"value": 1, "label": "1 — Yes"}],
}

_pipeline = None
_metadata = None


def load_resources():
    global _pipeline, _metadata
    if _pipeline is None:
        if MODEL_PATH.exists():
            _pipeline = joblib.load(MODEL_PATH)
        else:
            model = joblib.load(ROOT / "models" / "heart_failure_model.pkl")
            scaler = joblib.load(ROOT / "models" / "scaler.pkl")
            _pipeline = Pipeline([("scaler", scaler), ("model", model)])
    if _metadata is None:
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            _metadata = json.load(file)
    return _pipeline, _metadata


def dashboard_data():
    pipeline, metadata = load_resources()
    model = pipeline.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    global_importance = []
    if importances is not None:
        global_importance = [
            {"feature": FEATURES[index][1], "importance": float(value)}
            for index, value in enumerate(importances)
        ]
        global_importance.sort(key=lambda item: item["importance"], reverse=True)

    return {
        "metadata": metadata,
        "features": [
            {
                "name": name,
                "label": label,
                "min": minimum,
                "max": maximum,
                "unit": unit,
                "description": description,
                "binary": minimum == 0.0 and maximum == 1.0,
                "options": OPTIONS.get(name, []),
            }
            for name, label, minimum, maximum, unit, description in FEATURES
        ],
        "global_importance": global_importance,
    }


def predict(payload):
    pipeline, metadata = load_resources()
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), dict):
        raise ValueError("Request body must include a 'features' object.")

    supplied = payload["features"]
    missing = [name for name in FEATURE_NAMES if name not in supplied]
    if missing:
        raise ValueError(f"Missing required values: {', '.join(missing)}.")
    unexpected = [name for name in supplied if name not in FEATURE_NAMES]
    if unexpected:
        raise ValueError(f"Unexpected feature names: {', '.join(unexpected)}.")

    values = {}
    for name, label, minimum, maximum, unit, _ in FEATURES:
        raw_value = supplied[name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{label} must be a number.")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number.")
        if not minimum <= value <= maximum:
            raise ValueError(f"{label} must be within the observed training range of {minimum:g} to {maximum:g} {unit}.")
        if minimum == 0.0 and maximum == 1.0 and value not in (0.0, 1.0):
            raise ValueError(f"{label} must be 0 or 1.")
        values[name] = value

    sample = pd.DataFrame([values], columns=FEATURE_NAMES)
    probabilities = pipeline.predict_proba(sample)[0]
    classes = list(pipeline.named_steps["model"].classes_)
    if 1 not in classes or 0 not in classes:
        raise RuntimeError("The trained model is missing the expected binary classes.")
    positive_probability = float(probabilities[classes.index(1)])
    negative_probability = float(probabilities[classes.index(0)])
    threshold = float(metadata.get("classification_threshold", THRESHOLD))
    predicted_class = int(positive_probability >= threshold)

    # One-feature-at-a-time sensitivity against the fitted scaler's training mean.
    # These deltas are not additive explanations or statements of medical causation.
    scaler = pipeline.named_steps.get("scaler")
    if scaler is not None and hasattr(scaler, "mean_"):
        baseline = pd.DataFrame([scaler.mean_], columns=FEATURE_NAMES)
    else:
        baseline = sample.copy()
    baseline_probability = float(pipeline.predict_proba(baseline)[0][classes.index(1)])
    local_sensitivity = []
    for name, label, _, _, unit, _ in FEATURES:
        perturbed = baseline.copy()
        perturbed.loc[0, name] = values[name]
        changed_probability = float(pipeline.predict_proba(perturbed)[0][classes.index(1)])
        local_sensitivity.append(
            {
                "feature": label,
                "value": values[name],
                "unit": unit,
                "delta": changed_probability - baseline_probability,
            }
        )
    local_sensitivity.sort(key=lambda item: abs(item["delta"]), reverse=True)

    return {
        "predicted_class": predicted_class,
        "class_label": (
            "Death event recorded in dataset"
            if predicted_class == 1
            else "No death event recorded in dataset"
        ),
        "positive_probability": positive_probability,
        "negative_probability": negative_probability,
        "threshold": threshold,
        "target_definition": metadata.get("target_definition", {}).get(
            "description",
            "DEATH_EVENT = 1 means a death event was recorded; 0 means no death event was recorded.",
        ),
        "features": values,
        "local_sensitivity": local_sensitivity,
        "baseline_probability": baseline_probability,
        "notice": metadata.get(
            "educational_notice",
            "Educational model only. Not clinically validated; do not use for diagnosis or treatment.",
        ),
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, body):
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path.rstrip("/") != "/api/predict":
            self._send_json(404, {"error": "Not found."})
            return
        try:
            self._send_json(200, dashboard_data())
        except Exception as exc:
            self._send_json(500, {"error": f"Unable to load model information: {exc}"})

    def do_POST(self):
        if self.path.rstrip("/") != "/api/predict":
            self._send_json(404, {"error": "Not found."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 16384:
                self._send_json(400, {"error": "Request body is empty or too large."})
                return
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            self._send_json(200, predict(payload))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON."})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": f"Prediction failed: {exc}"})

    def log_message(self, format, *args):
        return
