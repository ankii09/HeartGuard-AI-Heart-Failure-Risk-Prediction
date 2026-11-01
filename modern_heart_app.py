from __future__ import annotations

import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import joblib
import numpy as np
import pandas as pd
from PIL import Image

try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    SHAP_AVAILABLE = False

FEATURES = [
    ("Age", "age", "years"),
    ("Anaemia", "anaemia", "1 = yes, 0 = no"),
    ("Creatinine Phosphokinase", "creatinine_phosphokinase", "mcg/L"),
    ("Diabetes", "diabetes", "1 = yes, 0 = no"),
    ("Ejection Fraction", "ejection_fraction", "%"),
    ("High Blood Pressure", "high_blood_pressure", "1 = yes, 0 = no"),
    ("Platelets", "platelets", "kiloplatelets/mL"),
    ("Serum Creatinine", "serum_creatinine", "mg/dL"),
    ("Serum Sodium", "serum_sodium", "mEq/L"),
    ("Sex", "sex", "1 = male, 0 = female"),
    ("Smoking", "smoking", "1 = yes, 0 = no"),
    ("Time", "time", "days"),
]

EDUCATIONAL_NOTICE = "This is an educational machine learning model and is not clinically validated. It must not be used for diagnosis or treatment."
TARGET_DEFINITION = "DEATH_EVENT = 1 means a death event was recorded in the dataset; 0 means no death event was recorded."


class ModernHeartApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.base_dir = Path(__file__).resolve().parent
        self.model_path = self.base_dir / "models" / "heart_failure_model.pkl"
        self.scaler_path = self.base_dir / "models" / "scaler.pkl"
        self.pipeline_path = self.base_dir / "models" / "heart_failure_pipeline.pkl"
        self.metadata_path = self.base_dir / "models" / "model_metadata.json"
        self.global_feature_path = self.base_dir / "reports" / "feature_importance.png"
        self.roc_curve_path = self.base_dir / "reports" / "roc_curve.png"
        self.pr_curve_path = self.base_dir / "reports" / "pr_curve.png"
        self.local_explanation_path = self.base_dir / "reports" / "local_explanation.png"

        self.model = None
        self.scaler = None
        self.pipeline = None
        self.model_info = self._load_model_info()
        self.threshold = float(self.model_info.get("classification_threshold", 0.5))
        self.current_prediction = None

        self.title("HeartGuard AI")
        self.geometry("1180x760")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.global_feature_photo = self._load_image(self.global_feature_path, 420, 300)
        self.roc_curve_photo = self._load_image(self.roc_curve_path, 420, 260)
        self.pr_curve_photo = self._load_image(self.pr_curve_path, 420, 260)
        self.local_plot_photo = self._load_image(self.local_explanation_path, 420, 300)

        self.load_assets()
        self.create_widgets()

    def _load_model_info(self):
        if self.metadata_path.exists():
            try:
                with self.metadata_path.open("r", encoding="utf-8") as file:
                    return json.load(file)
            except Exception:
                return {}
        return {}

    def _load_image(self, image_path: Path, width: int, height: int):
        if not image_path.exists():
            return None
        try:
            image = Image.open(image_path).resize((width, height), Image.Resampling.LANCZOS)
            return ctk.CTkImage(light_image=image, dark_image=image, size=(width, height))
        except Exception:
            return None

    def load_assets(self):
        try:
            if self.pipeline_path.exists():
                self.pipeline = joblib.load(self.pipeline_path)
                self.model = self.pipeline.named_steps["model"]
                self.scaler = self.pipeline.named_steps["scaler"]
            else:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
        except Exception as exc:  # pragma: no cover - environment-dependent
            messagebox.showerror("Error", f"Failed to load required model files: {exc}")
            self.destroy()
            return

    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_container = ctk.CTkFrame(self, corner_radius=18)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        self.header = ctk.CTkFrame(self.main_container, corner_radius=14, fg_color=("#f8fbff", "#1d2530"))
        self.header.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 10))
        self.header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.header,
            text="HeartGuard AI",
            font=("Segoe UI", 24, "bold"),
            text_color=("#1d4ed8", "#7dd3fc"),
        ).grid(row=0, column=0, padx=18, pady=12, sticky="w")

        self.status_label = ctk.CTkLabel(
            self.header,
            text="Ready",
            text_color=("#475569", "#cbd5e1"),
            font=("Segoe UI", 11),
        )
        self.status_label.grid(row=0, column=1, padx=(0, 18), pady=12, sticky="e")

        self.tabview = ctk.CTkTabview(self.main_container, width=1100, height=620)
        self.tabview.grid(row=1, column=0, padx=18, pady=(0, 18), sticky="nsew")
        self.tabview.add("Prediction")
        self.tabview.add("Explainability")
        self.tabview.add("Model Performance")
        self.tabview.add("About Model")
        self.tabview.set("Prediction")

        self._build_prediction_tab()
        self._build_explainability_tab()
        self._build_model_performance_tab()
        self._build_about_model_tab()

    def _create_input_fields(self, parent):
        self.entries = {}
        for feature, _, unit in FEATURES:
            frame = ctk.CTkFrame(parent, fg_color="transparent")
            frame.pack(fill="x", padx=10, pady=6)

            label = ctk.CTkLabel(frame, text=f"{feature}:", width=180, anchor="w")
            label.pack(side="left")

            entry = ctk.CTkEntry(frame, width=180)
            entry.pack(side="left", padx=8)

            ctk.CTkLabel(
                frame,
                text=unit,
                text_color=("#64748b", "#94a3b8"),
                font=("Segoe UI", 9),
            ).pack(side="left")
            self.entries[feature] = entry

    def _build_prediction_tab(self):
        tab = self.tabview.tab("Prediction")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        input_panel = ctk.CTkFrame(tab, corner_radius=16, fg_color=("#f8fbff", "#1f2937"))
        input_panel.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        ctk.CTkLabel(
            input_panel,
            text="Patient Information",
            font=("Segoe UI", 18, "bold"),
            text_color=("#0f172a", "#e2e8f0"),
        ).pack(padx=16, pady=(14, 10), anchor="w")

        canvas = ctk.CTkCanvas(input_panel, highlightthickness=0, background="#f8fbff")
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 12))
        scroll = ctk.CTkScrollbar(input_panel, orientation="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y", pady=(0, 12))

        form_frame = ctk.CTkFrame(canvas, corner_radius=8, fg_color="transparent")
        canvas.create_window((0, 0), window=form_frame, anchor="nw")
        form_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

        self._create_input_fields(form_frame)
        form_frame.update_idletasks()
        canvas.configure(yscrollcommand=scroll.set)

        button_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=(12, 8))

        ctk.CTkButton(
            button_frame,
            text="Predict Risk",
            command=self.predict,
            height=38,
            corner_radius=10,
        ).pack(side="left", expand=True, padx=6)

        ctk.CTkButton(
            button_frame,
            text="Clear All",
            command=self.clear_entries,
            fg_color=("#64748b", "#475569"),
            hover_color=("#475569", "#334155"),
            height=38,
            corner_radius=10,
        ).pack(side="left", expand=True, padx=6)

        result_panel = ctk.CTkFrame(tab, corner_radius=16, fg_color=("#f8fbff", "#1f2937"))
        result_panel.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)

        ctk.CTkLabel(
            result_panel,
            text="Prediction Result",
            font=("Segoe UI", 18, "bold"),
            text_color=("#0f172a", "#e2e8f0"),
        ).pack(anchor="w", padx=18, pady=(18, 10))

        self.result_card = ctk.CTkFrame(result_panel, corner_radius=16, fg_color=("#f8fbff", "#10273d"))
        self.result_card.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self.result_card.grid_columnconfigure(0, weight=1)

        self.gauge_canvas = tk.Canvas(
            self.result_card,
            width=220,
            height=120,
            bg="#f8fbff",
            highlightthickness=0,
        )
        self.gauge_canvas.grid(row=0, column=0, pady=(12, 0))

        self.result_percent_label = ctk.CTkLabel(
            self.result_card,
            text="--%",
            font=("Segoe UI", 28, "bold"),
            text_color=("#0f172a", "#e2e8f0"),
        )
        self.result_percent_label.grid(row=1, column=0, pady=(0, 4))

        self.result_class_label = ctk.CTkLabel(
            self.result_card,
            text="Awaiting prediction",
            font=("Segoe UI", 18, "bold"),
            text_color=("#475569", "#e2e8f0"),
        )
        self.result_class_label.grid(row=2, column=0, pady=4)

        self.probability_label = ctk.CTkLabel(
            self.result_card,
            text="Positive probability: --\nNegative probability: --",
            font=("Segoe UI", 12, "bold"),
            text_color=("#0f172a", "#f8fafc"),
            wraplength=260,
            justify="center",
        )
        self.probability_label.grid(row=3, column=0, pady=(10, 6))

        self.threshold_label = ctk.CTkLabel(
            self.result_card,
            text="Classification threshold: 0.50",
            text_color=("#475569", "#cbd5e1"),
            font=("Segoe UI", 11),
        )
        self.threshold_label.grid(row=4, column=0, pady=(4, 10))

        self.educational_label = ctk.CTkLabel(
            self.result_card,
            text=EDUCATIONAL_NOTICE,
            font=("Segoe UI", 10),
            text_color=("#475569", "#cbd5e1"),
            wraplength=330,
            justify="center",
        )
        self.educational_label.grid(row=5, column=0, padx=20, pady=(4, 16))

        self.summary_box = ctk.CTkTextbox(result_panel, height=180, wrap="word")
        self.summary_box.pack(fill="both", expand=True, padx=16, pady=(0, 18))
        self.summary_box.insert("end", "Submitted feature values will appear here.\n")
        self.summary_box.configure(state="disabled")

        self._reset_result_view()

    def _build_explainability_tab(self):
        tab = self.tabview.tab("Explainability")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)

        local_frame = ctk.CTkFrame(tab, corner_radius=16, fg_color=("#f8fbff", "#1f2937"))
        local_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        ctk.CTkLabel(local_frame, text="Local Explanation", font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=18, pady=(18, 8))

        if self.local_plot_photo is not None:
            ctk.CTkLabel(local_frame, image=self.local_plot_photo, text="").pack(padx=10, pady=(0, 8))
        else:
            ctk.CTkLabel(local_frame, text="Local explanation chart will appear after prediction.", text_color=("#64748b", "#94a3b8")).pack(padx=10, pady=(0, 12))

        self.local_summary_label = ctk.CTkLabel(
            local_frame,
            text="SHAP local explanation is available when the optional shap package is installed. Otherwise, feature contributions are approximated from the trained model weights.",
            justify="left",
            wraplength=420,
            text_color=("#475569", "#cbd5e1"),
        )
        self.local_summary_label.pack(anchor="w", padx=18, pady=(0, 16))

        global_frame = ctk.CTkFrame(tab, corner_radius=16, fg_color=("#f8fbff", "#1f2937"))
        global_frame.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)

        ctk.CTkLabel(global_frame, text="Global Feature Importance", font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=18, pady=(18, 8))

        if self.global_feature_photo is not None:
            ctk.CTkLabel(global_frame, image=self.global_feature_photo, text="").pack(padx=10, pady=(0, 12))
        else:
            ctk.CTkLabel(global_frame, text="Global feature importance image unavailable.", text_color=("#64748b", "#94a3b8")).pack(padx=10, pady=(0, 12))

        self.explanation_notice = ctk.CTkLabel(
            global_frame,
            text="Feature importance reflects model behavior, not medical causation.",
            text_color=("#475569", "#cbd5e1"),
            wraplength=420,
            justify="center",
        )
        self.explanation_notice.pack(padx=18, pady=(0, 18))

    def _build_model_performance_tab(self):
        tab = self.tabview.tab("Model Performance")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)

        metrics_frame = ctk.CTkFrame(tab, corner_radius=16, fg_color=("#f8fbff", "#1f2937"))
        metrics_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        ctk.CTkLabel(metrics_frame, text="Test Metrics", font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=16, pady=(18, 10))

        self.metric_box = ctk.CTkTextbox(metrics_frame, height=220, wrap="word")
        self.metric_box.pack(fill="both", expand=True, padx=16, pady=(0, 18))
        self.metric_box.configure(state="disabled")
        self._render_metrics_text()

        chart_frame = ctk.CTkFrame(tab, corner_radius=16, fg_color=("#f8fbff", "#1f2937"))
        chart_frame.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)

        ctk.CTkLabel(chart_frame, text="Performance Curves", font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=18, pady=(18, 8))
        if self.roc_curve_photo is not None:
            ctk.CTkLabel(chart_frame, image=self.roc_curve_photo, text="").pack(padx=10, pady=(0, 8))
        if self.pr_curve_photo is not None:
            ctk.CTkLabel(chart_frame, image=self.pr_curve_photo, text="").pack(padx=10, pady=(0, 8))

    def _build_about_model_tab(self):
        tab = self.tabview.tab("About Model")

        info_frame = ctk.CTkFrame(tab, corner_radius=16, fg_color=("#f8fbff", "#1f2937"))
        info_frame.pack(fill="both", expand=True, padx=18, pady=18)

        ctk.CTkLabel(info_frame, text="Model Information", font=("Segoe UI", 22, "bold")).pack(anchor="w", padx=18, pady=(18, 12))

        self.about_textbox = ctk.CTkTextbox(info_frame, height=430, wrap="word")
        self.about_textbox.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.about_textbox.configure(state="disabled")
        self._render_about_model_text()

    def _render_metrics_text(self):
        default = "No metrics available. Retrain the model to generate performance reports."
        metrics = self.model_info.get("test_metrics", {})
        if not metrics:
            self.metric_box.configure(state="normal")
            self.metric_box.delete("1.0", "end")
            self.metric_box.insert("end", default)
            self.metric_box.configure(state="disabled")
            return

        lines = [
            f"Accuracy: {metrics.get('accuracy', 'n/a')}",
            f"Precision: {metrics.get('precision', 'n/a')}",
            f"Recall: {metrics.get('recall', 'n/a')}",
            f"F1-score: {metrics.get('f1', 'n/a')}",
            f"ROC-AUC: {metrics.get('roc_auc', 'n/a')}",
            f"PR-AUC: {metrics.get('pr_auc', 'n/a')}",
            "",
            "Confusion matrix:",
            str(metrics.get("confusion_matrix", [])),
        ]
        self.metric_box.configure(state="normal")
        self.metric_box.delete("1.0", "end")
        self.metric_box.insert("end", "\n".join(lines))
        self.metric_box.configure(state="disabled")

    def _render_about_model_text(self):
        info = self.model_info
        dataset_source = info.get("dataset_source", "UCI heart failure clinical records dataset")
        methodology = info.get("training_methodology", "Not available")
        selected_model = info.get("selected_model", "Random Forest")
        training_date = info.get("training_date_utc", "Not available")
        target_definition = info.get("target_definition", {}).get("description", TARGET_DEFINITION)

        lines = [
            f"Dataset source: {dataset_source}",
            f"Target definition: {target_definition}",
            f"Number of training samples: {info.get('training_samples', 'n/a')}",
            f"Number of features: {info.get('num_features', 'n/a')}",
            f"Selected model: {selected_model}",
            f"Training methodology: {methodology}",
            f"Model version: {info.get('model_version', 'v2.0.0')}",
            f"Training date: {training_date}",
            "",
            "Scientific reliability:",
            "- The model uses the same preprocessing pipeline during training and prediction.",
            "- Separate validation and test data are used for model selection and final evaluation.",
            "- The reported probabilities are not confidence scores and should be interpreted as educational outputs only.",
            "- This application is not clinically validated and must not be used for diagnosis or treatment.",
        ]
        self.about_textbox.configure(state="normal")
        self.about_textbox.delete("1.0", "end")
        self.about_textbox.insert("end", "\n".join(lines))
        self.about_textbox.configure(state="disabled")

    def _draw_gauge(self, probability):
        canvas = self.gauge_canvas
        canvas.delete("all")
        cx, cy = 110, 92
        radius = 72
        start_angle = 180
        end_angle = 0

        canvas.create_arc(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            start=start_angle,
            extent=end_angle - start_angle,
            style="arc",
            outline="#dfeaf7",
            width=12,
        )

        color = "#22c55e" if probability < 0.5 else "#ef4444"
        fill_angle = 180 - probability * 180
        canvas.create_arc(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            start=180,
            extent=-int(probability * 180),
            style="arc",
            outline=color,
            width=12,
        )

        angle_rad = math.radians(fill_angle)
        x = cx + (radius - 10) * math.cos(angle_rad)
        y = cy - (radius - 10) * math.sin(angle_rad)
        canvas.create_line(cx, cy, x, y, fill=color, width=3, capstyle="round")
        canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=color, outline=color)

    def _reset_result_view(self):
        self._draw_gauge(0.5)
        self.result_percent_label.configure(text="--%", text_color=("#0f172a", "#e2e8f0"))
        self.result_class_label.configure(text="Awaiting prediction", text_color=("#475569", "#e2e8f0"))
        self.probability_label.configure(text="Positive probability: --\nNegative probability: --", text_color=("#0f172a", "#f8fafc"))
        self.threshold_label.configure(text=f"Classification threshold: {self.threshold:.2f}")
        self.educational_label.configure(text=EDUCATIONAL_NOTICE)

    def _build_feature_dataframe(self, feature_values):
        ordered = {model_name: feature_values.get(model_name, 0.0) for _, model_name, _ in FEATURES}
        ordered = {k: ordered[k] for k in ordered}
        return pd.DataFrame([ordered], columns=[model_name for _, model_name, _ in FEATURES])

    def _get_prediction_probabilities(self, input_frame):
        if self.pipeline is not None:
            return self.pipeline.predict_proba(input_frame)[0]
        if self.scaler is not None:
            scaled = self.scaler.transform(input_frame)
            return self.model.predict_proba(scaled)[0]
        raise RuntimeError("No model pipeline is available.")

    def _render_input_summary(self, feature_values):
        entries = []
        for feature_display, model_name, unit in FEATURES:
            value = feature_values[model_name]
            entries.append(f"{feature_display}: {value} {unit}")
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", "\n".join(entries))
        self.summary_box.configure(state="disabled")

    def _render_local_explanation(self, input_frame):
        if SHAP_AVAILABLE and hasattr(self.pipeline, "predict_proba"):
            explainer = shap.Explainer(self.pipeline.named_steps["model"], self.pipeline.named_steps["scaler"].transform(input_frame))
            shap_values = explainer(self.pipeline.named_steps["scaler"].transform(input_frame))
            if hasattr(shap_values, "values"):
                values = shap_values.values[0]
                feature_names = list(input_frame.columns)
                summary = []
                for feature, value in sorted(zip(feature_names, values), key=lambda pair: abs(pair[1]), reverse=True)[:5]:
                    direction = "pushes the positive class higher" if value > 0 else "pushes the positive class lower"
                    summary.append(f"{feature}: {value:.3f} ({direction})")
                self.local_summary_label.configure(text="\n".join(summary))
                return

        feature_names = list(input_frame.columns)
        weighted_importance = []
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_).ravel()
        else:
            importances = np.ones(len(feature_names))

        sample_values = input_frame.iloc[0].to_numpy(dtype=float)
        if hasattr(self.scaler, "var_"):
            scale = np.sqrt(self.scaler.var_)
            centered = (sample_values - self.scaler.mean_) / np.maximum(scale, 1e-8)
        else:
            centered = sample_values

        contribution_scores = centered * importances
        ordered_pairs = sorted(zip(feature_names, contribution_scores), key=lambda pair: abs(pair[1]), reverse=True)
        summary_lines = []
        for feature_name, contribution in ordered_pairs[:5]:
            direction = "pushes the positive class higher" if contribution > 0 else "pushes the positive class lower"
            summary_lines.append(f"{feature_name}: {contribution:.3f} ({direction})")
        self.local_summary_label.configure(text="\n".join(summary_lines))

    def predict(self):
        self.status_label.configure(text="Evaluating patient...", text_color=("#475569", "#cbd5e1"))
        self.update_idletasks()

        try:
            feature_values = {}
            for feature_display, model_name, _ in FEATURES:
                value = self.entries[feature_display].get().strip()
                if not value:
                    messagebox.showwarning("Missing Data", f"Please enter a value for {feature_display}.")
                    self.entries[feature_display].focus()
                    return
                try:
                    numeric_value = float(value)
                    if not np.isfinite(numeric_value):
                        raise ValueError
                    feature_values[model_name] = numeric_value
                except ValueError:
                    messagebox.showerror("Invalid Input", f"Please enter a valid number for {feature_display}.")
                    self.entries[feature_display].focus()
                    return

            input_frame = self._build_feature_dataframe(feature_values)
            probabilities = self._get_prediction_probabilities(input_frame)
            negative_probability = float(probabilities[0])
            positive_probability = float(probabilities[1])
            prediction = int(positive_probability >= self.threshold)
            predicted_label = "Death event predicted" if prediction == 1 else "No death event predicted"

            if prediction == 1:
                self.result_percent_label.configure(text=f"{positive_probability * 100:.0f}%", text_color=("#ef4444", "#fca5a5"))
                self.result_class_label.configure(text="Death event predicted", text_color=("#ef4444", "#fca5a5"))
                self._draw_gauge(positive_probability)
            else:
                self.result_percent_label.configure(text=f"{negative_probability * 100:.0f}%", text_color=("#22c55e", "#86efac"))
                self.result_class_label.configure(text="No death event predicted", text_color=("#22c55e", "#86efac"))
                self._draw_gauge(negative_probability)

            self.probability_label.configure(
                text=f"Positive probability: {positive_probability * 100:.1f}%\nNegative probability: {negative_probability * 100:.1f}%",
                text_color=("#0f172a", "#f8fafc"),
            )
            self.threshold_label.configure(text=f"Classification threshold: {self.threshold:.2f}")
            self.educational_label.configure(text=f"{TARGET_DEFINITION}\n\n{EDUCATIONAL_NOTICE}")

            self._render_input_summary(feature_values)
            self._render_local_explanation(input_frame)
            self.current_prediction = {
                "predicted_label": predicted_label,
                "positive_probability": positive_probability,
                "negative_probability": negative_probability,
                "threshold": self.threshold,
                "sample": feature_values,
            }
            self.status_label.configure(text="Prediction complete", text_color=("#16a34a", "#86efac"))

        except Exception as exc:  # pragma: no cover - GUI-level guard
            messagebox.showerror("Prediction Error", f"An unexpected error occurred during prediction: {exc}")
            self.status_label.configure(text="Prediction failed", text_color=("#ef4444", "#fca5a5"))

    def clear_entries(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self._reset_result_view()
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", "Submitted feature values will appear here.\n")
        self.summary_box.configure(state="disabled")
        self.local_summary_label.configure(text="SHAP local explanation is available when the optional shap package is installed. Otherwise, feature contributions are approximated from the trained model weights.")
        self.status_label.configure(text="Ready", text_color=("#475569", "#cbd5e1"))


if __name__ == "__main__":
    app = ModernHeartApp()
    app.mainloop()
