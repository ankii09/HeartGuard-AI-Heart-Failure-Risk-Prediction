from pathlib import Path
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
from PIL import Image, ImageTk

class ModernHeartApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.base_dir = Path(__file__).resolve().parent
        self.model_path = self.base_dir / "models" / "heart_failure_model.pkl"
        self.scaler_path = self.base_dir / "models" / "scaler.pkl"
        self.feature_img_path = self.base_dir / "reports" / "feature_importance.png"

        # Configure window
        self.title("HeartGuard AI")
        self.geometry("1000x700")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Load model and assets
        self.load_assets()

        # Create UI
        self.create_widgets()

    def load_assets(self):
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.feature_img = Image.open(self.feature_img_path).resize((400, 300))
            self.feature_photo = ImageTk.PhotoImage(self.feature_img)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load required files: {str(e)}")
            self.quit()
    
    def create_widgets(self):
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Main container
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Left panel - Input form
        self.create_input_panel()
        
        # Right panel - Results
        self.create_results_panel()
    
    def create_input_panel(self):
        input_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        input_frame.grid(row=0, column=0, padx=(0, 10), pady=10, sticky="nsew")
        
        # Title
        ctk.CTkLabel(
            input_frame, 
            text="Patient Information",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=(10, 5), padx=10, anchor="w")
        
        # Scrollable frame
        canvas = ctk.CTkCanvas(input_frame, highlightthickness=0)
        scrollbar = ctk.CTkScrollbar(input_frame, orientation="vertical", command=canvas.yview)
        self.scrollable_frame = ctk.CTkFrame(canvas, corner_radius=0)
        
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Input fields
        self.entries = {}
        features = [
            ('Age', 'years'),
            ('Anaemia', '1 = yes, 0 = no'),
            ('Creatinine Phosphokinase', 'mcg/L'),
            ('Diabetes', '1 = yes, 0 = no'),
            ('Ejection Fraction', '%'),
            ('High Blood Pressure', '1 = yes, 0 = no'),
            ('Platelets', 'kiloplatelets/mL'),
            ('Serum Creatinine', 'mg/dL'),
            ('Serum Sodium', 'mEq/L'),
            ('Sex', '1 = male, 0 = female'),
            ('Smoking', '1 = yes, 0 = no'),
            ('Time', 'days')
        ]
        
        for feature, unit in features:
            frame = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
            frame.pack(fill="x", padx=5, pady=2)
            
            ctk.CTkLabel(frame, text=f"{feature}:", width=180).pack(side="left")
            
            entry = ctk.CTkEntry(frame, width=150)
            entry.pack(side="left", padx=5)
            
            ctk.CTkLabel(
                frame, 
                text=unit, 
                text_color=("gray50", "gray60"),
                font=("Segoe UI", 9)
            ).pack(side="left")
            
            self.entries[feature] = entry
        
        # Buttons
        btn_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(10, 5))
        
        ctk.CTkButton(
            btn_frame,
            text="Predict Risk",
            command=self.predict,
            height=40,
            corner_radius=8
        ).pack(side="left", expand=True, padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="Clear All",
            command=self.clear_entries,
            fg_color=("gray60", "gray40"),
            hover_color=("gray50", "gray30"),
            height=40,
            corner_radius=8
        ).pack(side="left", expand=True, padx=5)
    
    def create_results_panel(self):
        results_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        results_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 0), pady=10)
        
        # Risk assessment
        ctk.CTkLabel(
            results_frame,
            text="Risk Assessment",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=(10, 5), padx=10, anchor="w")
        
        self.result_label = ctk.CTkLabel(
            results_frame,
            text="Enter patient data and click 'Predict Risk'",
            font=("Segoe UI", 12),
            text_color=("gray50", "gray60"),
            wraplength=300,
            justify="center"
        )
        self.result_label.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Feature importance
        ctk.CTkLabel(
            results_frame,
            text="Feature Importance",
            font=("Segoe UI", 12, "bold")
        ).pack(pady=(10, 5))
        
        ctk.CTkLabel(
            results_frame,
            image=self.feature_photo,
            text=""
        ).pack(pady=(0, 10))
    
    def predict(self):
        try:
            features = [
                'Age', 'Anaemia', 'Creatinine Phosphokinase', 'Diabetes',
                'Ejection Fraction', 'High Blood Pressure', 'Platelets',
                'Serum Creatinine', 'Serum Sodium', 'Sex', 'Smoking', 'Time'
            ]
            
            values = []
            for feature in features:
                value = self.entries[feature].get().strip()
                if not value:
                    messagebox.showwarning("Missing Data", f"Please enter a value for {feature}")
                    self.entries[feature].focus()
                    return
                try:
                    values.append(float(value))
                except ValueError:
                    messagebox.showerror("Invalid Input", f"Please enter a valid number for {feature}")
                    self.entries[feature].focus()
                    return
            
            # Scale and predict
            input_data = self.scaler.transform([values])
            prediction = self.model.predict(input_data)[0]
            probability = self.model.predict_proba(input_data)[0][1]
            
            # Update result
            risk = "High Risk" if prediction == 1 else "Low Risk"
            color = "#ef5350" if prediction == 1 else "#66bb6a"
            
            self.result_label.configure(
                text=f"Risk Level: {risk}\n\n"
                     f"Probability: {probability*100:.1f}%\n\n"
                     f"{'Consult a healthcare professional' if prediction == 1 else 'Continue regular check-ups'}",
                text_color=color,
                font=("Segoe UI", 12, "bold")
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def clear_entries(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_label.configure(
            text="Enter patient data and click 'Predict Risk'",
            text_color=("gray50", "gray60"),
            font=("Segoe UI", 12)
        )

if __name__ == "__main__":
    app = ModernHeartApp()
    app.mainloop()
