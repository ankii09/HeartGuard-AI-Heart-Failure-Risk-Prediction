
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

class HeartFailurePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Failure Risk Predictor")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Load model and scaler
        try:
            self.model = joblib.load('heart_failure_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model or scaler: {str(e)}")
            self.root.quit()
            return
        
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Heart Failure Risk Prediction", 
            font=('Helvetica', 18, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Patient Information", padding=10)
        input_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Feature entries
        self.entries = {}
        features = [
            ('Age', 'years'),
            ('Anaemia', '1 = yes, 0 = no'),
            ('Creatinine Phosphokinase', 'mcg/L'),
            ('Diabetes', '1 = yes, 0 = no'),
            ('Ejection Fraction', 'percentage'),
            ('High Blood Pressure', '1 = yes, 0 = no'),
            ('Platelets', 'kiloplatelets/mL'),
            ('Serum Creatinine', 'mg/dL'),
            ('Serum Sodium', 'mEq/L'),
            ('Sex', '1 = male, 0 = female'),
            ('Smoking', '1 = yes, 0 = no'),
            ('Time', 'days')
        ]
        
        for i, (feature, unit) in enumerate(features):
            ttk.Label(input_frame, text=f"{feature}:").grid(row=i, column=0, sticky="e", padx=5, pady=2)
            entry = ttk.Entry(input_frame, width=20)
            entry.grid(row=i, column=1, padx=5, pady=2, sticky="w")
            ttk.Label(input_frame, text=unit, font=('Helvetica', 8)).grid(row=i, column=2, sticky="w", padx=5)
            self.entries[feature] = entry
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=20)
        
        predict_btn = ttk.Button(button_frame, text="Predict", command=self.predict)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_entries)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Result frame
        self.result_frame = ttk.LabelFrame(main_frame, text="Prediction Result", padding=10)
        self.result_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        
        self.result_label = ttk.Label(
            self.result_frame, 
            text="Please enter patient data and click 'Predict'",
            font=('Helvetica', 11)
        )
        self.result_label.pack(pady=10)
        
        # Feature importance image
        try:
            self.feature_img = Image.open('feature_importance.png')
            self.feature_img = self.feature_img.resize((400, 300), Image.Resampling.LANCZOS)
            self.feature_photo = ImageTk.PhotoImage(self.feature_img)
            
            img_label = ttk.Label(main_frame, image=self.feature_photo)
            img_label.grid(row=1, column=1, rowspan=3, padx=20, pady=10, sticky="n")
        except Exception as e:
            print(f"Could not load feature importance image: {e}")
    
    def predict(self):
        try:
            # Get values from entries
            feature_values = []
            features_ordered = [
                'Age', 'Anaemia', 'Creatinine Phosphokinase', 'Diabetes',
                'Ejection Fraction', 'High Blood Pressure', 'Platelets',
                'Serum Creatinine', 'Serum Sodium', 'Sex', 'Smoking', 'Time'
            ]
            
            for feature in features_ordered:
                value = self.entries[feature].get()
                if not value:
                    messagebox.showwarning("Missing Data", f"Please enter a value for {feature}")
                    return
                try:
                    feature_values.append(float(value))
                except ValueError:
                    messagebox.showerror("Invalid Input", f"Please enter a valid number for {feature}")
                    return
            
            # Scale the input
            input_data = self.scaler.transform([feature_values])
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            probability = self.model.predict_proba(input_data)[0][1]
            
            # Display result
            result_text = (
                f"Prediction: {'High Risk of Heart Failure' if prediction == 1 else 'Low Risk of Heart Failure'}\n"
                f"Probability: {probability*100:.2f}%"
            )
            self.result_label.config(text=result_text)
            
            # Change color based on risk
            color = "red" if prediction == 1 else "green"
            self.result_label.config(foreground=color)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")
    
    def clear_entries(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_label.config(
            text="Please enter patient data and click 'Predict'",
            foreground='black'
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = HeartFailurePredictorApp(root)
    root.mainloop()
