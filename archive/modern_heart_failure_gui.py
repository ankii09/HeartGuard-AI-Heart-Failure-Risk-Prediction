import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
from PIL import Image, ImageTk
import os

class ModernHeartFailureApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("HeartGuard AI")
        self.geometry("1200x750")
        ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
        
        # Load assets
        self.load_assets()
        
        # Create layout
        self.create_widgets()
        
    def load_assets(self):
        try:
            # Load model and scaler
            self.model = joblib.load('heart_failure_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            
            # Load feature importance image if exists
            if os.path.exists('feature_importance.png'):
                self.feature_img = Image.open('feature_importance.png')
                self.feature_img = self.feature_img.resize((500, 350), Image.Resampling.LANCZOS)
                self.feature_photo = ImageTk.PhotoImage(self.feature_img)
            else:
                self.feature_photo = None
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load required files: {str(e)}")
            self.quit()
    
    def create_widgets(self):
        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create main container
        self.main_container = ctk.CTkFrame(self, corner_radius=10)
        self.main_container.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_container.grid_columnconfigure(1, weight=1)
        
        # Left sidebar for navigation/logo
        self.sidebar = ctk.CTkFrame(self.main_container, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        self.sidebar.grid_rowconfigure(4, weight=1)
        
        # App logo/title
        self.logo_label = ctk.CTkLabel(
            self.sidebar, 
            text="HeartGuard AI",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Navigation buttons
        self.nav_predict = ctk.CTkButton(
            self.sidebar, 
            text="Predict",
            command=self.show_prediction_tab,
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            anchor="w"
        )
        self.nav_predict.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        self.nav_about = ctk.CTkButton(
            self.sidebar, 
            text="About",
            command=self.show_about_tab,
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray70", "gray30"),
            anchor="w"
        )
        self.nav_about.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        # Appearance mode menu
        self.appearance_mode_label = ctk.CTkLabel(
            self.sidebar, 
            text="Appearance Mode:", 
            anchor="w"
        )
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_menu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode
        )
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=(0, 20))
        
        # Main content area
        self.content_frame = ctk.CTkFrame(self.main_container, corner_radius=10)
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # Prediction frame (default view)
        self.prediction_frame = ctk.CTkFrame(self.content_frame, corner_radius=10)
        self.prediction_frame.grid(row=0, column=0, sticky="nsew")
        self.prediction_frame.grid_columnconfigure(0, weight=1)
        self.prediction_frame.grid_rowconfigure(1, weight=1)
        
        # Create input form
        self.create_input_form()
        
        # About frame (initially hidden)
        self.about_frame = ctk.CTkFrame(self.content_frame, corner_radius=10)
        
        # Right sidebar for results/visualization
        self.visualization_frame = ctk.CTkFrame(self.main_container, corner_radius=10)
        self.visualization_frame.grid(row=0, column=2, rowspan=2, sticky="nsew", pady=10)
        self.visualization_frame.grid_columnconfigure(0, weight=1)
        self.visualization_frame.grid_rowconfigure(1, weight=1)
        
        # Visualization title
        ctk.CTkLabel(
            self.visualization_frame, 
            text="Feature Importance",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, pady=(15, 10))
        
        # Feature importance image
        if hasattr(self, 'feature_photo'):
            self.feature_label = ctk.CTkLabel(
                self.visualization_frame, 
                image=self.feature_photo,
                text=""
            )
            self.feature_label.grid(row=1, column=0, padx=10, pady=10)
        
        # Status bar
        self.status_bar = ctk.CTkLabel(
            self.main_container, 
            text="Ready",
            anchor="w",
            font=ctk.CTkFont(size=12),
            fg_color=("gray70", "gray30")
        )
        self.status_bar.grid(row=1, column=1, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        
        # Set default view
        self.show_prediction_tab()
    
    def create_input_form(self):
        # Form container
        form_frame = ctk.CTkFrame(self.prediction_frame, corner_radius=10)
        form_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Title
        ctk.CTkLabel(
            form_frame,
            text="Patient Information",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Input fields
        self.entries = {}
        features = [
            ('Age', 'years', 0, 120),
            ('Anaemia', '1 = yes, 0 = no', 0, 1),
            ('Creatinine Phosphokinase', 'mcg/L', 0, 10000),
            ('Diabetes', '1 = yes, 0 = no', 0, 1),
            ('Ejection Fraction', '%', 0, 100),
            ('High Blood Pressure', '1 = yes, 0 = no', 0, 1),
            ('Platelets', 'kiloplatelets/mL', 0, 1000000),
            ('Serum Creatinine', 'mg/dL', 0, 10),
            ('Serum Sodium', 'mEq/L', 0, 200),
            ('Sex', '1 = male, 0 = female', 0, 1),
            ('Smoking', '1 = yes, 0 = no', 0, 1),
            ('Time', 'days', 0, 1000)
        ]
        
        for i, (feature, unit, min_val, max_val) in enumerate(features):
            # Create frame for each input field
            field_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
            field_frame.grid(row=i+1, column=0, sticky="ew", pady=3)
            field_frame.grid_columnconfigure(1, weight=1)
            
            # Label
            ctk.CTkLabel(
                field_frame,
                text=f"{feature}:",
                width=180,
                anchor="e"
            ).grid(row=0, column=0, padx=(0, 10), sticky="e")
            
            # Entry field with validation
            def create_validate_command(name, min_val, max_val):
                return lambda: self.validate_number(self.entries[name], min_val, max_val)
            
            entry = ctk.CTkEntry(
                field_frame,
                placeholder_text=f"Enter {feature.lower()}",
                width=200
            )
            entry.grid(row=0, column=1, sticky="w")
            entry.bind("<FocusOut>", lambda e, n=feature, mn=min_val, mx=max_val: 
                      self.validate_number(e.widget, mn, mx, n))
            
            # Unit label
            ctk.CTkLabel(
                field_frame,
                text=unit,
                font=ctk.CTkFont(size=10),
                text_color=("gray50", "gray60")
            ).grid(row=0, column=2, padx=(10, 0), sticky="w")
            
            self.entries[feature] = entry
        
        # Buttons
        button_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        button_frame.grid(row=len(features)+1, column=0, columnspan=2, pady=(20, 0))
        
        ctk.CTkButton(
            button_frame,
            text="Predict",
            command=self.predict,
            width=120,
            height=40,
            font=ctk.CTkFont(weight="bold")
        ).pack(side=ctk.LEFT, padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Clear",
            command=self.clear_entries,
            width=120,
            height=40,
            fg_color="gray50",
            hover_color="gray40"
        ).pack(side=ctk.LEFT, padx=10)
        
        # Result display
        self.result_frame = ctk.CTkFrame(
            self.prediction_frame, 
            corner_radius=10,
            fg_color=("gray90", "gray16")
        )
        self.result_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.result_frame.grid_columnconfigure(0, weight=1)
        
        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Enter patient data and click 'Predict' to see results",
            font=ctk.CTkFont(size=14),
            text_color=("gray40", "gray60"),
            wraplength=400,
            justify="center"
        )
        self.result_label.pack(expand=True, padx=20, pady=40)
    
    def validate_number(self, entry_widget, min_val, max_val, feature_name=None):
        try:
            value = entry_widget.get()
            if value:  # Only validate if there's a value
                num = float(value)
                if not (min_val <= num <= max_val):
                    messagebox.showwarning(
                        "Invalid Input",
                        f"{feature_name or 'Value'} must be between {min_val} and {max_val}"
                    )
                    entry_widget.delete(0, tk.END)
                    entry_widget.focus_set()
        except ValueError:
            if entry_widget.get():  # Only show error if field is not empty
                messagebox.showerror("Invalid Input", "Please enter a valid number")
                entry_widget.delete(0, tk.END)
                entry_widget.focus_set()
    
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
                    self.entries[feature].focus_set()
                    return
                try:
                    feature_values.append(float(value))
                except ValueError:
                    messagebox.showerror("Invalid Input", f"Please enter a valid number for {feature}")
                    self.entries[feature].focus_set()
                    return
            
            # Scale the input
            input_data = self.scaler.transform([feature_values])
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            probability = self.model.predict_proba(input_data)[0][1]
            
            # Update result display
            self.show_prediction_result(prediction, probability)
            
            # Update status
            self.status_bar.configure(
                text=f"Prediction completed - {'High Risk' if prediction == 1 else 'Low Risk'} detected"
            )
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")
            self.status_bar.configure(text="Error during prediction")
    
    def show_prediction_result(self, prediction, probability):
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        color = "#ff4d4d" if prediction == 1 else "#4CAF50"
        
        # Clear previous result
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Risk level
        ctk.CTkLabel(
            self.result_frame,
            text=f"Risk Level: {risk_level}",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=color
        ).pack(pady=(20, 10))
        
        # Probability gauge
        gauge_frame = ctk.CTkFrame(self.result_frame, fg_color="transparent")
        gauge_frame.pack(fill="x", padx=40, pady=10)
        
        # Gauge background
        ctk.CTkFrame(
            gauge_frame,
            height=20,
            fg_color=("gray85", "gray20"),
            corner_radius=10
        ).pack(fill="x")
        
        # Gauge fill
        fill_width = min(int(probability * 100), 100)
        fill_frame = ctk.CTkFrame(
            gauge_frame,
            height=20,
            width=fill_width * 2,  # 2px per percent (200px max)
            fg_color=color,
            corner_radius=10
        )
        fill_frame.place(x=0, y=0, anchor="nw")
        
        # Probability percentage
        ctk.CTkLabel(
            self.result_frame,
            text=f"{probability*100:.1f}%",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color=color
        ).pack(pady=(10, 5))
        
        # Interpretation
        interpretation = (
            "The patient is at high risk of heart failure. "
            "Please consult a healthcare professional immediately."
            if prediction == 1 else
            "The patient is at low risk of heart failure. "
            "Regular check-ups are still recommended."
        )
        
        ctk.CTkLabel(
            self.result_frame,
            text=interpretation,
            font=ctk.CTkFont(size=14),
            wraplength=400,
            justify="center"
        ).pack(padx=20, pady=(10, 20))
    
    def clear_entries(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        
        # Reset result display
        for widget in self.result_frame.winfo_children():
            widget.destroy()
            
        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Enter patient data and click 'Predict' to see results",
            font=ctk.CTkFont(size=14),
            text_color=("gray40", "gray60"),
            wraplength=400,
            justify="center"
        )
        self.result_label.pack(expand=True, padx=20, pady=40)
        
        self.status_bar.configure(text="Form cleared")
    
    def show_prediction_tab(self):
        self.about_frame.grid_remove()
        self.prediction_frame.grid()
        self.nav_predict.configure(fg_color=("gray75", "gray25"))
        self.nav_about.configure(fg_color="transparent")
        self.status_bar.configure(text="Ready for prediction")
    
    def show_about_tab(self):
        self.prediction_frame.grid_remove()
        
        # Create about content if not exists
        if not hasattr(self, 'about_initialized'):
            self.initialize_about_tab()
            self.about_initialized = True
        
        self.about_frame.grid(row=0, column=0, sticky="nsew")
        self.nav_about.configure(fg_color=("gray75", "gray25"))
        self.nav_predict.configure(fg_color="transparent")
        self.status_bar.configure(text="Viewing about information")
    
    def initialize_about_tab(self):
        # Configure about frame
        self.about_frame = ctk.CTkFrame(self.content_frame, corner_radius=10)
        self.about_frame.grid_columnconfigure(0, weight=1)
        self.about_frame.grid_rowconfigure(1, weight=1)
        
        # Title
        ctk.CTkLabel(
            self.about_frame,
            text="About HeartGuard AI",
            font=ctk.CTkFont(size=24, weight="bold")
        ).grid(row=0, column=0, pady=(20, 10), padx=20, sticky="w")
        
        # Content
        about_text = """
HeartGuard AI is a predictive analytics tool designed to assess the risk of heart failure in patients based on clinical data.

This application uses a machine learning model trained on real-world clinical records to provide risk assessments.

Features:
• Modern, user-friendly interface
• Real-time input validation
• Visual representation of risk factors
• Detailed prediction results
• Dark/Light mode support

For more information or support, please contact our team.
"""
        
        ctk.CTkLabel(
            self.about_frame,
            text=about_text,
            font=ctk.CTkFont(size=14),
            justify="left",
            wraplength=600
        ).grid(row=1, column=0, padx=40, pady=20, sticky="nw")
        
        # Version info
        version_text = f"Version 1.0.0\n© {2025} HeartGuard AI"
        ctk.CTkLabel(
            self.about_frame,
            text=version_text,
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray50")
        ).grid(row=2, column=0, padx=20, pady=(0, 20), sticky="se")
    
    def change_appearance_mode(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)

if __name__ == "__main__":
    app = ModernHeartFailureApp()
    app.mainloop()
