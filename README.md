# Heart Failure Prediction Project

A machine learning project for predicting heart failure risk from clinical patient data. This project includes exploratory data analysis, model training, feature importance analysis, and a simple GUI-based risk prediction app.

## Project Overview

The project uses the publicly available heart failure clinical records dataset to train a binary classification model that predicts whether a patient is likely to experience a heart failure event.

The workflow includes:
- Loading and exploring the dataset
- Checking for missing data and feature distributions
- Training a Random Forest classifier
- Evaluating model performance with accuracy, precision, recall, and confusion matrix
- Saving the trained model and scaler
- Visualizing the most important features
- Providing a simple desktop prediction interface

## Dataset

The dataset is sourced from the UCI Machine Learning Repository and is stored locally in the `data/` folder after the first run.

## Main Files

- `heart_failure_prediction.py` — training, evaluation, and model export script
- `modern_heart_app.py` — main desktop application for predicting patient risk
- `requirements.txt` — required Python packages
- `data/` — dataset files
- `models/` — trained model and preprocessing scaler
- `reports/` — generated plots and visualizations
- `archive/` — older or duplicate experimental files

## Setup

1. Create a virtual environment (optional but recommended)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Model Training

```bash
python heart_failure_prediction.py
```

This script will:
- load the dataset
- train the Random Forest model
- print evaluation metrics
- save the trained model in `models/`
- save plots in `reports/`

## Run the GUI App

```bash
python modern_heart_app.py
```

This opens a desktop app where the user can enter patient information and receive a risk prediction.

## Model Performance

The current project version achieved approximately 83.3% test accuracy using a Random Forest classifier on the selected clinical dataset.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- joblib
- customtkinter
- Pillow

## Resume-Friendly Summary

> Developed a heart failure prediction system using clinical records to classify patient risk levels. Built a machine learning pipeline in Python with data preprocessing, exploratory analysis, model evaluation, and feature importance analysis. Also created an interactive GUI for real-time risk prediction.
