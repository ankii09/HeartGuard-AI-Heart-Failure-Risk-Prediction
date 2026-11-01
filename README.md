# Heart Failure Prediction Project

A machine learning project for exploring heart failure clinical records with a Vercel-ready web dashboard and a desktop GUI. This is an educational ML demo, is not clinically validated, and must not be used for diagnosis or treatment.

## Project Overview

The project uses the publicly available heart failure clinical records dataset to train a binary classification model that predicts whether a patient is likely to experience a heart failure event.

The workflow includes:
- Loading and exploring the dataset
- Checking for missing data and feature distributions
- Training a Random Forest classifier
- Evaluating model performance with accuracy, precision, recall, and confusion matrix
- Saving the trained model and scaler
- Visualizing the most important features
- Providing a Vercel web dashboard and a desktop prediction interface

## Dataset

The dataset is sourced from the UCI Machine Learning Repository and is stored locally in the `data/` folder after the first run.

## Main Files

- `heart_failure_prediction.py` — training, evaluation, and model export script
- `modern_heart_app.py` — main desktop application for predicting patient risk
- `index.html`, `styles.css`, `app.js` — static Vercel dashboard
- `api/predict.py` — Python serverless prediction and model-information endpoint
- `vercel.json` — Vercel function configuration and security headers
- `requirements.txt` — pinned dependencies for the Vercel Python function
- `requirements-desktop.txt` — additional desktop GUI and plotting dependencies
- `data/` — dataset files
- `models/` — trained model and preprocessing scaler
- `reports/` — generated plots and visualizations
- `archive/` — older or duplicate experimental files

## Setup

1. Create a virtual environment (optional but recommended)
2. Install dependencies for the desktop application:

```bash
pip install -r requirements-desktop.txt
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

## Deploy the Web Dashboard to Vercel

The repository root is the Vercel project root. Import the Git repository into Vercel and deploy with the default build settings; the static dashboard is served from `index.html`, and `/api/predict` is handled by the Python serverless function. Keep the `models/` artifacts and `models/model_metadata.json` in the deployed repository.

For local development, install the packages from `requirements.txt` and run the app with the Vercel CLI:

```bash
npm install --global vercel
vercel dev
```

The web dashboard validates values against the observed input ranges in the training dataset. It reports the dataset's `DEATH_EVENT` target class, model output probabilities, threshold, held-out test metrics, validation model comparison, and one-feature-at-a-time sensitivity. Local sensitivity is not a causal explanation.

### Vercel deployment notes

- The serialized estimator was trained with scikit-learn 1.5.2; the web requirements pin compatible versions.
- The API loads `models/heart_failure_pipeline.pkl`, or composes the existing model/scaler artifacts if the pipeline file is absent.
- Commit the web files, the API function, model artifacts, and metadata before deploying from Git.
- The Python function is subject to Vercel's Python runtime and function-size limits. This project's model artifact is small; verify the deployment log after publishing.
- This application is not clinically validated and must not be used for diagnosis or treatment.

## Model Performance

Current recorded held-out test metrics are stored in `models/model_metadata.json`. Metrics are specific to the recorded train/validation/test split and are not guarantees of future performance.

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
