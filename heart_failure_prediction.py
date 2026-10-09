from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "heart_failure_clinical_records_dataset.csv"

# Load the dataset
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
    df = pd.read_csv(url)
    df.to_csv(DATASET_PATH, index=False)

# Display basic information about the dataset
print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize the distribution of the target variable (DEATH_EVENT)
plt.figure(figsize=(8, 6))
sns.countplot(x="DEATH_EVENT", data=df)
plt.title("Distribution of Death Events")
plt.xlabel("Death Event (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.savefig(REPORTS_DIR / "death_event_distribution.png")
plt.close()

# Prepare features and target variable
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
print("\nModel Evaluation:")
print("-" * 50)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "feature_importance.png")
plt.close()

print("\nFeature Importance:")
print(feature_importance)

# Save the model and scaler
joblib.dump(rf_model, MODELS_DIR / "heart_failure_model.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

print("\nModel training complete. The model and scaler have been saved to disk.")
print(f"Model files: {MODELS_DIR}")
print(f"Visualization files: {REPORTS_DIR}")
