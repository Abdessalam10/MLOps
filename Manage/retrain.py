import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REAL_DATA_PATH = os.path.join(
    BASE_DIR,
    '../datasets/earthquake_alert_balanced_dataset.csv'
)

AUGMENTED_DATA_PATH = os.path.join(
    BASE_DIR,
    '../datasets/augmented_earthquake_data.csv'
)

# =========================
# Load datasets
# =========================
real_df = pd.read_csv(REAL_DATA_PATH)
augmented_df = pd.read_csv(AUGMENTED_DATA_PATH)

# =========================
# Normalize column names
# =========================
real_df.columns = [col.strip().lower() for col in real_df.columns]
augmented_df.columns = [col.strip().lower() for col in augmented_df.columns]

# =========================
# Keep only common columns
# =========================
common_cols = [c for c in augmented_df.columns if c in real_df.columns]

real_df = real_df[common_cols]
augmented_df = augmented_df[common_cols]

# =========================
# Merge datasets
# =========================
merged_df = pd.concat([real_df, augmented_df], ignore_index=True)

print("Merged dataset shape:", merged_df.shape)

# =========================
# Features and target
# =========================
X = merged_df[['magnitude', 'depth', 'cdi', 'mmi', 'sig']]
y = merged_df['alert']

# =========================
# Encode labels properly
# =========================
# Convert everything to string first
y = y.astype(str)

le = LabelEncoder()
y = le.fit_transform(y)

# =========================
# Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Train model
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=8
)

model.fit(X_train, y_train)

# =========================
# Predictions
# =========================
y_pred = model.predict(X_test)

# =========================
# Evaluation
# =========================
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# =========================
# Save model
# =========================
joblib.dump(model, 'earthquake_alert_model_retrained.pkl')

# Save encoder too
joblib.dump(le, 'label_encoder.pkl')

print("\nModel saved successfully.")