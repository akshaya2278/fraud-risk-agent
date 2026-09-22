import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score
import joblib
from data_prep import prepare_and_engineer_data

print("1. Loading and engineering data (this takes a minute)...")
df = prepare_and_engineer_data("creditcard.csv")

print("2. Preparing Train/Test splits...")
# Select only the features the model should learn from
exclude_cols = ["Class", "user_id", "merchant_category", "Time", "Amount"]
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
y = df["Class"]

# Split 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Training Supervised Classifier (XGBoost)...")
# We calculate a weight to handle the extreme imbalance between Normal and Fraud
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=42)
xgb.fit(X_train, y_train)

print("4. Training Unsupervised Anomaly Detector (Isolation Forest)...")
# Isolation forest only trains on normal transactions to learn standard behavior
X_train_normal = X_train[y_train == 0]
iso = IsolationForest(contamination=0.002, random_state=42)
iso.fit(X_train_normal)

print("5. Evaluating the Dual-Engine Risk Agent...")
# Get probabilities from XGBoost
xgb_probs = xgb.predict_proba(X_test)[:, 1]

# Get anomaly scores from Isolation Forest (and normalize them to 0-1)
iso_scores = iso.decision_function(X_test)
iso_probs = 1.0 / (1.0 + np.exp(iso_scores * 10))

# Combine them: 70% weight to XGBoost, 30% to Anomaly Detector
hybrid_scores = (0.7 * xgb_probs) + (0.3 * iso_probs)
hybrid_scores = np.clip(hybrid_scores, 0.0, 1.0)

# If final score is > 0.5, flag as Fraud
y_pred = (hybrid_scores > 0.5).astype(int)

print("\n" + "="*50)
print("MODEL EVALUATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
pr_auc = average_precision_score(y_test, hybrid_scores)
print(f"Precision-Recall AUC (Crucial for Imbalanced Data): {pr_auc:.4f}")
print("="*50)

print("\n6. Saving Models for the Dashboard...")
# Save the feature column names and trained models for the UI
joblib.dump(feature_cols, "feature_cols.pkl")
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(iso, "iso_model.pkl")

# Save a small chunk of test data to upload into our dashboard later
df.tail(2000).to_csv("dashboard_test_data.csv", index=False)
print("✅ Models and test data saved successfully!")