# binary_logistic_regression.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    accuracy_score
)
import matplotlib.pyplot as plt

# 1. Load binary classification dataset (Breast Cancer)
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Train/test split and standardize features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Fit logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Evaluate with confusion matrix, precision, recall, ROC-AUC
y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 5. Tune threshold manually
custom_threshold = 0.3
y_pred_custom = (y_proba >= custom_threshold).astype(int)

print(f"\nAfter threshold tuning (Threshold = {custom_threshold}):")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_custom))
print(f"Precision: {precision_score(y_test, y_pred_custom):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred_custom):.2f}")
