# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ------------------- Load & Inspect -------------------
data_path = r"C:\Users\SHIVAM AGARWAL\OneDrive\Downloads\creditcard.csv.zip"
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print("\nSample rows:")
print(df.sample(5))

# Quick check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Correlation heatmap for quick EDA
plt.figure(figsize=(12, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

# Distribution of target
plt.figure(figsize=(5,3))
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Non-Fraud Counts")
plt.show()

# ------------------- Feature / Target Split -------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale numerical features using RobustScaler (robust to outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ------------------- Simple Manual Undersampling -------------------
# Get majority & minority class
fraud_df = df[df["Class"] == 1]
nonfraud_df = df[df["Class"] == 0].sample(len(fraud_df) * 2, random_state=42)  # keep 2x more genuine than fraud

balanced_df = pd.concat([fraud_df, nonfraud_df], axis=0)
print("\nAfter balancing:")
print(balanced_df["Class"].value_counts())

X_balanced = balanced_df.drop("Class", axis=1)
y_balanced = balanced_df["Class"]

# Scale again after balancing
X_balanced_scaled = scaler.fit_transform(X_balanced)

# ------------------- Train/Test Split -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced_scaled, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
)

# ------------------- Train Models -------------------
# Logistic Regression
log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# ------------------- Evaluation Helper -------------------
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
    plt.title(f"{name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

# Evaluate both models
evaluate_model("Logistic Regression", y_test, log_preds)
evaluate_model("Random Forest", y_test, rf_preds)

# ------------------- Feature Importance (for RF) -------------------
feat_importances = rf_model.feature_importances_
sorted_idx = np.argsort(feat_importances)[-15:]  # top 15

plt.figure(figsize=(8,5))
plt.barh(range(len(sorted_idx)), feat_importances[sorted_idx], color="teal")
plt.yticks(range(len(sorted_idx)), np.array(X_balanced.columns)[sorted_idx])
plt.title("Top 15 Important Features (Random Forest)")
plt.show()
