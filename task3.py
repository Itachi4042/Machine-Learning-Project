# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ------------------- Load Dataset -------------------
titanic_path = r"C:\Users\SHIVAM AGARWAL\OneDrive\Downloads\Titanic-Dataset.csv"
df = pd.read_csv(titanic_path)

print(f"Shape of dataset: {df.shape}")
print("\nSample rows:")
print(df.sample(5))

# ------------------- Basic Cleaning -------------------
# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with most common value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing) if it exists
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

# Drop PassengerId, Name, Ticket as they are not predictive
drop_cols = [col for col in ['PassengerId', 'Name', 'Ticket'] if col in df.columns]
df.drop(columns=drop_cols, inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ------------------- EDA -------------------
# Survival countplot
plt.figure(figsize=(5,3))
sns.countplot(x='Survived', data=df)
plt.title("Survival Distribution")
plt.show()

# Survival by Sex
plt.figure(figsize=(5,3))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# Survival by Pclass
plt.figure(figsize=(5,3))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Correlation heatmap ONLY for numeric columns
plt.figure(figsize=(8,5))
numeric_df = df.select_dtypes(include=[np.number])  # only numeric cols
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Titanic Numeric Features)")
plt.show()

# ------------------- Encode Categorical -------------------
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# ------------------- Split Features/Target -------------------
X = df_encoded.drop('Survived', axis=1)
y = df_encoded['Survived']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------- Train Models -------------------
# Logistic Regression
log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
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
    print(classification_report(y_true, y_pred))

    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Died", "Survived"],
                yticklabels=["Died", "Survived"])
    plt.title(f"{name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

# Evaluate both models
evaluate_model("Logistic Regression", y_test, log_preds)
evaluate_model("Random Forest", y_test, rf_preds)

# ------------------- Feature Importance (for RF) -------------------
feat_importances = rf_model.feature_importances_
sorted_idx = np.argsort(feat_importances)

plt.figure(figsize=(8,5))
plt.barh(range(len(sorted_idx)), feat_importances[sorted_idx], color="teal")
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.title("Feature Importance (Random Forest)")
plt.show()
