
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Step 1: Load dataset from file path
# -----------------------------
file_path = r"C:\Users\Divya P\Downloads\breast-cancer.csv" # <-- update this path
df = pd.read_csv(file_path)

print("Shape of dataset:", df.shape)
print("Columns:", df.columns)

# -----------------------------
# Step 2: Prepare dataset
# -----------------------------
# Drop ID column if it exists
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Map diagnosis to numeric (M = malignant, B = benign)
if "diagnosis" in df.columns:
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop(columns=["diagnosis"])   # Features
y = df["diagnosis"]                  # Target

# -----------------------------
# Step 3: Train-test split & scaling
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Step 4: Train SVM with Linear Kernel
# -----------------------------
linear_svm = SVC(kernel="linear", C=1)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

print("\n--- Linear Kernel Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# -----------------------------
# Step 5: Train SVM with RBF Kernel
# -----------------------------
rbf_svm = SVC(kernel="rbf", C=1, gamma="scale")
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

print("\n--- RBF Kernel Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

# Confusion matrix (RBF)
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rbf), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix (RBF Kernel)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# -----------------------------
# Step 6: Hyperparameter tuning (RBF)
# -----------------------------
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
    "kernel": ["rbf"]
}
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5, verbose=1)
grid.fit(X_train, y_train)

print("\n--- Best Hyperparameters (RBF) ---")
print(grid.best_params_)

# Evaluate best model
best_svm = grid.best_estimator_
y_pred_best = best_svm.predict(X_test)
print("Tuned Model Accuracy:", accuracy_score(y_test,y_pred_best))