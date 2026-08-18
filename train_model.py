import pandas as pd
import joblib
import numpy as np
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save test data for Streamlit app
test_data = X_test.copy()
test_data["target"] = y_test
test_data.to_csv("test_data.csv", index=False)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Create model directory
os.makedirs("model", exist_ok=True)

# Dictionary to store metrics
results = []

print("TRAINING MODELS AND CALCULATING METRICS")

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save model with simple filename
    model_filename = name.lower().replace(" ", "_")
    if model_filename == "logistic_regression":
        model_filename = "logistic"
    elif model_filename == "decision_tree":
        model_filename = "decision_tree"
    elif model_filename == "random_forest":
        model_filename = "random_forest"
    elif model_filename == "naive_bayes":
        model_filename = "naive_bayes"
    
    joblib.dump(model, f"model/{model_filename}.pkl")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = roc_auc_score(y_test, y_pred)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Store results
    results.append({
        "ML Model Name": name,
        "Accuracy": f"{accuracy:.4f}",
        "AUC": f"{auc:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1": f"{f1:.4f}",
        "MCC": f"{mcc:.4f}"
    })
    
    print(f"✓ {name} trained successfully!\n")

# Create results DataFrame
results_df = pd.DataFrame(results)

print("MODEL PERFORMANCE COMPARISON")

print("| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |")
print("|---------------|----------|-----|-----------|--------|-----|-----|")
for _, row in results_df.iterrows():
    print(f"| **{row['ML Model Name']}** | {row['Accuracy']} | {row['AUC']} | {row['Precision']} | {row['Recall']} | {row['F1']} | {row['MCC']} |")

print("Models saved successfully in 'model/' directory!")