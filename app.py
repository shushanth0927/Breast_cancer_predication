import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.title("🎗️ Breast Cancer Prediction App")

st.sidebar.header("Configuration")

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload CSV file with test data",
    type=["csv"],
    help="Upload a CSV file containing breast cancer features"
)

# Model selection
model_name = st.sidebar.selectbox(
    "🤖 Choose Classification Model",
    [
        "logistic",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest"
    ],
    format_func=lambda x: {
        "logistic": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "knn": "K-Nearest Neighbors",
        "naive_bayes": "Naive Bayes",
        "random_forest": "Random Forest"
    }[x]
)

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
    
    # Show data preview
    with st.expander("📊 View Data Preview"):
        st.dataframe(data.head())
    
    # Separate features and target
    X = data.drop("target", axis=1)
    y = data["target"]
    
    # Load model
    try:
        model = joblib.load(f"model/{model_name}.pkl")
        st.success(f"✅ Model loaded: **{model_name}**")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # Make predictions
    predictions = model.predict(X)
    
    # Get probability scores if available
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(X)[:, 1]
    else:
        prediction_proba = predictions
    
    # Calculate metrics
    st.header("📈 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accuracy = accuracy_score(y, predictions)
        st.metric("Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
    
    with col2:
        try:
            auc = roc_auc_score(y, prediction_proba)
            st.metric("AUC Score", f"{auc:.4f}", f"{auc*100:.2f}%")
        except:
            st.metric("AUC Score", "N/A")
    
    with col3:
        precision = precision_score(y, predictions)
        st.metric("Precision", f"{precision:.4f}", f"{precision*100:.2f}%")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        recall = recall_score(y, predictions)
        st.metric("Recall", f"{recall:.4f}", f"{recall*100:.2f}%")
    
    with col5:
        f1 = f1_score(y, predictions)
        st.metric("F1 Score", f"{f1:.4f}", f"{f1*100:.2f}%")
    
    with col6:
        mcc = matthews_corrcoef(y, predictions)
        st.metric("MCC", f"{mcc:.4f}")
    
    # Confusion Matrix
    st.header("🔢 Confusion Matrix")
    
    cm = confusion_matrix(y, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Malignant (0)', 'Benign (1)'],
                yticklabels=['Malignant (0)', 'Benign (1)'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    
    st.pyplot(fig)
    
    # Classification Report
    st.header("📋 Classification Report")
    
    report = classification_report(y, predictions, 
                                   target_names=['Malignant', 'Benign'],
                                   output_dict=True)
    
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0))
    
    # Prediction Distribution
    st.header("📊 Prediction Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual Distribution")
        actual_counts = pd.Series(y).value_counts()
        st.bar_chart(actual_counts)
    
    with col2:
        st.subheader("Predicted Distribution")
        pred_counts = pd.Series(predictions).value_counts()
        st.bar_chart(pred_counts)

else:
    st.info("👆 Please upload a CSV file to begin prediction")
