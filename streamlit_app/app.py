
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Wine Quality Prediction with SHAP", layout="wide")
st.title("🍷 Wine Quality Prediction with KNN + SHAP + Dashboard")

model = joblib.load("models/knn_model.pkl")

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

tabs = st.tabs(["🔢 Predict", "📊 Model Dashboard"])

# Tab 1: Prediction
with tabs[0]:
    st.header("Predict Wine Quality")
    input_data = []
    for feat in features:
        val = st.number_input(f"{feat.title()}", value=5.0, step=0.1, format="%.2f")
        input_data.append(val)

    X_input = np.array(input_data).reshape(1, -1)

    if st.button("Predict"):
        pred = model.predict(X_input)
        st.success(f"Predicted Wine Quality: {int(pred[0])}")

        # SHAP Explanation
        st.subheader("🧠 SHAP Explanation")
        df = pd.read_csv("data/winequality.csv")
        X_train = df.drop("quality", axis=1)
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_input)

        shap.initjs()
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=11, show=False)
        st.pyplot(fig)

# Tab 2: Performance Dashboard
with tabs[1]:
    st.header("Model Performance Dashboard")

    df = pd.read_csv("data/winequality.csv")
    X = df.drop("quality", axis=1)
    y_true = df["quality"]
    y_pred = model.predict(X)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Classification Report
    st.subheader("📄 Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))
