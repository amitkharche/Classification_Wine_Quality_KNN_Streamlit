import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report

# --- Page Config ---
st.set_page_config(page_title="Wine Quality Prediction with SHAP", layout="wide")
st.title("🍷 Wine Quality Prediction with KNN + SHAP + Dashboard")

# --- Load Model ---
model = joblib.load("models/knn_model.pkl")

# --- Feature Names ---
features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]

# --- Tabs ---
tabs = st.tabs(["🔢 Predict", "📊 Model Dashboard"])

# ---------------- Tab 1: Prediction ----------------
with tabs[0]:
    st.header("🔢 Predict Wine Quality")

    st.markdown("### 🧪 Enter Input Features")

    input_data = []
    cols = st.columns(3)  # Layout in 3 columns

    for idx, feat in enumerate(features):
        with cols[idx % 3]:
            val = st.number_input(f"{feat.title()}", value=5.0, step=0.1, format="%.2f")
            input_data.append(val)

    X_input = np.array(input_data).reshape(1, -1)

    if st.button("🔍 Predict"):
        pred = model.predict(X_input)
        st.success(f"✅ Predicted Wine Quality: **{int(pred[0])}**")

        # SHAP Explanation
        st.subheader("🧠 SHAP Explanation")
        df = pd.read_csv("data/winequality.csv")
        X_train = df.drop("quality", axis=1)

        explainer = shap.Explainer(model.predict, X_train, feature_names=features)
        shap_values = explainer(X_input)

        shap.initjs()
        fig, ax = plt.subplots(figsize=(6, 4))  # Smaller plot
        shap.plots.waterfall(shap_values[0], max_display=8, show=False)  # Fewer features
        st.pyplot(fig, clear_figure=True)

# ---------------- Tab 2: Dashboard ----------------
with tabs[1]:
    st.header("📊 Model Performance Dashboard")

    df = pd.read_csv("data/winequality.csv")
    X = df.drop("quality", axis=1)
    y_true = df["quality"]
    y_pred = model.predict(X)

    # Confusion Matrix
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))  # Smaller plot size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 12})  # Bigger text
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    st.pyplot(fig, clear_figure=True)

    # Classification Report
    st.subheader("📄 Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.style.format({
            "precision": "{:.2f}",
            "recall": "{:.2f}",
            "f1-score": "{:.2f}",
            "support": "{:.0f}"
        })
    )
