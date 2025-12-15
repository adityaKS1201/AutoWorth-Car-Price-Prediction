import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# -----------------------------------------
# Fix import paths
# -----------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

import joblib
from feature_engineering import add_features
from preprocessing import clean_data

# -----------------------------------------
# Load model & data
# -----------------------------------------
model = joblib.load("models/final_model.pkl")
df = pd.read_csv("data/cardekho_dataset.csv")

# -----------------------------------------
# Page setup & title
# -----------------------------------------
st.set_page_config(page_title="Model Insights – AutoWorth", layout="wide")

st.markdown("<h1 style='color:#1E88E5;'>🤖 Machine Learning Model Insights</h1>", 
            unsafe_allow_html=True)

st.write("Explore how AutoWorth's ML model makes predictions and what features influence price the most.")

st.write("---")

# -------------------------------------------------
# MODEL PERFORMANCE (Static Explanation)
# -------------------------------------------------
st.subheader("📈 Model Performance (Based on Training)")

# st.info("""
# The model was trained using **CatBoostRegressor** with a **log-transformed target**, 
# which greatly improves prediction accuracy for cheaper and expensive cars.
# """)

col1, col2, col3 = st.columns(3)

col1.metric("R² Score", "0.947+", "Excellent")
col2.metric("RMSE", "≈ ₹1.9 lakh", "- Lower is better")
col3.metric("MAE", "≈ ₹93,000", "- Average deviation")

st.write("---")

# -------------------------------------------------
# FEATURE IMPORTANCE — CATBOOST
# -------------------------------------------------
st.subheader("🔥 Feature Importance (CatBoost)")

try:
    # Feature list (must match training order)
    feature_names = [
        "car_name", "brand", "model", "vehicle_age", "km_driven",
        "seller_type", "fuel_type", "transmission_type",
        "mileage", "engine", "max_power", "seats",
        "mileage_per_year", "power_engine_ratio", "engine_per_seat"
    ]

    # Get raw importance values
    importances = model.get_feature_importance()

    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(feat_imp_df.style.highlight_max(color="lightgreen"), use_container_width=True)

except Exception as e:
    st.error(f"Feature importance could not be generated: {e}")

st.write("---")

# -------------------------------------------------
# SHAP VALUES (GLOBAL EXPLAINABILITY)
# -------------------------------------------------
st.subheader("🧠 SHAP Explainability – How the Model Thinks")

st.write("SHAP shows how each feature contributes to increasing or decreasing the predicted price.")

try:
    # Sample smaller dataframe for SHAP
    sample_df = df.sample(200, random_state=42)

    # Apply cleaning & features
    sample_df = clean_data(sample_df)
    sample_df = add_features(sample_df)

    # Remove target
    X_sample = sample_df.drop(columns=["selling_price"])

    # Build SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Predict SHAP values
    shap_values = explainer.shap_values(X_sample)

    st.write("### 📊 SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig)

except Exception as e:
    st.error(f"SHAP could not be generated: {e}")
