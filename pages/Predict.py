import sys, os
sys.path.append(os.path.join(os.getcwd(), "src"))


import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="AutoWorth – Predict Price", layout="wide")

# Load CatBoost model (trained on log-price)
model = joblib.load("models/final_model.pkl")

# Load dataset for dropdown filtering
df = pd.read_csv("data/cardekho_dataset.csv")

# Extract unique dropdown lists
brands = sorted(df["brand"].unique())

st.markdown("<h2 style='color:#1E88E5;'>🔧 Enter Car Details for Price Prediction</h2>", unsafe_allow_html=True)

# ----------------------------
# Dynamic Dropdown Filtering
# ----------------------------

# 1️⃣ Brand selection
brand = st.selectbox("Select Car Brand", brands)

# Car names filtered by brand
filtered_names = sorted(df[df["brand"] == brand]["car_name"].unique())
car_name = st.selectbox("Car Name", filtered_names)

# Models filtered by brand and car_name
filtered_models = sorted(df[(df["brand"] == brand) & (df["car_name"] == car_name)]["model"].unique())
model_name = st.selectbox("Model", filtered_models)

# ----------------------------
# Remaining Inputs
# ----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    fuel = st.selectbox("Fuel Type", sorted(df["fuel_type"].unique()))
    seller = st.selectbox("Seller Type", sorted(df["seller_type"].unique()))
    seats = st.number_input("Seats", min_value=2, max_value=10, value=5)

with col2:
    transmission = st.selectbox("Transmission Type", sorted(df["transmission_type"].unique()))
    mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, value=18.0)
    engine = st.number_input("Engine CC", min_value=500, max_value=5000, value=1200)

with col3:
    vehicle_age = st.slider("Vehicle Age (Years)", 0, 25, 5)
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
    max_power = st.number_input("Max Power (bhp)", min_value=20, max_value=500, value=80)

# ----------------------------
# Predict Button
# ----------------------------
if st.button("🚗 Predict Price"):
    
    # Valid engineered features
    mileage_per_year = km_driven / (vehicle_age + 1)
    power_engine_ratio = max_power / engine
    engine_per_seat = engine / seats

    # Build the input dataframe
    input_data = pd.DataFrame([{
        "car_name": car_name,
        "brand": brand,
        "model": model_name,
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "seller_type": seller,
        "fuel_type": fuel,
        "transmission_type": transmission,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "mileage_per_year": mileage_per_year,
        "power_engine_ratio": power_engine_ratio,
        "engine_per_seat": engine_per_seat
    }])

    # Predict (model outputs log(price))
    log_pred = model.predict(input_data)[0]

    # Convert log(price) → actual price
    prediction = np.exp(log_pred) - 1

    st.success(f"Estimated Selling Price: ₹ {prediction:,.2f}")
