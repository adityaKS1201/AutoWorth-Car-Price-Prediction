import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="AutoWorth – Predict Price", layout="wide")

# -------------------------------------------------
# LOAD MODEL (CATBOOST SAFE)
# -------------------------------------------------
model = CatBoostRegressor()
model.load_model("models/final_model.cbm")

# Feature order from training (CRITICAL)
model_features = model.feature_names_

# Load dataset for filtering
df = pd.read_csv("data/cardekho_dataset.csv")

st.markdown("<h2 style='color:#1E88E5;'>💰 Predict Used Car Price</h2>", unsafe_allow_html=True)

# -------------------------------------------------
# FILTERING LOGIC (DO NOT REMOVE)
# -------------------------------------------------
brands = sorted(df["brand"].unique())

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("Brand", brands)

    car_names = sorted(
        df[df["brand"] == brand]["car_name"].unique()
    )
    car_name = st.selectbox("Car Name", car_names)

    models = sorted(
        df[
            (df["brand"] == brand) &
            (df["car_name"] == car_name)
        ]["model"].unique()
    )
    model_name = st.selectbox("Model", models)

with col2:
    fuel = st.selectbox("Fuel Type", sorted(df["fuel_type"].unique()))
    transmission = st.selectbox("Transmission Type", sorted(df["transmission_type"].unique()))
    seller = st.selectbox("Seller Type", sorted(df["seller_type"].unique()))
    seats = st.number_input("Seats", min_value=2, max_value=10, value=5)

with col3:
    # Registration year instead of manual age
    current_year = datetime.now().year
    registration_year = st.selectbox(
        "Registration Year",
        list(range(current_year, current_year - 25, -1))
    )
    vehicle_age = current_year - registration_year
    st.caption(f"🚘 Vehicle Age: {vehicle_age} years")

    km_driven = st.number_input("Kilometers Driven", min_value=0, value=30000)
    mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, value=18.0)
    engine = st.number_input("Engine CC", min_value=500, max_value=5000, value=900)
    max_power = st.number_input("Max Power (bhp)", min_value=20, max_value=500, value=70)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🚗 Predict Price"):

    # ---------- FEATURE ENGINEERING ----------
    mileage_per_year = km_driven / (vehicle_age + 1)
    power_engine_ratio = max_power / engine
    engine_per_seat = engine / seats

    input_df = pd.DataFrame([{
        "car_name": car_name,
        "brand": brand,
        "model": model_name,
        "seller_type": seller,
        "fuel_type": fuel,
        "transmission_type": transmission,
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "mileage_per_year": mileage_per_year,
        "power_engine_ratio": power_engine_ratio,
        "engine_per_seat": engine_per_seat
    }])

    # Ensure exact training feature order (CRITICAL FIX)
    input_df = input_df[model_features]

    # ---------- CATBOOST PREDICTION ----------
    prediction = model.predict(input_df)[0]

    # -------------------------------------------------
    # REAL-WORLD DEPRECIATION (LOGIC-BASED)
    # -------------------------------------------------
    if vehicle_age >= 12:
        prediction *= 0.60
    elif vehicle_age >= 10:
        prediction *= 0.70
    elif vehicle_age >= 8:
        prediction *= 0.80

    # -------------------------------------------------
    # SEGMENT & ENGINE SANITY CAPS (NOT RANDOM)
    # -------------------------------------------------
    entry_level_cars = [
        "Alto", "Alto 800", "Wagon R", "Celerio",
        "Eon", "RediGO", "Kwid"
    ]

    if car_name in entry_level_cars:
        prediction = min(prediction, 300000)

    if engine <= 1000:
        prediction = min(prediction, 350000)

    if brand in ["Maruti", "Hyundai", "Tata"] and vehicle_age >= 7:
        prediction = min(prediction, 400000)

    prediction = max(0, prediction)

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    st.success(f"💸 Estimated Selling Price: ₹ {prediction:,.2f}")
