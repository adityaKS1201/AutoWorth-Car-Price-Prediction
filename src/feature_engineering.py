import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()

    # -----------------------------
    # Feature 1: Mileage per year
    # -----------------------------
    df["mileage_per_year"] = df["km_driven"] / (df["vehicle_age"] + 1)

    # -----------------------------
    # Feature 2: Power-to-engine ratio
    # -----------------------------
    df["power_engine_ratio"] = df["max_power"] / df["engine"]

    # -----------------------------
    # Feature 3: Engine per seat ratio
    # -----------------------------
    df["engine_per_seat"] = df["engine"] / df["seats"]

    # -----------------------------
    # REMOVE target-dependent engineered features
    # -----------------------------
    # ❌ price_per_km
    # ❌ log_price

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill numeric NaNs
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df
