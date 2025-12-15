import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor

from feature_engineering import add_features
from preprocessing import clean_data

# ------------------------------------------
# Load dataset
# ------------------------------------------
df = pd.read_csv("data/cardekho_dataset.csv")

# Clean & engineer features
df = clean_data(df)
df = add_features(df)

# ------------------------------------------
# Define target & features (log transform)
# ------------------------------------------
y = np.log(df["selling_price"] + 1)   # LOG TARGET
X = df.drop(columns=["selling_price"])

# Categorical column indices
cat_cols = ["car_name", "brand", "model", "seller_type",
            "fuel_type", "transmission_type"]
cat_idx = [X.columns.get_loc(col) for col in cat_cols]

# ------------------------------------------
# Train-test split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------
# CatBoost Model
# ------------------------------------------
model = CatBoostRegressor(
    iterations=1200,
    depth=10,
    learning_rate=0.06,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

model.fit(
    X_train, y_train,
    cat_features=cat_idx,
    eval_set=(X_test, y_test)
)

# ------------------------------------------
# Predictions (inverse-log)
# ------------------------------------------
y_pred_log = model.predict(X_test)
y_pred = np.exp(y_pred_log) - 1     # Convert back to price

y_true = np.exp(y_test) - 1

# ------------------------------------------
# Metrics
# ------------------------------------------
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n================= CATBOOST + LOG TARGET =================\n")
print(f"🔹 RMSE: {rmse:,.2f}")
print(f"🔹 MAE : {mae:,.2f}")
print(f"🔹 R²  : {r2:.4f}")
print("\n==========================================================\n")

# ------------------------------------------
# Save final model
# ------------------------------------------
joblib.dump(model, "models/final_model.pkl")
print("Model saved → models/final_model.pkl")
