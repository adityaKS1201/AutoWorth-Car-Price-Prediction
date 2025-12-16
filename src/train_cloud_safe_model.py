import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import clean_data
from feature_engineering import add_features

# ------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------
df = pd.read_csv("data/cardekho_dataset.csv")

# ------------------------------------------------------
# 2. CLEAN & FEATURE ENGINEERING
# ------------------------------------------------------
df = clean_data(df)
df = add_features(df)

# ------------------------------------------------------
# 3. DEFINE FEATURES AND TARGET
# ------------------------------------------------------
y = df["selling_price"]
X = df.drop(columns=["selling_price"])

# CatBoost automatically handles categorical features
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# ------------------------------------------------------
# 4. TRAIN–TEST SPLIT (for evaluation)
# ------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ------------------------------------------------------
# 5. DEFINE MODEL (Cloud-Safe)
# ------------------------------------------------------
model = CatBoostRegressor(
    depth=8,
    iterations=1000,
    learning_rate=0.07,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

# ------------------------------------------------------
# 6. TRAIN MODEL
# ------------------------------------------------------
model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

# ------------------------------------------------------
# 7. MODEL EVALUATION
# ------------------------------------------------------
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("-----------------------------------------------------")
print("MODEL PERFORMANCE (CATBOOST)")
print("-----------------------------------------------------")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# ------------------------------------------------------
# 8. SAVE MODEL (CLOUD-SAFE FORMAT)
# ------------------------------------------------------
model.save_model("models/final_model.cbm")
print("\n🚀 Model saved successfully as: models/final_model.cbm")
print("This model is fully Streamlit-Cloud compatible!")
