import streamlit as st
import pandas as pd
import sys, os
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# Allow imports from src/
sys.path.append(os.path.join(os.getcwd(), "src"))
from preprocessing import clean_data
from feature_engineering import add_features

st.set_page_config(page_title="Model Insights – AutoWorth", layout="wide")

st.markdown("<h1 style='color:#1E88E5;'>🤖 Model Insights</h1>", unsafe_allow_html=True)

# -------------------------------
# Load model
# -------------------------------
model = CatBoostRegressor()
model.load_model("models/final_model.cbm")

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("data/cardekho_dataset.csv")
df = clean_data(df)
df = add_features(df)

y = df["selling_price"]
X = df.drop(columns=["selling_price"])

# -------------------------------
# Feature Importance
# -------------------------------
st.subheader("🔥 Feature Importance")

importances = model.get_feature_importance()
feature_names = X.columns

imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.dataframe(imp_df.head(15))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp_df["Feature"].head(10), imp_df["Importance"].head(10))
ax.invert_yaxis()
ax.set_title("Top 10 Important Features")
st.pyplot(fig)
