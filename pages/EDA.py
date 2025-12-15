import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------
st.set_page_config(page_title="EDA – AutoWorth", layout="wide")

# -----------------------------------------------------
# Load dataset
# -----------------------------------------------------
df = pd.read_csv("data/cardekho_dataset.csv")

# -----------------------------------------------------
# GLOBAL GLASSMORPHISM CSS (same as homepage)
# -----------------------------------------------------
page_css = """
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

[data-testid="stHeader"] {
    background: rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
}

.card {
    background: rgba(255, 255, 255, 0.10);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.20);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.25);
    margin-bottom: 25px;
}

.big-title {
    font-size: 45px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

</style>
"""

st.markdown(page_css, unsafe_allow_html=True)

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------
st.markdown("<h1 class='big-title'>📊 Exploratory Data Analysis</h1>", unsafe_allow_html=True)

st.markdown("""
<div class="card">
Welcome to the EDA section. Below are key insights and visual patterns extracted from the CarDekho dataset used in AutoWorth.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# 1. Selling Price Distribution
# -----------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("💰 Selling Price Distribution")

fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(df["selling_price"], kde=True, color="#1E88E5", ax=ax)
ax.set_xlabel("Selling Price")
ax.set_ylabel("Count")
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# 2. Mileage vs Price Scatterplot
# -----------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📈 Mileage vs Selling Price")

fig, ax = plt.subplots(figsize=(8,4))
sns.scatterplot(x=df["mileage"], y=df["selling_price"], color="#039BE5", ax=ax)
ax.set_xlabel("Mileage")
ax.set_ylabel("Selling Price")
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# 3. Correlation Heatmap
# -----------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔥 Correlation Heatmap")

num_df = df.select_dtypes(include=["int64", "float64"])

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(num_df.corr(), annot=False, cmap="Blues", ax=ax)
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
