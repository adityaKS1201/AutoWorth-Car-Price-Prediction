import streamlit as st
import random

# -----------------------------------------------------
# GLOBAL GLASSMORPHISM + PREMIUM DASHBOARD CSS
# -----------------------------------------------------
page_css = """
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding-top: 20px;
}

[data-testid="stHeader"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(255,255,255,0.15);
}

[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.20);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.15);
}

/* Big Clickable Navigation Cards */
.nav-card {
    background: rgba(255, 255, 255, 0.12);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.25);
    text-align: center;
    cursor: pointer;
    transition: 0.3s ease;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.25);
    font-size: 25px;
    font-weight: 700;
    color: white;
}

.nav-card:hover {
    background: rgba(255, 255, 255, 0.18);
    transform: translateY(-7px);
    box-shadow: 0px 15px 40px rgba(0,0,0,0.35);
}

/* Title Gradient */
.big-title {
    font-size: 50px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.quote-box {
    font-size: 20px;
    padding: 15px;
    text-align:center;
    font-style: italic;
    margin-bottom: 25px;
}

.card {
    background: rgba(255, 255, 255, 0.10);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.25);
    margin-bottom: 25px;
}

</style>
"""
st.markdown(page_css, unsafe_allow_html=True)

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------
st.markdown("<h1 class='big-title'>AutoWorth – Car Price Prediction Dashboard</h1>", unsafe_allow_html=True)

# -----------------------------------------------------
# BANNER
# -----------------------------------------------------
st.image("images/banner.png", use_container_width=True)

# -----------------------------------------------------
# RANDOM QUOTE
# -----------------------------------------------------
quotes = [
    "“The cars we drive say a lot about us.” – Alexandra Paul",
    "“A dream without ambition is like a car without gas.”",
    "“The best way to predict the future is to create it.”",
    "“It's not about the car you drive, but how you drive it.”",
]
quote = random.choice(quotes)

st.markdown(f"""
<div class="card quote-box">
🚗 <i>{quote}</i>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# WELCOME CARD
# -----------------------------------------------------
st.markdown("""
<div class="card" style="text-align:center;">
    <h2 style='color:#4facfe;'>Welcome to AutoWorth</h2>
    <p style='font-size:18px;'>
        Analyze car data, explore machine learning insights, and predict used car prices 
        with our advanced AutoWorth ML engine.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# CLICKABLE NAVIGATION CARDS (CORRECT STREAMLIT PATHS)
# -----------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <a href="/EDA" target="_self">
        <div class='nav-card'>📊 EDA Dashboard</div>
    </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <a href="/Model" target="_self">
        <div class='nav-card'>🤖 Model Insights</div>
    </a>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <a href="/Predict" target="_self">
        <div class='nav-card'>💰 Predict Price</div>
    </a>
    """, unsafe_allow_html=True)

# -----------------------------------------------------
# END OF HOMEPAGE
# -----------------------------------------------------
