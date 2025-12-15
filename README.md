# 🚗 AutoWorth – Used Car Price Prediction System

AutoWorth is a Machine Learning powered web application that predicts the selling price of used cars based on multiple car attributes such as brand, model, age, mileage, engine capacity, power, and more.

This project uses **CatBoost Regressor**, advanced **feature engineering**, and a beautiful **Streamlit UI** to deliver an accurate and interactive price prediction experience.

---

## 📌 Features

### 🔹 Machine Learning
- CatBoost Regressor for high accuracy  
- Log-transformed pricing model for stable predictions  
- Automatic feature engineering  
- Handling of categorical and numerical features  

### 🔹 Streamlit Web App
- Attractive glassmorphism UI  
- Dynamic dropdown filters (Brand → Car Name → Model)  
- Real-time prediction  
- EDA dashboard  
- Model insights page with:
  - Feature importance  
  - SHAP explainability  
  - Model performance metrics  

### 🔹 Dataset Includes
- Car name  
- Brand & model  
- Fuel type  
- Transmission type  
- Seller type  
- Mileage  
- Engine CC  
- Max power  
- Seating capacity  
- Kilometers driven  
- Selling price  

---

## 🧠 ML Workflow

1. Data Cleaning  
2. Feature Engineering  
   - Mileage per year  
   - Power-to-engine ratio  
   - Engine-per-seat ratio  
3. Train/test split  
4. CatBoost training with log-target  
5. Evaluation (RMSE, MAE, R²)  
6. Saving model  
7. Deploying with Streamlit  

---

## 📊 Model Performance

| Metric | Score |
|--------|--------|
| **RMSE** | ~198,000 |
| **MAE** | ~93,500 |
| **R² Score** | ~0.947 |

The model performs very well despite large variation in car price ranges (₹50K to ₹40L).

---

## 🗂 Project Structure

AutoWorth/
│── app.py
│── pages/
│ ├── EDA.py
│ ├── Model.py
│ ├── Predict.py
│── src/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│── data/
│ └── cardekho_dataset.csv
│── models/
│ └── final_model.pkl
│── images/
│ └── banner.png
│── requirements.txt
│── README.md


---

## 🚀 How to Run Locally

### 1️⃣ Install Dependencies

### pip install -r requirements.txt

### 2️⃣ Run Streamlit App


Your app will open in the browser.

---

## 🔥 Future Improvements
- Deployment on Streamlit Cloud  
- Add model comparison (RF, XGBoost, LightGBM)  
- Add variant & location-based price adjustments  
- Add user login system  
- Add data visualizations for each brand  

---

## 👨‍💻 Author
**Aditya Singh**  
Used Car Price Prediction Project (AutoWorth)  
Built using Python, ML, CatBoost, and Streamlit.

---

## ⭐ If you find this project helpful, please ⭐ star the repository!
