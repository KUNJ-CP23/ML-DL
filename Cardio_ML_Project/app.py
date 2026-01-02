import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Cardio Risk Predictor", page_icon="❤️")

st.title("❤️ Cardiovascular Disease Prediction")

# ---------- load model & scalers ----------
rf_model = pickle.load(open("models/rf_model.pkl","rb"))
scaler = pickle.load(open("models/scalar.pkl","rb"))
feature_columns = pickle.load(open("models/feature_columns.pkl","rb"))

# ---------- USER INPUT ----------
st.subheader("🧍 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age_years = st.number_input("Age (years)", 1, 120, 40)
    height = st.number_input("Height (cm)", 120, 220, 165)
    weight = st.number_input("Weight (kg)", 30, 200, 70)
    gender = st.selectbox("Gender", ["Female","Male"])

with col2:
    ap_hi = st.number_input("Systolic BP", 80, 250, 120)
    ap_lo = st.number_input("Diastolic BP", 50, 200, 80)
    smoke = st.selectbox("Smoking", ["No","Yes"])
    alco = st.selectbox("Alcohol use", ["No","Yes"])

chol = st.selectbox("Cholesterol",
                    ["Normal","Above Normal","Well Above Normal"])

gluc = st.selectbox("Glucose",
                    ["Normal","Above Normal","Well Above Normal"])

# ---------- convert to numeric like dataset ----------
gender = 2 if gender=="Male" else 1
smoke = 1 if smoke=="Yes" else 0
alco = 1 if alco=="Yes" else 0

map3 = {"Normal":1,"Above Normal":2,"Well Above Normal":3}
chol = map3[chol]
gluc = map3[gluc]

# ---------- create dataframe EXACTLY like training ----------
row = pd.DataFrame([{
    "age_years": age_years,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "gender": gender,
    "cholesterol": chol,
    "gluc": gluc,
    "smoke": smoke,
    "alco": alco,
    "active": 1
}])

# ---------- one-hot encode ----------
row = pd.get_dummies(row)

# ---------- add missing columns ----------
for col in feature_columns:
    if col not in row.columns:
        row[col] = 0

# ---------- reorder ----------
row = row[feature_columns]

# ---------- scale numeric ----------
num_cols = ['age_years','height','weight','ap_hi','ap_lo']
row[num_cols] = scaler.transform(row[num_cols])

# ---------- predict ----------
if st.button("🔍 Predict Risk"):
    prob = rf_model.predict_proba(row)[0][1]
    percent = round(prob*100,2)

    st.subheader(f"🩺 Risk Probability: **{percent}%**")
    st.progress(int(percent))

    if percent >= 60:
        st.error("⚠ High Risk")
    elif percent >= 30:
        st.warning("🟡 Moderate Risk")
    else:
        st.success("🟢 Low Risk")

    # ---------- mini dashboard ----------
    st.markdown("### 📊 Health Summary")

    chart = pd.DataFrame({
        "Indicator":["Age","Systolic BP","Diastolic BP","Cholesterol","Glucose"],
        "Value":[age_years,ap_hi,ap_lo,chol,gluc]
    }).set_index("Indicator")

    st.bar_chart(chart)

    st.markdown("### 🥧 Lifestyle Profile")
    life = pd.DataFrame({
        "Lifestyle":["Smoking","Alcohol","Non-Alcohol"],
        "Score":[smoke,alco,1-alco]
    }).set_index("Lifestyle")
    st.area_chart(life)