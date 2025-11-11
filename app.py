import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("💓 Heart Disease Prediction App")
st.write("Enter the details below to check whether a person has heart disease or not:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200)
chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

# Convert categorical input
sex = 1 if sex == "Male" else 0

if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("💔 The person has Heart Disease.")
    else:
        st.success("❤️ The person does NOT have Heart Disease.")
