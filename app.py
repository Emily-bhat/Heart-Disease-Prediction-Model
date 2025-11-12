import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('heart_disease_model.pkl')


# PAGE CONFIGURATION

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")


# STYLING 

page_style = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: white;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
h1 {
    color: #b30000 !important;
    text-align: center;
    font-weight: 800;
}
label, .stSelectbox label, .stNumberInput label {
    color: #b30000 !important;
    font-weight: 600 !important;
}
.stButton>button {
    background-color: #b30000;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #d60000;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)


# TITLE

st.markdown("<h1>Heart Disease Prediction Model</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #b30000; font-size: 17px;'>"
    "Enter the details below to check whether a person has heart disease or not."
    "</p>", unsafe_allow_html=True)
st.write("---")


# INPUT SECTION (Single Column)

age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])


# CONVERT CATEGORICAL INPUTS

sex = 1 if sex == "Male" else 0


# PREDICTION

if st.button("🔍 Predict Heart Disease"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("The person has Heart Disease.")
    else:
        st.success("The person does NOT have Heart Disease.")

st.write("---")
st.caption("Developed using Python, Logistic Regression, and Streamlit")
