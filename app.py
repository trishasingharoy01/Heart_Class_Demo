import streamlit as st
import joblib
import numpy as np

# Load Heart Failure model and scaler
heart_model = joblib.load('heart_failure_model.pkl')
heart_scaler = joblib.load('scaler.pkl')

st.title("Heart Failure Prediction using ML")


st.image(
    "https://images.ctfassets.net/ut7rzv8yehpf/1DhC3uX3EeKnjU02LWyTXH/9c82e6ae82662ed5903eafb40d888d90/8_Main_Types_of_Heart_Disease.jpg?w=1800&h=900&fl=progressive&q=50&fm=jpg",
    caption="Human Heart Anatomy",
    width=250,  # smaller image size
    use_container_width=False
)

# Input fields
age = st.slider("Age", 1, 120, 60)
anaemia = st.selectbox("Anaemia", [0, 1])
creatinine = st.number_input("Creatinine Phosphokinase", 0, 8000)
diabetes = st.radio("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
ef = st.number_input("Ejection Fraction", 0, 100)
hbp = st.radio("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x else "No")
platelets = st.number_input("Platelets", 0.0, 1000000.0)
serum_creatinine = st.number_input("Serum Creatinine", 0.0, 10.0)
serum_sodium = st.number_input("Serum Sodium", 0, 200)
sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
smoking = st.selectbox("Smoking", [0, 1])
time = st.number_input("Follow-up time (in days)", 0, 300)

# Prediction button

if st.button("Heart Failure Test Result"):
    heart_input = [[age, anaemia, creatinine, diabetes, ef, hbp, platelets, serum_creatinine,
                        serum_sodium, sex, smoking, time]]
    scaled = heart_scaler.transform(heart_input)
    result = heart_model.predict(scaled)
    if result[0] == 1:
        st.success("High Risk of Heart Failure")
    else:
        st.success("Low Risk of Heart Failure")