import streamlit as st
import joblib
import numpy as np

# Load model and scaler
heart_model = joblib.load('heart_failure_model.pkl')
heart_scaler = joblib.load('scaler.pkl')

# ========== CUSTOM CSS & HTML STYLING ==========
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1e293b;
            background: linear-gradient(120deg, #f0f9ff 0%, #e0f2fe 50%, #c7d2fe 100%);
            background-size: 400% 400%;
            animation: gradientFlow 15s ease infinite;
        }

        @keyframes gradientFlow {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .title-container {
            text-align: center;
            margin-bottom: 40px;
        }

        .app-title {
            font-size: 42px;
            font-weight: 800;
            color: #1e3a8a;
            letter-spacing: 1px;
            text-shadow: 0px 2px 8px rgba(30,58,138,0.2);
        }

        .subtitle {
            font-size: 18px;
            color: #334155;
            margin-top: 5px;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.45);
            border-radius: 25px;
            border: 1px solid rgba(255,255,255,0.5);
            box-shadow: 0 12px 30px rgba(0,0,0,0.1);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 45px;
            margin: 0 auto 50px;
            width: 90%;
            max-width: 950px;
            transition: 0.4s ease;
        }

        .glass-card:hover {
            transform: scale(1.005);
            box-shadow: 0 15px 40px rgba(29,78,216,0.15);
        }

        .section-title {
            font-size: 22px;
            font-weight: 700;
            color: #1d4ed8;
            border-left: 5px solid #1d4ed8;
            padding-left: 10px;
            margin-bottom: 20px;
            margin-top: 25px;
        }

        .stButton button {
            background: linear-gradient(90deg, #2563eb, #1e40af);
            color: #fff;
            border: none;
            padding: 14px 32px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            letter-spacing: 0.3px;
        }

        .stButton button:hover {
            transform: translateY(-2px);
            background: linear-gradient(90deg, #1e3a8a, #1e40af);
            box-shadow: 0 6px 18px rgba(29,78,216,0.4);
        }

        .result-box {
            background: rgba(240, 249, 255, 0.9);
            border-left: 6px solid #2563eb;
            padding: 15px 20px;
            border-radius: 10px;
            margin-top: 25px;
            font-size: 17px;
            font-weight: 500;
            color: #1e3a8a;
        }

        img {
            display: block;
            margin: 20px auto 40px auto;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
    </style>
""", unsafe_allow_html=True)

# ========== PAGE HEADER ==========
st.markdown("""
    <div class="title-container">
        <h1 class="app-title">💙 Heart Failure Prediction Portal</h1>
        <p class="subtitle">AI-powered diagnostic tool to assess your risk of heart failure</p>
    </div>
""", unsafe_allow_html=True)

st.image(
    "https://images.ctfassets.net/ut7rzv8yehpf/1DhC3uX3EeKnjU02LWyTXH/9c82e6ae82662ed5903eafb40d888d90/8_Main_Types_of_Heart_Disease.jpg?w=1800&h=900&fl=progressive&q=50&fm=jpg",
    caption="Human Heart Anatomy",
    width=500,
    use_container_width=False
)

# ========== MAIN CARD ==========
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

# Personal Info
st.markdown('<div class="section-title">🧍 Personal Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age (Years)", 1, 120, 60)
    sex = st.selectbox("Sex", ["Female (0)", "Male (1)"])
    anaemia = st.radio("Anaemia", [0, 1], format_func=lambda x: "Yes" if x else "No")
    diabetes = st.radio("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
with col2:
    smoking = st.radio("Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")
    hbp = st.radio("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x else "No")
    time = st.number_input("Follow-up Time (Days)", 0, 300, 130)

# Clinical Parameters
st.markdown('<div class="section-title">🧬 Clinical Measurements</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    ef = st.number_input("Ejection Fraction (%)", 0, 100, 45)
    platelets = st.number_input("Platelets (kiloplatelets/mL)", 0.0, 1000000.0, 250000.0)
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", 0, 200, 138)
with col4:
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.0, 10.0, 1.1)
    creatinine = st.number_input("Creatinine Phosphokinase (mcg/L)", 0, 8000, 600)

# ========== PREDICT BUTTON ==========
predict = st.button("🔍 Predict Heart Failure Risk")

if predict:
    sex_val = 1 if "Male" in sex else 0
    heart_input = [[age, anaemia, creatinine, diabetes, ef, hbp,
                    platelets, serum_creatinine, serum_sodium, sex_val, smoking, time]]

    scaled = heart_scaler.transform(heart_input)
    result = heart_model.predict(scaled)

    if result[0] == 1:
        st.markdown('<div class="result-box">💔 <b>High Risk of Heart Failure</b> — Please consult a cardiologist immediately for detailed diagnosis.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box">❤️ <b>Low Risk of Heart Failure</b> — Maintain a healthy lifestyle and regular checkups.</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
