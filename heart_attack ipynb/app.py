# IMPORTANT:
# If your current app works ONLY with the numeric-column scaler fix,
# replace:
#     scaled_input = scaler.transform(input_df)
#     prediction = model.predict(scaled_input)[0]
#
# with:
#
# numeric_col = [
#     "Age", "RestingBP", "Cholesterol",
#     "FastingBS", "MaxHR", "Oldpeak"
# ]
#
# input_df[numeric_col] = scaler.transform(
#     input_df[numeric_col]
# )
#
# prediction = model.predict(input_df)[0]
#
# before deploying.

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)



BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "KNN_model.joblib")
scaler = joblib.load(BASE_DIR / "scaler.joblib")
expected_columns = joblib.load(BASE_DIR / "columns.joblib")



st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)



st.markdown("""
<h1 style='text-align:center;color:#FF4B4B;'>
❤️ Heart Disease Risk Predictor
</h1>

<h4 style='text-align:center;'>
AI Powered Heart Health Assessment
</h4>
""", unsafe_allow_html=True)

st.divider()


with st.sidebar:

    st.header("ℹ️ About")

    st.info("""
This application predicts heart disease risk using Machine Learning.

⚠️ Educational purposes only.
Please consult a healthcare professional.
""")

    st.markdown("---")

    st.markdown("""
### Features

✅ Heart Disease Prediction

✅ Machine Learning Model

✅ Fast Results

✅ Interactive Dashboard
""")



col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        18,
        100,
        40
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        80,
        200,
        120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        100,
        600,
        200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

with col2:

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Max Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

st.divider()



if st.button("🔍 Predict Heart Disease Risk"):

    input_data = {col: 0 for col in expected_columns}

    # Numerical Features
    input_data["Age"] = age
    input_data["RestingBP"] = resting_bp
    input_data["Cholesterol"] = cholesterol
    input_data["FastingBS"] = fasting_bs
    input_data["MaxHR"] = max_hr
    input_data["Oldpeak"] = oldpeak

    # Encoded Features
    if f"Sex_{sex}" in input_data:
        input_data[f"Sex_{sex}"] = 1

    if f"ChestPainType_{chest_pain}" in input_data:
        input_data[f"ChestPainType_{chest_pain}"] = 1

    if f"RestingECG_{resting_ecg}" in input_data:
        input_data[f"RestingECG_{resting_ecg}"] = 1

    if f"ExerciseAngina_{exercise_angina}" in input_data:
        input_data[f"ExerciseAngina_{exercise_angina}"] = 1

    if f"ST_Slope_{st_slope}" in input_data:
        input_data[f"ST_Slope_{st_slope}"] = 1

    input_df = pd.DataFrame([input_data])

    try:

        # Prediction
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        st.divider()

        if prediction == 1:

            st.error("⚠️ High Risk of Heart Disease")

            st.markdown("""
### Recommendations

• Consult a cardiologist

• Exercise regularly

• Reduce cholesterol intake

• Monitor blood pressure

• Maintain a healthy diet
""")

        else:

            st.success("✅ Low Risk of Heart Disease")

            st.balloons()

            st.markdown("""
### Great News

• Continue healthy eating

• Stay physically active

• Maintain regular checkups

• Keep monitoring your health
""")

    except Exception as e:

        st.error(f"Error: {e}")

        st.write("Expected Columns:")
        st.write(expected_columns)

        st.write("Input Columns:")
        st.write(input_df.columns.tolist())