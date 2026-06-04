import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("KNN_model.joblib")
scaler = joblib.load("scaler.joblib")
expected_columns = joblib.load("columns.joblib")

# Title
st.title("Heart Disease Prediction by supremeinferno 😳")
st.markdown("Provide the following details to check your heart disease risk.")

# Inputs
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


if st.button("Predict"):

    # Initialize all expected columns to zero
    input_data = {col: 0 for col in expected_columns}

    # Numerical features
    input_data["Age"] = age
    input_data["RestingBP"] = resting_bp
    input_data["Cholesterol"] = cholesterol
    input_data["FastingBS"] = fasting_bs
    input_data["MaxHR"] = max_hr
    input_data["Oldpeak"] = oldpeak

    # One-hot encoded features
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

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    try:
        # Scale
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except Exception as e:
        st.error(f"Error: {e}")

        st.write("Expected Columns:")
        st.write(expected_columns)

        st.write("Input Columns:")
        st.write(input_df.columns.tolist())



#python3 -m streamlit run app.py