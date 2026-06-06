import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="InsureIQ — AI Cost Estimator",
    page_icon="🛡️",
    layout="wide"
)

# ==================================================
# LOAD FILES
# ==================================================

BASE_DIR = Path(__file__).parent

model          = joblib.load(BASE_DIR / "xgb_model.joblib")
scaler         = joblib.load(BASE_DIR / "scaler.joblib")
expected_columns = joblib.load(BASE_DIR / "columns.joblib")

# ==================================================
# CSS  (only styling, zero layout HTML)
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, .stApp {
    background: #040c18;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container { padding-top: 2.5rem; max-width: 1100px; }

/* Typography */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #07111f !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background: #0f1e30 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* Number input */
input[type="number"] {
    background: #0f1e30 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* Labels */
.stSlider label, .stSelectbox label, .stNumberInput label {
    color: #64748b !important;
    font-size: 13px !important;
    letter-spacing: 0.04em !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    height: 58px !important;
    width: 100% !important;
    box-shadow: 0 0 28px rgba(79,70,229,0.4) !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    box-shadow: 0 0 44px rgba(79,70,229,0.6) !important;
    transform: translateY(-1px) !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #0f1e30;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 16px 20px !important;
}

div[data-testid="metric-container"] label {
    color: #475569 !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 20px !important;
}

/* Info / warning / success boxes */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 13px !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* Caption */
.stCaption { color: #334155 !important; font-size: 11.5px !important; }

</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🛡️ InsureIQ")
    st.caption("Medical Insurance Cost Estimator · XGBoost")

    st.divider()

    st.markdown("**MODEL FEATURES**")
    features = ["Age", "Gender", "BMI", "Children", "Smoking Status", "Region", "BMI Category"]
    for f in features:
        st.markdown(f"&nbsp;&nbsp;✦ {f}")

    st.divider()

    st.markdown("🟢 &nbsp;**Model Status** — Loaded & Ready")
    st.markdown("🤖 &nbsp;**Algorithm** — XGBoost Regressor")
    st.markdown("📦 &nbsp;**Scaler** — Standard Scaler")

    st.divider()

    st.caption("Developed by SupremeInferno 😳")

# ==================================================
# HEADER
# ==================================================

st.title("💰 Insurance Charges Predictor")
st.markdown("##### AI-powered estimate of your annual medical insurance cost")
st.divider()

# ==================================================
# INPUTS
# ==================================================

st.subheader("🧾 Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Gender", ["Male", "Female"])

with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

with col3:
    smoker = st.selectbox("Smoking Status", ["No", "Yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

st.divider()

# ==================================================
# BMI CATEGORY
# ==================================================

if bmi < 18.5:
    bmi_category = "🔵 Underweight"
    bmi_color    = "blue"
elif bmi < 25:
    bmi_category = "🟢 Normal Weight"
    bmi_color    = "green"
elif bmi < 30:
    bmi_category = "🟡 Overweight"
    bmi_color    = "orange"
else:
    bmi_category = "🟠 Obese"
    bmi_color    = "red"

bmi_col1, bmi_col2, bmi_col3, bmi_col4 = st.columns(4)

with bmi_col1:
    st.metric("BMI Category", bmi_category)
with bmi_col2:
    st.metric("BMI Value", f"{bmi:.1f}")
with bmi_col3:
    st.metric("Age", f"{age} yrs")
with bmi_col4:
    st.metric("Smoker", smoker)

st.divider()

# ==================================================
# PREDICT BUTTON
# ==================================================

btn_col, _ = st.columns([1, 2])

with btn_col:
    predict = st.button("⚡ Estimate Insurance Cost")

# ==================================================
# PREDICTION
# ==================================================

if predict:

    input_df = pd.DataFrame([{
        "age":                       age,
        "is_female":                 1 if sex == "Female" else 0,
        "bmi":                       bmi,
        "children":                  children,
        "is_smoker":                 1 if smoker == "Yes" else 0,
        "region_northeast":          1 if region == "northeast" else 0,
        "region_northwest":          1 if region == "northwest" else 0,
        "region_southeast":          1 if region == "southeast" else 0,
        "region_southwest":          1 if region == "southwest" else 0,
        "bmi_category_underweight":  1 if bmi < 18.5 else 0,
        "bmi_category_normalweight": 1 if 18.5 <= bmi < 25 else 0,
        "bmi_category_overweight":   1 if 25 <= bmi < 30 else 0,
        "bmi_category_obese":        1 if bmi >= 30 else 0
    }])

    input_df   = input_df[expected_columns]
    scaled_inp = scaler.transform(input_df)
    prediction = model.predict(scaled_inp)[0]

    # ── Cost category ──
    if prediction < 5000:
        category = "🟢 Low Cost"
        st.success(f"### Estimated Charges: **${prediction:,.2f}**\n**{category}** — Great news! Your estimated premium is low.")
    elif prediction < 15000:
        category = "🟡 Moderate Cost"
        st.warning(f"### Estimated Charges: **${prediction:,.2f}**\n**{category}** — Your estimated premium is in the average range.")
    else:
        category = "🔴 High Cost"
        st.error(f"### Estimated Charges: **${prediction:,.2f}**\n**{category}** — Your estimated premium is on the higher side.")

    st.divider()

    # ── Summary metrics ──
    st.subheader("📊 Prediction Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Age",    f"{age} yrs")
        st.metric("BMI",    f"{bmi:.1f}")

    with c2:
        st.metric("Gender",  sex)
        st.metric("Smoker",  smoker)

    with c3:
        st.metric("Children", children)
        st.metric("Region",   region.capitalize())

    st.divider()
    st.caption("⚠️ This prediction is generated by a Machine Learning model and is intended for educational purposes only.")

# ==================================================
# FOOTER
# ==================================================

st.divider()
st.caption("Built with ❤️ using Streamlit · Scikit-Learn · XGBoost &nbsp;|&nbsp; InsureIQ by SupremeInferno")



#python3 -m streamlit run insurance.py
