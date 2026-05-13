import os
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model artefacts
model = joblib.load(os.path.join(BASE_DIR, "logis_heart_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "heart_scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "heart_columns.pkl"))
expected_columns = list(columns)

# ── Pre-scaler statistics from the original training dataset ─────────────────
# These are the mean / std of the CONTINUOUS numeric columns in the Kaggle
# Heart Disease dataset (918 rows).  They match the first StandardScaler that
# was applied before heart_scaler.pkl was fitted.
PRE_SCALE_STATS = {
    "Age":         {"mean": 53.51,  "std": 9.43},
    "RestingBP":   {"mean": 132.40, "std": 18.51},
    "Cholesterol": {"mean": 198.80, "std": 109.38},
    "MaxHR":       {"mean": 136.81, "std": 25.46},
    "Oldpeak":     {"mean": 0.887,  "std": 1.067},
}

# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align: center;'>
        ❤️🩺 Heart Disease Prediction <br>
         💻📊
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("""
### Using Machine Learning & AI to predict heart disease and support early healthcare decisions 🚀🤖

Please enter the following details to predict the likelihood of heart disease:
""")
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["Male", "Female"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ASY", "ATA", "NAP", "TA"],
    help="ASY = Asymptomatic (highest risk), ATA = Atypical Angina, "
         "NAP = Non-Anginal Pain, TA = Typical Angina",
)

resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)

cholesterol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1],
    format_func=lambda x: "Yes (> 120 mg/dl)" if x == 1 else "No (≤ 120 mg/dl)",
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"],
    help="Normal, ST = ST-T wave abnormality, LVH = Left Ventricular Hypertrophy",
)

max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

exercise_angina = st.selectbox(
    "Exercise Induced Angina", ["N", "Y"],
    format_func=lambda x: "Yes" if x == "Y" else "No",
)

oldpeak = st.slider(
    "Oldpeak (ST depression induced by exercise)", 0.0, 6.0, 1.0, step=0.1
)

st_slope = st.selectbox(
    "Slope of the Peak Exercise ST Segment",
    ["Up", "Flat", "Down"],
    help="Up = upsloping (lower risk), Flat = flat, Down = downsloping (higher risk)",
)

# ── Prediction ───────────────────────────────────────────────────────────────
if st.button("Predict"):

    # ── Step 1: Build raw numeric + binary features ──────────────────────────
    raw_numerics = {
        "Age":         age,
        "RestingBP":   resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS":   fasting_bs,   # binary — NOT pre-scaled
        "MaxHR":       max_hr,
        "Oldpeak":     oldpeak,
    }

    # ── Step 2: Pre-standardise continuous columns ────────────────────────────
    # (FastingBS is binary and was NOT pre-scaled in training; leave it raw.)
    pre_scaled = dict(raw_numerics)
    for col in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]:
        mu, sigma = PRE_SCALE_STATS[col]["mean"], PRE_SCALE_STATS[col]["std"]
        pre_scaled[col] = (raw_numerics[col] - mu) / sigma

    one_hot = {
        "Sex_M":               1 if sex == "Male" else 0,
        "ChestPainType_ATA":   1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP":   1 if chest_pain == "NAP" else 0,
        "ChestPainType_TA":    1 if chest_pain == "TA"  else 0,
        # ASY → all three above = 0  (baseline)
        "RestingECG_Normal":   1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST":       1 if resting_ecg == "ST"     else 0,
        # LVH  → both above = 0  (baseline)
        "ExerciseAngina_Y":    1 if exercise_angina == "Y"    else 0,
        "ST_Slope_Flat":       1 if st_slope == "Flat" else 0,
        "ST_Slope_Up":         1 if st_slope == "Up"   else 0,
        # Down → both above = 0  (baseline)
    }

    # ── Step 4: Assemble DataFrame in the exact column order the model expects ─
    combined = {**pre_scaled, **one_hot}
    input_df = pd.DataFrame([combined])

    # Fill any still-missing columns with 0 and reorder
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # ── Step 5: Apply heart_scaler and predict ────────────────────────────────
    scaled_input = scaler.transform(input_df)
    prediction   = model.predict(scaled_input)[0]
    probability  = model.predict_proba(scaled_input)[0]

    # ── Step 6: Display result ────────────────────────────────────────────────
    st.markdown("---")
    if prediction == 1:
        st.error(
            f"⚠️ **High Risk of Heart Disease**\n\n"
            f"Estimated probability: **{probability[1]*100:.1f}%**"
        )
    else:
        st.success(
            f"✅ **Low Risk of Heart Disease**\n\n"
            f"Estimated probability of disease: **{probability[1]*100:.1f}%**"
        )

    with st.expander("Show prediction details"):
        st.write("**Processed feature values sent to model:**")
        for feature_name, feature_value in input_df.iloc[0].items():
            st.write(f"- **{feature_name}**: {feature_value}")
        st.write(f"**Model raw probabilities:** No Disease = {probability[0]:.4f} | Disease = {probability[1]:.4f}")
