
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# -----------------------------------------
# Locate files
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "logistic_regression_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# -----------------------------------------
# Load trained model and scaler
# -----------------------------------------

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)

# -----------------------------------------
# App title
# -----------------------------------------

st.title("Diabetes Prediction Using Logistic Regression")

st.write(
    "Enter the patient information below to generate "
    "a prediction using the trained Logistic Regression model."
)

# -----------------------------------------
# User Inputs
# -----------------------------------------

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    value=80.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    value=30
)

# -----------------------------------------
# Prediction
# -----------------------------------------

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    # Apply the same scaling used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Calculate probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Display prediction
    if prediction == 1:
        st.warning(
            "Model Prediction: Diabetes Positive"
        )
    else:
        st.success(
            "Model Prediction: Diabetes Negative"
        )

    st.write(
        f"Predicted probability of positive class: "
        f"{probability:.2%}"
    )

    st.caption(
        "This application demonstrates a machine-learning "
        "model and is not a medical diagnosis."
    )
