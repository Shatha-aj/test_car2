import streamlit as st
import pandas as pd
import joblib

# Load model and helpers
model = joblib.load("accident_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Accident Severity Predictor", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5em 2em;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚦 Accident Severity Prediction")
st.markdown("Use this app to predict the likely severity of a road accident based on driving and environmental conditions.")

# Input fields
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        time = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
        sex = st.selectbox("Sex of Driver", ["Male", "Female"])
        age_band = st.selectbox("Age Band of Driver", ["18-30", "31-50", "Over 51", "Under 18"])
        experience = st.selectbox("Driving Experience", ["1-2yr", "2-5yr", "5-10yr", "Above 10yr", "No Experience"])
        area = st.selectbox("Area Accident Occurred", ["Urban", "Rural"])
    with col2:
        day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        vehicle = st.selectbox("Type of Vehicle", ["Automobile", "Lorry", "Bus", "Taxi", "Other"])
        road = st.selectbox("Road Surface Conditions", ["Dry", "Wet", "Snow", "Flood"])
        light = st.selectbox("Light Conditions", ["Daylight", "Night", "Twilight"])
        weather = st.selectbox("Weather Conditions", ["Clear", "Rainy", "Foggy", "Windy"])
        cause = st.selectbox("Cause of Accident", ["Over Speeding", "Driving under the influence", "No seat belt", "Mechanical failure", "Unknown"])

    submitted = st.form_submit_button("Predict Severity")

if submitted:
    # Prepare input DataFrame
    input_dict = {
        "Time": [time],
        "Day_of_week": [day],
        "Age_band_of_driver": [age_band],
        "Sex_of_driver": [sex],
        "Driving_experience": [experience],
        "Type_of_vehicle": [vehicle],
        "Area_accident_occured": [area],
        "Road_surface_conditions": [road],
        "Light_conditions": [light],
        "Weather_conditions": [weather],
        "Cause_of_accident": [cause],
    }

    input_df = pd.DataFrame(input_dict)
    input_df = pd.get_dummies(input_df)

    # Add missing columns and reorder
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Predict
    prediction = model.predict(input_df)
    severity = label_encoder.inverse_transform(prediction)

    st.success(f"🚨 Predicted Accident Severity: {severity[0]}")
