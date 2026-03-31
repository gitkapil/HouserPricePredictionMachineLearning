
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="California House Price Prediction", page_icon=":house:", layout="wide")

# Enhanced colorful CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stTextInput>div>div>input, .stNumberInput>div>input {
        background-color: #fffbe7;
        border-radius: 8px;
        border: 1.5px solid #ffd54f;
        font-size: 18px;
        padding: 10px;
        color: #333;
    }
    .stSelectbox>div>div>div>div {
        background-color: #e1f5fe;
        border-radius: 8px;
        border: 1.5px solid #4fc3f7;
        font-size: 18px;
        color: #0277bd;
    }
    .stButton>button {
        background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        color: #fff;
        border-radius: 8px;
        font-size: 18px;
        padding: 10px 24px;
        border: none;
        margin-top: 10px;
        font-weight: bold;
        box-shadow: 0 2px 8px #b2f7ef44;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #38f9d7 0%, #43e97b 100%);
        color: #222;
    }
    .result-box {
        background: linear-gradient(135deg, #fceabb 0%, #f8b500 100%);
        border-radius: 16px;
        padding: 32px 20px;
        margin-top: 40px;
        font-size: 28px;
        color: #fff;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 24px #f8b50033;
        border: 2px solid #fffde7;
    }
    .result-title {
        font-size: 20px;
        color: #fffde7;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;'>
    <h1 style='color:#1565c0; font-size:2.8rem; font-weight:900; letter-spacing:2px;'>🏡 California House Price Prediction</h1>
    <p style='color:#4fc3f7; font-size:1.3rem; font-weight:500;'>Enter house features and get instant price predictions!</p>
</div>
""", unsafe_allow_html=True)

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


if not (os.path.exists(MODEL_FILE) and os.path.exists(PIPELINE_FILE)):
    st.error("Model or pipeline file not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

# Example input columns (update as per your model's features)
input_columns = [
    "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income", "ocean_proximity"
]

# Ocean proximity options (update as per your data)
ocean_options = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

# Use columns for layout
left, right = st.columns([1.2, 1])

with left:
    with st.form("prediction_form"):
        st.subheader("Enter House Features:")
        longitude = st.number_input("Longitude", value=-120.0, format="%.4f")
        latitude = st.number_input("Latitude", value=35.0, format="%.4f")
        housing_median_age = st.number_input("Housing Median Age", min_value=1, max_value=100, value=30)
        total_rooms = st.number_input("Total Rooms", min_value=1, value=1000)
        total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=200)
        population = st.number_input("Population", min_value=1, value=500)
        households = st.number_input("Households", min_value=1, value=200)
        median_income = st.number_input("Median Income", min_value=0.0, value=3.0, format="%.2f")
        ocean_proximity = st.selectbox("Ocean Proximity", options=ocean_options)
        submitted = st.form_submit_button("Predict House Value")

# Show prediction on the right
with right:
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    if submitted:
        input_df = pd.DataFrame({
            "longitude": [longitude],
            "latitude": [latitude],
            "housing_median_age": [housing_median_age],
            "total_rooms": [total_rooms],
            "total_bedrooms": [total_bedrooms],
            "population": [population],
            "households": [households],
            "median_income": [median_income],
            "ocean_proximity": [ocean_proximity]
        })
        try:
            transformed = pipeline.transform(input_df)
            prediction = model.predict(transformed)[0]
            st.session_state['prediction'] = prediction
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.session_state['prediction'] = None
    if st.session_state['prediction'] is not None:
        st.markdown(f'''
            <div class="result-box">
                <div class="result-title">Predicted Median House Value</div>
                <span style="color:#2e7d32; font-size:2.2rem;">${st.session_state['prediction']:,.2f}</span>
                <div style="margin-top:10px; font-size:1.1rem; color:#fffde7;">(Based on your input)</div>
            </div>
        ''', unsafe_allow_html=True)
