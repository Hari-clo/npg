import streamlit as st
import pickle
import numpy as np
import os

# Path to the trained model
MODEL_PATH = "linear_regression_model_using_pickle.pkl"

# Check model file
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at '{MODEL_PATH}'. Please make sure the model file is in the same folder as this app.")
    st.stop()

# Load model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Page settings
st.set_page_config(page_title="MPG Predictor 🚗", page_icon="🚗", layout="centered")

# CSS Styling
st.markdown("""
    <style>
        .main { background-color: #f0f8ff; }
        h1 { color: #333; text-align: center; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://openclipart.org/image/800px/343980", use_container_width=True)
st.sidebar.header("🔧 Instructions")
st.sidebar.markdown("""
Enter vehicle specs on the main panel and click **Predict MPG**.

📘 Based on a Linear Regression model trained on the Auto MPG dataset.
""")
st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ using Streamlit")

# Title
st.title("🚗 MPG Prediction App")
st.markdown("### Estimate your car's fuel efficiency with a few inputs!")

# Form Input
with st.form("mpg_form"):
    st.markdown("#### 🔢 Enter Vehicle Specifications")
    col1, col2 = st.columns(2)

    with col1:
        cylinders = st.number_input("Cylinders", min_value=3, max_value=16, value=6)
        displacement = st.number_input("Displacement", min_value=50.0, max_value=500.0, value=199.0)
        horsepower = st.number_input("Horsepower", min_value=40.0, max_value=300.0, value=97.0)

    with col2:
        weight = st.number_input("Weight (lbs)", min_value=1000.0, max_value=6000.0, value=2774.0)
        acceleration = st.number_input("Acceleration (0-60 mph time)", min_value=5.0, max_value=30.0, value=15.5)

    submitted = st.form_submit_button("Predict MPG")

# Predict
if submitted:
    input_features = np.array([[cylinders, displacement, horsepower, weight, acceleration]])
    prediction = model.predict(input_features)
    mpg = round(prediction[0], 2)

    st.markdown("## 🎯 Prediction Result")
    st.metric(label="Estimated MPG", value=f"{mpg} miles/gallon")

    st.info("MPG is calculated based on input features using a Linear Regression model.")
