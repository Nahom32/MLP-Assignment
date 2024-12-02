import streamlit as st
import numpy as np
import pickle

# Load the saved model
@st.cache_resource
def load_model():
    with open("regression_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Input slugs
st.title("MPG Prediction App")

# Collect user inputs for features
cylinders = st.number_input("Cylinders", min_value=3, max_value=12, value=4, step=1)
displacement = st.number_input("Displacement (cu. inches)", min_value=50.0, max_value=500.0, value=200.0, step=1.0)
horsepower = st.number_input("Horsepower", min_value=40.0, max_value=300.0, value=100.0, step=1.0)
weight = st.number_input("Weight (lbs)", min_value=1500.0, max_value=5000.0, value=3000.0, step=1.0)
acceleration = st.number_input("Acceleration (0-60 mph in seconds)", min_value=5.0, max_value=30.0, value=15.0, step=0.1)
model_year = st.slider("Model Year", min_value=1970, max_value=1985, value=1980, step=1)
origin = st.selectbox("Origin", options=[1, 2, 3], format_func=lambda x: {1: "USA", 2: "Europe", 3: "Asia"}[x])

# Prepare input for the model
input_data = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]])

# Predict MPG
if st.button("Predict MPG"):
    prediction = model.predict(input_data)
    st.write(f"### Predicted MPG: {prediction[0]:.2f}")
