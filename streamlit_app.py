import os
import pickle
import numpy as np
import streamlit as st

# 🔧 Load model and encoder
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
encoder_path = os.path.join(BASE_DIR, "models", "encoder_district.pkl")

model = pickle.load(open(model_path, "rb"))
encoder = pickle.load(open(encoder_path, "rb"))

# 🌐 Page configuration
st.set_page_config(page_title="Punjab Crop Area Estimator", layout="centered")

# 🏷️ Title and description
st.title("Punjab Crop Area Estimator 🌾")
st.write("Estimate Rabi Fruits crop area (in acres) based on selected district and rainfall input.")

st.divider()

# 📍 Input section
st.subheader("Select Inputs")
districts = sorted([d for d in encoder.classes_ if isinstance(d, str)])
selected_district = st.selectbox("District", districts)

rainfall = st.slider("Rainfall (mm)", min_value=0, max_value=500, value=200, step=10)

# 📊 Feature engineering
district_encoded = encoder.transform([selected_district])[0]
features = np.array([[district_encoded, rainfall, np.log1p(rainfall), district_encoded * rainfall]])

# 🔮 Prediction
predicted_area = model.predict(features)[0]

# 📢 Output section
st.divider()
st.subheader("Crop Area Estimate")

st.success(
    f"With {rainfall} mm rainfall in {selected_district}, the estimated crop area for Rabi Fruits is approximately {predicted_area:,.2f} acres. This result is generated using a trained prediction model and intended to guide planning."
)