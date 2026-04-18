import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Loading the model
try:
    model = joblib.load("random_forest_model_v1.joblib")
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Engine Predictive Maintenance", layout="centered")

st.title("Engine Predictive Maintenance System")

st.write("""
This application predicts whether an **engine is healthy or likely to fail**
based on real-time sensor inputs.

### Objective:
- Detect potential engine failures early
- Reduce downtime and maintenance costs
- Improve engine performance and lifespan
""")

# ==============================
# User Inputs
# ==============================

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Engine Predictive Maintenance", layout="centered")

st.title("Engine Predictive Maintenance System")

st.write("""
Predict whether an **engine is healthy or likely to fail** based on sensor data.
""")

# ==============================
# Inputs (UPDATED WITH REAL RANGES)
# ==============================

st.subheader("Enter Engine Sensor Values")

engine_rpm = st.number_input(
    "Engine RPM",
    min_value=50, max_value=2500, value=800, step=50
)

lub_oil_pressure = st.number_input(
    "Lubricating Oil Pressure (bar)",
    min_value=0.0, max_value=8.0, value=3.0, step=0.1
)

fuel_pressure = st.number_input(
    "Fuel Pressure (bar)",
    min_value=0.0, max_value=25.0, value=6.5, step=0.1
)

coolant_pressure = st.number_input(
    "Coolant Pressure (bar)",
    min_value=0.0, max_value=8.0, value=2.5, step=0.1
)

lub_oil_temp = st.number_input(
    "Lubricating Oil Temperature (°C)",
    min_value=70.0, max_value=100.0, value=80.0, step=1.0
)

coolant_temp = st.number_input(
    "Coolant Temperature (°C)",
    min_value=60.0, max_value=120.0, value=85.0, step=1.0
)

# ==================================
# Validation Checks
# ==================================
errors = []

if not (50 <= engine_rpm <= 2500):
    errors.append("Engine RPM must be between 50 and 2500.")

if not (0 <= lub_oil_pressure <= 8):
    errors.append("Lub Oil Pressure must be between 0 and 8 bar.")

if not (0 <= fuel_pressure <= 25):
    errors.append("Fuel Pressure must be between 0 and 25 bar.")

if not (0 <= coolant_pressure <= 8):
    errors.append("Coolant Pressure must be between 0 and 8 bar.")

if not (70 <= lub_oil_temp <= 100):
    errors.append("Lub Oil Temperature must be between 70–100°C.")

if not (60 <= coolant_temp <= 120):
    errors.append("Coolant Temperature must be between 60–120°C.")

if errors:
    st.error("Invalid Input Detected")
    for err in errors:
        st.warning(err)
    st.stop()

# ==============================
# Create DataFrame
# ==============================


input_data = pd.DataFrame([{
    'Engine rpm': int(engine_rpm),
    'Lub oil pressure': float(lub_oil_pressure),
    'Fuel pressure': float(fuel_pressure),
    'Coolant pressure': float(coolant_pressure),
    'lub oil temp': float(lub_oil_temp),
    'Coolant temp': float(coolant_temp)
}])
# ==============================
# Prediction
# ==============================

if st.button("Predict Engine Condition"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")

    if prediction == 0:
        st.success("Engine is Operating Normally (No Maintenance Required)")
    else:
        st.error("Engine Failure Likely! Maintenance Required")

# ==============================
# Footer
# ==============================

st.write("---")
st.write("""
### About the Model:
- Model Type: Random Forest Classifier
- Task: Binary Classification (0 = Normal, 1 = Failure)

### Benefits:
Reduce unplanned breakdowns
Optimize maintenance schedules
Improve engine lifespan
Enable data-driven decisions
""")
