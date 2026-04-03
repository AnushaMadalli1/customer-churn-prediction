import streamlit as st
import requests
import pandas as pd

st.title("Customer Churn Prediction App")

st.header("Enter Customer Details")
# Example fields for telco churn
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
# Add more fields as per dataset

if st.button("Predict Churn"):
    customer_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        # Add more
    }
    
    # Call API
    try:
        response = requests.post("http://localhost:8000/predict", json={"data": customer_data})
        prob = response.json()["churn_probability"]
        st.metric("Churn Probability", f"{prob:.2%}")
        if prob > 0.5:
            st.warning("High churn risk!")
        else:
            st.success("Low churn risk.")
    except:
        st.error("Start API server first: uvicorn src.api.main:app --reload")

# Data upload for batch prediction
uploaded = st.file_uploader("Upload CSV for batch prediction")
if uploaded:
    df = pd.read_csv(uploaded)
    # Process and predict...

