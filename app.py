import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Approval Prediction App")

st.write("Enter customer details below:")

# User Inputs
principal = st.number_input("Principal Amount", value=1000)
terms = st.number_input("Loan Terms", value=30)
effective_date = st.number_input("Effective Date (Encoded)", value=2)
due_date = st.number_input("Due Date (Encoded)", value=5)
paid_off_time = st.number_input("Paid Off Time", value=50)
past_due_days = st.number_input("Past Due Days", value=0)
age = st.number_input("Age", value=35)
education = st.selectbox("Education (0=Low,1=Medium,2=High)", [0,1,2])
gender = st.selectbox("Gender (0=Female,1=Male)", [0,1])

# Predict Button
if st.button("Predict Loan Status"):

    input_data = np.array([[principal, terms, effective_date, due_date,
                            paid_off_time, past_due_days,
                            age, education, gender]])

    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved (Probability: {probability[0][1]:.2f})")
    else:
        st.error(f"❌ Loan Rejected (Probability: {probability[0][1]:.2f})")
