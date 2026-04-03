import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# =====================
# Custom CSS for good UX
# =====================
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 20px;
        padding: 15px;
        border-radius: 10px;
        border: none;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #FF0000;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
    }
    .danger {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# =====================
# Load Model & Columns
# =====================
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    model_columns = json.load(f)

# =====================
# Header
# =====================
st.title("📡 Telecom Customer Churn Predictor")
st.subheader("Predict whether a customer will leave the service or not")
st.write("---")

# =====================
# Input Form - 3 Columns
# =====================
st.header("👤 Customer Details")
st.write("")

# Row 1
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    SeniorCitizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
    SeniorCitizen = 1 if SeniorCitizen == "Yes" else 0

with col3:
    Partner = st.selectbox("Has Partner?", ["Yes", "No"])

# Row 2
col4, col5, col6 = st.columns(3)

with col4:
    Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

with col5:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col6:
    Contract = st.selectbox("Contract Type", 
                            ["Month-to-month", "One year", "Two year"])

# Row 3
st.write("")
st.header("📞 Services")
st.write("")

col7, col8, col9 = st.columns(3)

with col7:
    PhoneService = st.selectbox("Phone Service?", ["Yes", "No"])

with col8:
    MultipleLines = st.selectbox("Multiple Lines?", 
                                  ["Yes", "No", "No phone service"])

with col9:
    InternetService = st.selectbox("Internet Service?", 
                                    ["DSL", "Fiber optic", "No"])

# Row 4
col10, col11, col12 = st.columns(3)

with col10:
    OnlineSecurity = st.selectbox("Online Security?", 
                                   ["Yes", "No", "No internet service"])

with col11:
    OnlineBackup = st.selectbox("Online Backup?", 
                                 ["Yes", "No", "No internet service"])

with col12:
    DeviceProtection = st.selectbox("Device Protection?", 
                                     ["Yes", "No", "No internet service"])

# Row 5
col13, col14, col15 = st.columns(3)

with col13:
    TechSupport = st.selectbox("Tech Support?", 
                                ["Yes", "No", "No internet service"])

with col14:
    StreamingTV = st.selectbox("Streaming TV?", 
                                ["Yes", "No", "No internet service"])

with col15:
    StreamingMovies = st.selectbox("Streaming Movies?", 
                                    ["Yes", "No", "No internet service"])

# Row 6
st.write("")
st.header("💰 Billing Details")
st.write("")

col16, col17, col18 = st.columns(3)

with col16:
    PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])

with col17:
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

with col18:
    MonthlyCharges = st.number_input("Monthly Charges ($)", 
                                      min_value=0.0, 
                                      max_value=200.0, 
                                      value=50.0)

TotalCharges = st.number_input("Total Charges ($)", 
                                min_value=0.0, 
                                max_value=10000.0, 
                                value=500.0)

# =====================
# Predict Button
# =====================
st.write("---")
if st.button("🔮 Predict Churn"):

    # Build input dict
    input_dict = {
        'SeniorCitizen': SeniorCitizen,
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'gender_Female': 1 if gender == 'Female' else 0,
        'gender_Male': 1 if gender == 'Male' else 0,
        'Partner_No': 1 if Partner == 'No' else 0,
        'Partner_Yes': 1 if Partner == 'Yes' else 0,
        'Dependents_No': 1 if Dependents == 'No' else 0,
        'Dependents_Yes': 1 if Dependents == 'Yes' else 0,
        'PhoneService_No': 1 if PhoneService == 'No' else 0,
        'PhoneService_Yes': 1 if PhoneService == 'Yes' else 0,
        'MultipleLines_No': 1 if MultipleLines == 'No' else 0,
        'MultipleLines_No phone service': 1 if MultipleLines == 'No phone service' else 0,
        'MultipleLines_Yes': 1 if MultipleLines == 'Yes' else 0,
        'InternetService_DSL': 1 if InternetService == 'DSL' else 0,
        'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
        'InternetService_No': 1 if InternetService == 'No' else 0,
        'OnlineSecurity_No': 1 if OnlineSecurity == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if OnlineSecurity == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if OnlineSecurity == 'Yes' else 0,
        'OnlineBackup_No': 1 if OnlineBackup == 'No' else 0,
        'OnlineBackup_No internet service': 1 if OnlineBackup == 'No internet service' else 0,
        'OnlineBackup_Yes': 1 if OnlineBackup == 'Yes' else 0,
        'DeviceProtection_No': 1 if DeviceProtection == 'No' else 0,
        'DeviceProtection_No internet service': 1 if DeviceProtection == 'No internet service' else 0,
        'DeviceProtection_Yes': 1 if DeviceProtection == 'Yes' else 0,
        'TechSupport_No': 1 if TechSupport == 'No' else 0,
        'TechSupport_No internet service': 1 if TechSupport == 'No internet service' else 0,
        'TechSupport_Yes': 1 if TechSupport == 'Yes' else 0,
        'StreamingTV_No': 1 if StreamingTV == 'No' else 0,
        'StreamingTV_No internet service': 1 if StreamingTV == 'No internet service' else 0,
        'StreamingTV_Yes': 1 if StreamingTV == 'Yes' else 0,
        'StreamingMovies_No': 1 if StreamingMovies == 'No' else 0,
        'StreamingMovies_No internet service': 1 if StreamingMovies == 'No internet service' else 0,
        'StreamingMovies_Yes': 1 if StreamingMovies == 'Yes' else 0,
        'Contract_Month-to-month': 1 if Contract == 'Month-to-month' else 0,
        'Contract_One year': 1 if Contract == 'One year' else 0,
        'Contract_Two year': 1 if Contract == 'Two year' else 0,
        'PaperlessBilling_No': 1 if PaperlessBilling == 'No' else 0,
        'PaperlessBilling_Yes': 1 if PaperlessBilling == 'Yes' else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if PaymentMethod == 'Bank transfer (automatic)' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0,
    }

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_columns]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # =====================
    # Show Results
    # =====================
    st.write("---")
    st.header("📊 Prediction Result")

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.metric("Will Churn? 🔴", 
                  "YES - Will Leave!" if prediction == 1 else "NO - Will Stay!")

    with col_r2:
        st.metric("Churn Probability 📉", 
                  f"{probability[1]*100:.1f}%")

    with col_r3:
        st.metric("Retention Probability 📈", 
                  f"{probability[0]*100:.1f}%")

    st.write("")

    if prediction == 1:
        st.error("⚠️ This customer is LIKELY TO LEAVE! Take action immediately!")
        st.write("**Suggested Actions:**")
        st.write("- 📞 Call the customer and offer a discount")
        st.write("- 🎁 Offer a loyalty reward or upgrade")
        st.write("- 📋 Switch them to a longer contract")
    else:
        st.success("✅ This customer is LIKELY TO STAY! Keep up the good work!")
        st.write("**Suggested Actions:**")
        st.write("- 🌟 Send a thank you message")
        st.write("- 🎯 Offer them premium upgrades")
        st.write("- 💎 Add them to loyalty program")