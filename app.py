import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model pipeline
try:
    model = joblib.load('credit_risk_model.joblib')
except FileNotFoundError:
    st.error("Error: 'credit_risk_model.joblib' not found. Make sure you downloaded it from Colab and placed it in the same directory.")
    st.stop()

# Define the risk categorization function from Step 4
def get_risk_category(prob):
    # Using the thresholds we established earlier
    if prob < 0.20: return '🟢 Low Risk (Auto-Approve)'
    elif prob < 0.60: return '🟡 Medium Risk (Manual Review)'
    else: return '🔴 High Risk (Auto-Reject)'

st.set_page_config(page_title="Credit Risk Score", layout="wide")
st.title("Credit Risk Scoring Application 💳")
st.markdown("---")

# --- USER INPUTS (Using features from the dataset) ---
with st.sidebar:
    st.header("Applicant Data Input")
    
    # 1. Numeric Inputs (Top importance features like Age are first)
    age = st.slider('Age', min_value=18, max_value=100, value=30)
    income = st.number_input('Annual Income ($)', min_value=1000, value=50000, step=1000)
    loan_amount = st.number_input('Loan Amount ($)', min_value=100, value=10000, step=500)
    
    # Other features
    int_rate = st.slider('Interest Rate (%)', min_value=5.0, max_value=25.0, value=12.5)
    emp_length = st.slider('Employment Length (Years)', min_value=0.0, max_value=60.0, value=5.0)
    cred_hist_length = st.slider('Credit History Length (Years)', min_value=0.0, max_value=40.0, value=5.0)

# 2. Categorical Inputs
col1, col2, col3 = st.columns(3)
with col1:
    home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
with col2:
    loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
with col3:
    default_on_file = st.selectbox('Previous Default on File', ['No', 'Yes'])

# Map Y/N to 'Y'/'N' string as expected by the model
cb_person_default_on_file = 'Y' if default_on_file == 'Yes' else 'N'

# Calculate loan_percent_income as it's a derived feature
loan_percent_income = loan_amount / income if income > 0 else 0

# --- PREDICTION LOGIC ---
if st.button('Calculate Risk Score', type="primary"):
    
    # Create a DataFrame matching the EXACT format of the training data columns
    input_data = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_home_ownership': [home_ownership],
        'person_emp_length': [emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': ['A'], # Model needs a value, but we didn't expose this, so we use a dummy
        'loan_amnt': [loan_amount],
        'loan_int_rate': [int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cred_hist_length]
    })
    
    with st.spinner('Predicting...'):
        # Predict probability
        prediction_proba = model.predict_proba(input_data)[:, 1][0]
        
        # Get risk category
        risk_category = get_risk_category(prediction_proba)

    st.success("✅ Prediction Complete")
    
    # Display Results in Metrics
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric(label="Default Probability", value=f"{prediction_proba:.2%}")
    with col_b:
        st.markdown(f"### **Recommended Action: {risk_category}**")
