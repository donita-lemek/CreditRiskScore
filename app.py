import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Configuration ---
MODEL_PATH = 'credit_risk_model.joblib'

# Set up the main page properties
st.set_page_config(
    page_title="Credit Risk Score Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading ---
# We use st.cache_resource to load the large model file only once
@st.cache_resource
def load_model():
    """Loads the pre-trained XGBoost pipeline model."""
    try:
        # joblib.load is used to deserialize the saved pipeline
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it is in the same directory.")
        st.stop()
    except Exception as e:
        # This will catch the serialization mismatch if the config.toml fix fails
        st.error(f"Error loading model due to version mismatch. Please verify Python version in config.toml. Details: {e}")
        st.stop()

model = load_model()

# --- Risk Categorization Function ---
# This uses the validated thresholds from your Colab analysis
def get_risk_category(prob):
    """Translates default probability into a clear risk group."""
    if prob < 0.20: 
        return '🟢 Low Risk', 'Successfully projected to repay the loan. Recommended for Auto-Approval.'
    elif prob < 0.60: 
        return '🟡 Medium Risk', 'Risk is elevated but not critical. Recommended for Manual Review and secondary verification.'
    else: 
        return '🔴 High Risk', 'High probability of default. Recommended for Rejection or requiring collateral.'

# --- UI Setup ---
st.title("Credit Risk Score Model Deployment 📊")
st.markdown("""
    This application uses an **XGBoost Classifier (AUC: 0.9479)** to predict the probability 
    of a loan applicant defaulting based on key financial and demographic data.
    **Key Predictor:** Applicant Age (`person_age`).
""")
st.markdown("---")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Applicant Profile Data")
    
    # 1. Key Numeric Inputs (Matching Feature Importance findings)
    st.subheader("Financial & Demographic Factors")
    
    # num__person_age (Key Predictor)
    age = st.slider('Age', min_value=18, max_value=100, value=30)
    # num__person_income
    income = st.number_input('Annual Income ($)', min_value=1000, max_value=500000, value=50000, step=1000)
    
    st.subheader("Loan Details")
    # num__loan_amnt
    loan_amount = st.number_input('Loan Amount ($)', min_value=100, max_value=50000, value=10000, step=500)
    # num__loan_int_rate
    int_rate = st.slider('Interest Rate (%)', min_value=5.0, max_value=25.0, value=12.5)
    
    # 2. Other Features
    st.subheader("Other Required Factors")
    # num__person_emp_length
    emp_length = st.slider('Employment Length (Years)', min_value=0.0, max_value=60.0, value=5.0)
    # num__cb_person_cred_hist_length
    cred_hist_length = st.slider('Credit History Length (Years)', min_value=0.0, max_value=40.0, value=5.0)


# --- Main Column for Categorical Inputs ---
col1, col2, col3 = st.columns(3)
with col1:
    # cat__person_home_ownership
    home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
with col2:
    # cat__loan_intent
    loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
with col3:
    # cat__cb_person_default_on_file
    default_on_file = st.selectbox('Previous Default on File', ['No', 'Yes'])

# Map Y/N to 'Y'/'N' string as expected by the model's preprocessing
cb_person_default_on_file = 'Y' if default_on_file == 'Yes' else 'N'

# Calculate the derived feature: loan_percent_income (num__loan_percent_income)
loan_percent_income = loan_amount / income if income > 0 else 0

# The model requires loan_grade, which we didn't use as input, so we use a safe default ('A')
loan_grade = 'A' 


# --- Prediction Button ---
if st.button('GENERATE RISK ASSESSMENT', type="primary", use_container_width=True):
    
    # Create a DataFrame matching the EXACT column names and order of the training data
    input_data = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_home_ownership': [home_ownership],
        'person_emp_length': [emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade], 
        'loan_amnt': [loan_amount],
        'loan_int_rate': [int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cred_hist_length]
    })
    
    with st.spinner('Running XGBoost Model...'):
        # Predict probability of class 1 (default)
        prediction_proba = model.predict_proba(input_data)[:, 1][0]
        
        risk_label, risk_advice = get_risk_category(prediction_proba)

    st.success("Assessment Complete")
    st.markdown("---")

    # --- Display Results ---
    
    col_result1, col_result2 = st.columns([1, 2])
    
    with col_result1:
        st.metric(
            label="Predicted Default Probability (P=1)", 
            value=f"{prediction_proba * 100:.2f}%"
        )
    
    with col_result2:
        st.subheader(f"Risk Group: {risk_label}")
        st.info(risk_advice)
        
    st.markdown("---")
    st.markdown(f"**Calculated Loan % Income (Key Derived Feature):** `{loan_percent_income:.2%}`")
