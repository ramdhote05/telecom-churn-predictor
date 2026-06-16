import streamlit as st
import joblib
import numpy as np

# 1. Load the trained model
model = joblib.load("churn_model.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Telecom Customer Churn Predictor")
st.write("Fill in the customer attributes below to evaluate churn risk and view retention plans.")

# 2. Recreate input layout for all 19 features in training order
st.header("Customer Demographics & Services")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"], index=0)
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
    partner = st.selectbox("Has Partner?", ["No", "Yes"], index=0)
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"], index=0)
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"], index=1)
    multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"], index=0)

with col2:
    internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"], index=1)
    online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"], index=0)
    online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"], index=0)
    device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"], index=0)
    tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"], index=0)
    streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"], index=0)
    streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"], index=0)

st.header("Billing & Contract Account Details")
col3, col4 = st.columns(2)

with col3:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], index=1)
    payment_method = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], index=2)

with col4:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0, step=0.1)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=780.0, step=1.0)

# 3. Map categorical selections back to LabelEncoder integers
gender_map = {"Female": 0, "Male": 1}
binary_map = {"No": 0, "Yes": 1}
multi_map = {"No": 0, "No phone service": 1, "No internet service": 1, "Yes": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map = {"Bank transfer (automatic)": 0, "Credit card (automatic)": 1, "Electronic check": 2, "Mailed check": 3}

# Assemble the precise 19-element feature list
feature_values = [
    gender_map[gender],
    binary_map[senior_citizen],
    binary_map[partner],
    binary_map[dependents],
    int(tenure),
    binary_map[phone_service],
    multi_map[multiple_lines],
    internet_map[internet_service],
    multi_map[online_security],
    multi_map[online_backup],
    multi_map[device_protection],
    multi_map[tech_support],
    multi_map[streaming_tv],
    multi_map[streaming_movies],
    contract_map[contract],
    binary_map[paperless_billing],
    payment_map[payment_method],
    float(monthly_charges),
    float(total_charges)
]

# 4. Prediction Execution
if st.button("Analyze Churn Risk", type="primary"):
    input_data = np.array([feature_values])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    st.subheader("Analysis Results")
    
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk Detector!** (Probability: {probability * 100:.2f}%)")
        
        # Actionable Business Recommendation Engine
        st.warning("### 📋 AI Retention Strategy Recommendation:")
        if contract_map[contract] == 0:
            st.write("- **Contract Incentive:** Customer is on a month-to-month plan. Offer a 15% discount if they upgrade to a 1-year stable contract.")
        if multi_map[tech_support] == 0 and internet_map[internet_service] != 2:
            st.write("- **Support Upgrades:** Customer does not have Tech Support enabled. Provide 3 months of complimentary Technical Support setup.")
    else:
        st.success(f"✅ **Low Churn Risk Customer.** (Probability: {probability * 100:.2f}%)")
        st.write("Customer shows strong healthy account trends. Maintain normal outreach.")
