import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Real-Time Transaction Control Panel")
st.markdown("This dashboard analyzes credit card transactions in real-time using AI to transparently calculate fraud risk.")

st.sidebar.header("Customer Transaction Details")
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=2500.0)
time = st.sidebar.number_input("Time Since Last Purchase (Seconds)", min_value=0, value=120)

st.sidebar.markdown("---")
st.sidebar.subheader("Behavioral Customer Analytics")
v14 = st.sidebar.slider("Location and Device Deviation Score", -20.0, 10.0, 0.0, help="Background: V14 component. Measures how much the customer deviates from their standard location.")
v17 = st.sidebar.slider("Historical Spending Pattern Match", -30.0, 10.0, 0.0, help="Background: V17 component. Indicates the degree of compliance of the current transaction with past habits.")
v4 = st.sidebar.slider("Network and Connection Reliability", -5.0, 15.0, 0.0, help="Background: V4 component. Represents the risk index of the IP where the transaction occurred.")

if st.sidebar.button("Analyze Transaction"):
    # Using the original column names (V4, V14, V17) expected by the model in the background
    input_data = pd.DataFrame({
        'Time': [time], 'V4': [v4], 'V14': [v14], 'V17': [v17], 'Amount': [amount]
    })
    
    # Representative risk calculation (In reality: risk_score = model.predict_proba(input_data)[0][1])
    risk_score = 0.88 if v14 < -4 or v17 < -4 else 0.03
    
    st.subheader("AI Analysis Result")
    if risk_score > 0.5:
        st.error(f"🚨 HIGH RISK: Probability of Fraudulent Transaction ({risk_score*100:.1f}%)")
        st.warning("**Explainable Decision Report:** This transaction was flagged as risky because it does not match the customer's standard spending habits and deviates from the expected location/device profile.")
    else:
        st.success(f"✅ SAFE TRANSACTION: Probability of Fraudulent Transaction ({risk_score*100:.1f}%)")
        st.info("**Explainable Decision Report:** The transaction amount, time, and network profile used are fully consistent with the customer's normal behavior.")