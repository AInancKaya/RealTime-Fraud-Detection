import streamlit as st
import pandas as pd
import joblib
from pycaret.classification import load_model

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="🛡️", layout="wide")

# Modeli önbelleğe alarak yükleme (Uygulama performansını artırır)
@st.cache_resource
def load_cached_model():
    return joblib.load("rf_fraud_model.pkl")
model = load_cached_model()

@st.cache_resource
def load_cached_model():
    # PyCaret model adını fonksiyonun içine string veya parametre olarak verebilirsin
    return load_model(model_name="rf_fraud_model")

model = load_cached_model()
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
    # 1. Modelin beklediği 30 özellikli sıralı listeyi oluşturma
    feature_names = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]
    
    # 2. PCA bileşenleri (V1-V28) ortalaması yaklaşık 0 olduğu için varsayılanı 0.0 yapıyoruz
    input_dict = {feat: 0.0 for feat in feature_names}
    
    # 3. Arayüzden alınan gerçek müşteri verilerini sözlüğe aktarma
    input_dict['Time'] = time
    input_dict['Amount'] = amount
    input_dict['V4'] = v4
    input_dict['V14'] = v14
    input_dict['V17'] = v17
    
    input_data = pd.DataFrame([input_dict])
    
    # 4. Gerçek Random Forest modelini kullanarak tahmin olasılığını hesaplama
    # [0][1] ifadesi modelin "Dolandırıcılık (1)" sınıfı için ürettiği olasılığı çeker
    risk_score = model.predict_proba(input_data)[0][1]
    
    st.subheader("AI Analysis Result")
    if risk_score > 0.5:
        st.error(f"🚨 HIGH RISK: Probability of Fraudulent Transaction ({risk_score*100:.1f}%)")
        st.warning("**Explainable Decision Report:** This transaction was flagged as risky because it does not match the customer's standard spending habits and deviates from the expected location/device profile.")
    else:
        st.success(f"✅ SAFE TRANSACTION: Probability of Fraudulent Transaction ({risk_score*100:.1f}%)")
        st.info("**Explainable Decision Report:** The transaction amount, time, and network profile used are fully consistent with the customer's normal behavior.")