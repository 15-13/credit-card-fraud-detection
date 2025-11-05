import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# Set page config (must be first Streamlit command)
# ------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

# ------------------------------
# Load Model and Scaler
# ------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load('xgboost_model.pkl')   # or random_forest_model.pkl
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title(" Credit Card Fraud Detection")
st.markdown("""
This interactive app uses a trained XGBoost Machine Learning model  
to predict whether a credit card transaction is fraudulent or legitimate.
""")

st.divider()
st.header("Enter Transaction Details")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input(" Transaction Amount (₹)", min_value=0.0, value=500.0, step=10.0)
with col2:
    time = st.number_input("Time (in seconds since first transaction)", min_value=0.0, value=100000.0, step=1000.0)

st.caption("PCA-based features (V1–V28) are filled with neutral defaults for demo purposes.")

# ------------------------------
# Prepare Input Data
# ------------------------------
input_data = pd.DataFrame(np.zeros((1, 30)), columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])
input_data['Time'] = time
input_data['Amount'] = amount

# Scale Time and Amount
input_data[['Time', 'Amount']] = scaler.transform(input_data[['Time', 'Amount']])

# ------------------------------
# Prediction Section
# ------------------------------
if st.button(" Predict Fraud Risk"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    st.divider()
    st.subheader(" Prediction Result")

    # Risk Level Logic
    if prob < 0.30:
        risk_level = "Low Risk"
        color = "🟢"
        bar_color = "green"
    elif prob < 0.70:
        risk_level = "Moderate Risk"
        color = "🟡"
        bar_color = "orange"
    else:
        risk_level = "High Risk"
        color = "🔴"
        bar_color = "red"

    # Fraud or Legit Text
    result = " Fraudulent Transaction Detected!" if prediction == 1 else "✅ Legitimate Transaction"
    
    # Display Main Result
    st.markdown(f"### {result}")
    st.markdown(f"**Confidence:** `{prob:.2%}`")
    st.markdown(f"**Risk Level:** {color} {risk_level}")

    # Progress Bar Visualization
    st.progress(float(prob))

    # Color-coded visual bar using markdown (optional for more style)
    st.markdown(f"""
    <div style="padding:10px; border-radius:10px; background-color:#f9f9f9; border:1px solid #ddd;">
        <b>Fraud Probability:</b> {prob:.2%} <br>
        <b>Risk Level:</b> <span style="color:{bar_color};">{risk_level}</span> <br>
        <b>Transaction Amount:</b> ₹{amount:.2f}
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.divider()
st.markdown("""
**🧠 Model:** XGBoost | **Developer:** Harshil Undhad  
📘 [GitHub Repository](https://github.com/harshilundhad/credit-card-fraud-detection)  
💡 *End-to-end ML pipeline for fraud detection with Streamlit deployment.*
""")
