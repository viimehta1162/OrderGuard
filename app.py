import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OrderGuard", page_icon="📦", layout="centered")

st.markdown("""
    <style>
    h1 {
        color: #ff9900;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        text-align: center;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>📦 OrderGuard: Order Cancellation Predictor</h1>",
            unsafe_allow_html=True)
st.write("Estimate how likely an order is to be cancelled, using a Random Forest "
         "model trained on Amazon India sales data.")


@st.cache_resource
def load_files():
    model = joblib.load("cancel_model.pkl")
    with open("options.json") as f:
        options = json.load(f)
    with open("metrics.json") as f:
        metrics = json.load(f)
    return model, options, metrics


model, options, metrics = load_files()

st.subheader("🛍️ Order details")
col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Order Amount (INR)", min_value=0.0, max_value=20000.0,
                             value=float(round(options["amount_median"])), step=50.0)
    category = st.selectbox("Category", options["Category"])
    size = st.selectbox("Size", options["Size"])
with col2:
    fulfilment = st.selectbox("Fulfilment", options["Fulfilment"])
    service_level = st.selectbox("Service Level", options["ship-service-level"])
    channel = st.selectbox("Sales Channel", options["Sales Channel"])

st.subheader("📍 Destination")
col3, col4 = st.columns(2)
with col3:
    ship_state = st.selectbox("Shipping State", options["ship-state"])
with col4:
    b2b = st.selectbox("Is this a B2B sale?", ["No", "Yes"])

threshold = st.slider(
    "Decision threshold (lower = catches more cancellations, but more false alarms)",
    0.10, 0.90, 0.50, 0.05)

if st.button("🔮 Predict Cancellation Risk", use_container_width=True):
    row = pd.DataFrame([{
        "Fulfilment": fulfilment,
        "Sales Channel": channel,
        "ship-service-level": service_level,
        "Category": category,
        "Size": size,
        "ship-state": ship_state,
        "B2B": 1 if b2b == "Yes" else 0,
        "Amount": amount,
    }])
    proba = float(model.predict_proba(row)[0, 1])

    if proba >= threshold:
        st.markdown(f"""
            <div class='prediction-box' style='background-color:#ffe6e6; color:#cc0000; border:2px solid #ff9999;'>
                ⚠️ High risk of cancellation<br>
                <span style='font-size:15px;'>Cancellation risk: {proba * 100:.1f}%</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='prediction-box' style='background-color:#e6ffe6; color:#008000; border:2px solid #99ff99;'>
                ✅ Low risk: order likely to be fulfilled<br>
                <span style='font-size:15px;'>Cancellation risk: {proba * 100:.1f}%</span>
            </div>
        """, unsafe_allow_html=True)

st.divider()
st.subheader("📊 Model performance (test data)")
cols = st.columns(len(metrics))
for c, (name, val) in zip(cols, metrics.items()):
    c.metric(name, val)
st.caption("Random Forest trained on Amazon India sales data (Mar–Jun 2022). "
           "About 14% of orders in this data were cancelled.")
