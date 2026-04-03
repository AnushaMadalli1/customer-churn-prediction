import streamlit as st
import requests

# 🔥 Page config
st.set_page_config(page_title="Churn Predictor", layout="wide")

# 🎨 Custom CSS
st.markdown("""
<style>
body {
    background-color: #0E1117;
}

.main-title {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    color: #ffffff;
    animation: fadeIn 1.5s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.5);
    margin-top: 20px;
}

.stButton>button {
    background: linear-gradient(90deg, #ff4b2b, #ff416c);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
}
</style>
""", unsafe_allow_html=True)

# 🏷 Title
st.markdown('<div class="main-title">🚀 Customer Churn Predictor</div>', unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("🎛 Input Features")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1000.0)

# 🧠 Reason function
def get_reason(tenure, monthly, total):
    reasons = []

    if tenure < 12:
        reasons.append("Low tenure (new customer, higher churn risk)")

    if monthly > 80:
        reasons.append("High monthly charges may cause dissatisfaction")

    if total < 500:
        reasons.append("Low engagement (customer not spending much)")

    if tenure > 36:
        reasons.append("Long-term customer (loyal, less likely to churn)")

    if monthly < 50:
        reasons.append("Affordable monthly charges (good retention factor)")

    # 🔥 NEW: default fallback
    if not reasons:
        reasons.append("Moderate usage pattern, model detected churn risk based on combined factors")

    return reasons

# Prepare data
data = [{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}]

# Layout
col1, col2 = st.columns([1, 2])

# 🔮 Predict Button
with col1:
    if st.button("🔮 Predict Churn"):
        try:
            response = requests.post(
                "http://localhost:5000/predict",
                json={"data": data},
                timeout=10
            )

            response.raise_for_status()
            result = response.json()

            preds = result["predictions"]
            probs = result["probabilities"]

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)

                st.success("Prediction from API")

                st.metric("Churn", "Yes" if preds[0] == 1 else "No")
                st.metric("Churn Probability", f"{probs[0]:.1%}")

                # 🧠 Reasons
                reasons = get_reason(tenure, monthly_charges, total_charges)

                st.subheader("🧠 Reason for Prediction")
                if preds[0] == 1:
                    st.error("Customer likely to churn due to:")
                else:
                    st.success("Customer likely to stay because:")

                for r in reasons:
                    st.write(f"👉 {r}")

                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

# 🧪 Test API
with st.expander("🧪 Test API"):
    if st.button("⚡ Call API"):
        try:
            response = requests.post(
                "http://localhost:5000/predict",
                json={"data": data},
                timeout=10
            )

            response.raise_for_status()
            st.json(response.json())

        except Exception as e:
            st.error(f"Error: {e}")