import streamlit as st
import pandas as pd
from utils import load_model, preprocess_input, run_prediction

def run_live_prediction():
    models = load_model()

    # Heading
    st.markdown("<h1 style='text-align: center;'>📲 Live Telco Customer Churn Prediction</h1>", unsafe_allow_html=True)

    # Scoped Button Styling - only applies to this form
    st.markdown("""
        <style>
        div[data-testid="stFormSubmitButton"] button {
            background-color: #0d6efd;
            color: white;
            width: 60%;
            height: 3em;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin: auto;
            display: block;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.form("customer_form"):
        st.subheader("Enter Customer Details")

        left, right = st.columns(2)

        with left:
            tenure = st.number_input("Tenure (months)", min_value=0)
            Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
            OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No'])
            PaymentMethod = st.selectbox("Payment Method", [
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ])

        with right:
            TotalCharges = st.number_input("Total Charges", min_value=0.0)
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
            InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            TechSupport = st.selectbox("Tech Support", ['Yes', 'No'])

        # Submit button — styled and centered
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_dict = {
            "tenure": tenure,
            "TotalCharges": TotalCharges,
            "MonthlyCharges": MonthlyCharges,
            "Contract": Contract,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "TechSupport": TechSupport,
            "PaymentMethod": PaymentMethod
        }

        input_df = pd.DataFrame([input_dict])
        processed_df = preprocess_input(input_df)

        result = run_prediction(models, processed_df)

        st.success(f"🔍 Prediction: **{result['label']}**")
        st.info(f"🔢 Probability of churn: **{result['prob']:.2f}**")
