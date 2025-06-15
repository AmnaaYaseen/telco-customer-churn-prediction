import streamlit as st
from PIL import Image
from io import BytesIO
import base64

def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def run_home():
    st.markdown(
        "<h1 style='text-align: center; color: white;'>🏠 Welcome to the Telco Customer Churn Prediction App</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='color: white;'>🎯 What This App Does</h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size: 19px;'>This app predicts whether a customer is likely to <strong>churn (leave the service)</strong> based on their behavior, usage, and subscription information.</p>",
        unsafe_allow_html=True
    )

    # 👉 Section: Features + Image side by side
    col1, col2 = st.columns([1.5, 1.5], gap="large")
    
    with col1:
        st.markdown("<h3 style='margin-bottom: 10px;'>✨ Features Included</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ul style='font-size:19px; line-height:1.7'>
            <li><b>EDA</b>: Understand the data visually</li>
            <li><b>Model Comparison</b>: Evaluate different ML models</li>
            <li><b>Live Prediction</b>: Try out predictions with real-time input</li>
            <li><b>About</b>: Dive deep into the project</li>
        </ul>
        """, unsafe_allow_html=True)

        st.markdown("<h3 style='margin-top: 30px;'>🤖 Models We Trained</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ul style='font-size:19px; line-height:1.7'>
            <li>Logistic Regression</li>
            <li>Random Forest</li>
            <li>XGBoost (Extreme Gradient Boosting)</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='margin-top: 30px;'>🧪 Try it Out!</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style='font-size: 19px; line-height: 1.6'>
            Head over to the <strong>Live Prediction</strong> page to test your own inputs and see the churn prediction in action — powered by machine learning! 🔮
            </p>
            """,
            unsafe_allow_html=True
        )

    with col2:
        try:
            image = Image.open("assets/image1.png").convert("RGBA")
            original_width, original_height = image.size  # Get original dimensions
            desired_height = 2200  # The height you want
            # Calculate the new width to maintain the aspect ratio
            aspect_ratio = original_width / original_height
            desired_width = int(desired_height * aspect_ratio)
            resized_image = image.resize((desired_width, desired_height), Image.Resampling.LANCZOS)
            encoded_img = image_to_base64(resized_image)
            st.markdown(
                f"""
                <img src="data:image/png;base64,{encoded_img}"
                     style="max-width:100%; height:auto; display:block; margin:auto;" />
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"⚠️ Could not load or process the image: {e}")

    # Optional horizontal rule
    st.markdown("<hr>", unsafe_allow_html=True)

    # fun fact
    st.markdown(
        """
        <p style='font-size: 18px; font-style: italic; text-align: center; color: #ccc;'>
         <strong> 🐣 </strong> Predicting churn is like finding the friend who's about to leave a party — this app helps you talk them into staying! 🎉
        </p>
        """,
        unsafe_allow_html=True
    )