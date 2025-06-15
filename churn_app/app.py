import streamlit as st
import base64
import os

# ✅ Set Streamlit page config
st.set_page_config(page_title="Telco Customer Churn App", layout="wide")

# ✅ Initialize session state
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

# ✅ Import app pages
from app_pages.eda_app import run_eda
from app_pages.model_comparison import run_model_comparison
from app_pages.live_prediction import run_live_prediction
from app_pages.home import run_home
from app_pages.about import run_about

# ✅ Load and inject CSS with background image
def apply_custom_css(css_path, bg_path):
    if os.path.exists(css_path) and os.path.exists(bg_path):
        with open(bg_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode()

        with open(css_path, "r") as css_file:
            css = css_file.read().replace("{{BG_IMAGE}}", encoded_img)

        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.error("Missing CSS or background image!")

# ✅ Apply CSS
apply_custom_css("assets/styles.css", "assets/bg.png")

# ✅ App Title
st.markdown('<div class="custom-banner">📊 Telco Customer Churn Prediction App</div>', unsafe_allow_html=True)

# ✅ Navigation Buttons
st.markdown('<div class="nav-buttons-container">', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("🏠 Home", key="home_btn"):
        st.session_state.selected_page = "Home"
with col2:
    if st.button("📊 EDA", key="eda_btn"):
        st.session_state.selected_page = "EDA"
with col3:
    if st.button("⚖️ Model Comparison", key="model_btn"):
        st.session_state.selected_page = "Model Comparison"
with col4:
    if st.button("📲 Live Prediction", key="live_btn"):
        st.session_state.selected_page = "Live Prediction"
with col5:
    if st.button("ℹ️ About", key="about_btn"):
        st.session_state.selected_page = "About"
st.markdown('</div>', unsafe_allow_html=True)

# ✅ Load selected page
if st.session_state.selected_page == "Home":
    run_home()
elif st.session_state.selected_page == "EDA":
    run_eda()
elif st.session_state.selected_page == "Model Comparison":
    run_model_comparison()
elif st.session_state.selected_page == "Live Prediction":
    run_live_prediction()
elif st.session_state.selected_page == "About":
    run_about()
