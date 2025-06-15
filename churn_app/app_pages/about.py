import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_about():
    st.markdown("<h1 style='text-align: center; color: white;'>📘 About the Project</h1>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: white;'>📌 Project Overview</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:19px; line-height:1.6'>
    This project was built to <strong>analyze, predict, and explain customer churn</strong> using machine learning. We used the popular 
    <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn" target="_blank">Telco Customer Churn Dataset</a> from Kaggle.<br><br>
    After identifying the most important features using full-feature models, we retrained the models using the top 10 features 
    for better generalization and faster prediction. SHAP and feature importance plots were then generated using the retrained models 
    to reflect accurate explanations.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color: white;'>🔍 Why Random Forest for Prediction?</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:19px; line-height:1.6'>
    Although all models had their strengths, we chose <strong>Random Forest</strong> for live prediction because:</p>
    <ul style='font-size:19px; line-height:1.7'>
        <li>✅ It achieved <strong>the highest Recall (0.73)</strong> among all models.</li>
        <li>✅ It also maintained a <strong>strong F1-score (0.59)</strong>.</li>
    </ul>
    <p style='font-size:19px; line-height:1.6'>
    In churn problems:
    <br>🔁 <strong>Recall</strong> is critical — we want to <strong>catch as many churners as possible</strong>.<br>
    ⚖️ <strong>F1-Score</strong> balances precision and recall — helping avoid false alarms too often.
    </p>
    """, unsafe_allow_html=True)

    
    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("<h3 style='color: white;'>📊 Model Comparison</h3>", unsafe_allow_html=True)

        st.markdown("""
        <table style='font-size:18px;'>
        <thead><tr><th>Model</th><th>Recall (Yes)</th><th>F1-Score (Yes)</th></tr></thead>
        <tbody>
        <tr><td>Logistic Regression</td><td>0.63</td><td>0.60</td></tr>
        <tr><td><strong>Random Forest</strong></td><td><strong>0.73</strong></td><td><strong>0.59</strong></td></tr>
        <tr><td>XGBoost</td><td>0.35</td><td>0.45</td></tr>
        </tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("<h3 style='color: white;'>🎯 Why Use a Threshold of 0.3?</h3>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-size:19px; line-height:1.6'>
        Instead of the usual <strong>0.5</strong> threshold, we chose <strong>0.3</strong> for live prediction because:<br>
        - It makes the model <strong>more sensitive</strong> to churn.<br>
        - Helps <strong>catch more churners</strong>, even those with subtle signals — increasing <strong>recall</strong>.<br><br>
        In business: it's better to <strong>over-warn than under-warn</strong> when the cost of losing a customer is high!
        </p>
        """, unsafe_allow_html=True)

    with col2:
        model_names = ["Logistic Regression", "Random Forest", "XGBoost"]
        recall_scores = [0.63, 0.73, 0.35]
        f1_scores = [0.60, 0.59, 0.45]

        fig, ax = plt.subplots(figsize=(5, 2.5), facecolor='none')  # Transparent figure
        ax.set_facecolor('none')  # Transparent axes

        bar_width = 0.35
        index = range(len(model_names))

        ax.barh([i + bar_width for i in index], recall_scores, bar_width, label='Recall', color='skyblue')
        ax.barh(index, f1_scores, bar_width, label='F1-Score', color='lightgreen')

        ax.set(yticks=[i + bar_width / 2 for i in index], yticklabels=model_names)
        ax.invert_yaxis()
        ax.set_xlabel('Score', color='white')
        ax.set_title('Performance by Model', fontsize=10, color='white')
        ax.tick_params(colors='white')
        sns.despine()

        # Set legend and make text white
        legend = ax.legend(fontsize=8, facecolor='none', edgecolor='none')
        for text in legend.get_texts():
            text.set_color('white')

        st.pyplot(fig)




    st.markdown("<h3 style='color: white;'>🧪 How We Handled Imbalance</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:19px; line-height:1.6'>
    The dataset had more non-churners than churners — a common issue. We applied 
    <strong>SMOTE (Synthetic Minority Oversampling Technique)</strong> during training to balance the classes 
    and reduce model bias.
    </p>
    """, unsafe_allow_html=True)


    col3, col4 = st.columns([1.1, 1])

    with col3:
        st.markdown("<h3 style='color: white;'>🌟 Top 10 Features Used for Retraining</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ul style='font-size:19px; line-height:1.7'>
            <li><code>tenure</code></li>
            <li><code>Contract_Month-to-month</code></li>
            <li><code>TotalCharges</code></li>
            <li><code>InternetService_Fiber optic</code></li>
            <li><code>MonthlyCharges</code></li>
            <li><code>Contract_Two year</code></li>
            <li><code>InternetService_DSL</code></li>
            <li><code>OnlineSecurity_No</code></li>
            <li><code>TechSupport_No</code></li>
            <li><code>PaymentMethod_Electronic check</code></li>
        </ul>
        """, unsafe_allow_html=True)

    with col4:
        # st.markdown("<h6 style='color: white;'>🔍 Feature Correlation Heatmap</h6>", unsafe_allow_html=True)


        df = pd.read_csv("data/X_test_top.csv")
        y = pd.read_csv("data/y_test_top.csv")

        # Combine to form a full DataFrame for correlation
        df['Churn'] = y

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='none')
        ax.set_facecolor('none')

        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax, fmt=".2f", cbar=True)

        # Style labels
        ax.set_title("Correlation Between Top Features & Churn", fontsize=12, color='white')
        ax.tick_params(colors='white')
        colorbar = ax.collections[0].colorbar
        colorbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(colorbar.ax.axes, 'yticklabels'), color='white')

        st.pyplot(fig)


    # 🧠 Final Thought
    st.markdown("<h3 style='color: white;'>🧠 Final Thought</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:19px; line-height:1.6'>
    This project is more than just code — it's about <strong>using data to understand people</strong> 
    and make smarter business decisions.
    </p>
    """, unsafe_allow_html=True)

    # 🤔 Why did the customer churn?
    st.markdown("<h3 style='color: white;'>🤔 Why did the customer churn?</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:19px; line-height:1.6'>
    👉 Because no one showed them this awesome app! 😜
    </p>
    """, unsafe_allow_html=True)

    # Divider
    st.markdown("<hr style='border-top: 1px solid #999;'>", unsafe_allow_html=True)

    # Thanks Message
    st.markdown("""
    <p style='font-size:18px; font-style: italic; text-align: center; color: #ccc;'>
    Thanks for visiting — now go catch those churners before they ghost you! 👻📞
    </p>
    """, unsafe_allow_html=True)
