import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import io

# Load Data
@st.cache_data
def load_data():
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    return X_test, y_test

# Load Models
@st.cache_resource
def load_models():
    logistic = joblib.load("models/logistic.pkl")
    rf = joblib.load("models/rf.pkl")
    xgb = joblib.load("models/xgb.pkl")
    return {"Logistic Regression": logistic, "Random Forest": rf, "XGBoost": xgb}

# Evaluate Models (Uncached due to unhashable model objects)
def evaluate_models(X, y, _models_dict):
    rows = []
    for name, model in _models_dict.items():
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        rows.append({
            'Model': name,
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred),
            'Recall': recall_score(y, y_pred),
            'F1-Score': f1_score(y, y_pred),
            'AUC': roc_auc_score(y, y_proba)
        })
    return pd.DataFrame(rows)

# Main Function
def run_model_comparison():
    st.markdown("<h1 style='text-align: center;'>📊 Model Comparison</h1>", unsafe_allow_html=True)

    X, y = load_data()
    models = load_models()

    # Ensure correct feature order
    reference_model = models["Logistic Regression"]
    X = X.reindex(columns=reference_model.feature_names_in_, fill_value=0)

    # Evaluation Metrics
    metrics_df = evaluate_models(X, y, models)
    st.subheader("📋 Evaluation Metrics Table")
    st.dataframe(metrics_df, use_container_width=True)

    # Classification Reports
    st.subheader("📄 Classification Reports")
    for name, model in models.items():
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, target_names=["No", "Yes"])
        st.markdown(f"**{name}**")
        st.code(report, language='text')

    # 💼 Business Assumptions
    # --------------------------
    st.subheader("💼 Business Assumptions")
    retention_offer_cost = st.number_input(
        "💰 Cost of Retention Offer (per customer)", 
        min_value=0, max_value=1000, value=20, step=5, 
        help="Amount spent on trying to retain a customer"
    )
    churn_loss = st.number_input(
        "📉 Loss from Churned Customer", 
        min_value=0, max_value=5000, value=100, step=10, 
        help="Revenue lost when a customer churns"
    )
    st.info("These values are used to calculate profit/loss based on predicted churn outcomes.")

    # 💰 Profit-Based Evaluation
    # --------------------------
    st.subheader("💰 Profit-Based Evaluation")
    # Load true labels and predictions
    X_test = pd.read_csv("data/X_test_top.csv")
    y_test = pd.read_csv("data/y_test_top.csv").values.ravel()
    profits = {}
    for model_name, model in models.items():
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= 0.3).astype(int)  # same threshold as live prediction
        profit = 0
        for true, pred in zip(y_test, y_pred):
            if pred == 1:  # predicted churn
                if true == 1:
                    profit += churn_loss - retention_offer_cost  # prevented actual churn
                else:
                    profit -= retention_offer_cost  # wasted offer on non-churner
            # If predicted not churn, no cost/profit
        profits[model_name] = profit
    # Display profit estimates
    st.markdown("### 📊 Estimated Net Profit per Model")
    for model, profit in profits.items():
        st.write(f"**{model}**: Estimated Net Profit = `${profit:,.2f}`")

    # ROC Curve Comparison
    st.subheader("📈 ROC Curve Comparison")
    fig, ax = plt.subplots(figsize=(14.5, 6))
    for name, model in models.items():
        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate", color='white')
    ax.set_ylabel("True Positive Rate", color='white')
    ax.set_title("ROC Curve Comparison", color='white')
    ax.legend()
    ax.grid(True)
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # Confusion Matrices
    st.subheader("🧮 Confusion Matrices")
    for name, model in models.items():
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        fig, ax = plt.subplots(figsize=(2, 1), dpi=600)
        cmap_choice = 'mako'
        disp.plot(ax=ax, cmap=cmap_choice, colorbar=False)
        ax.set_title(f"{name} - Confusion Matrix", color='white', fontsize=3)
        ax.set_xlabel("Predicted Label", color='white', fontsize=3)
        ax.set_ylabel("True Label", color='white', fontsize=3)
        ax.tick_params(axis='x', colors='white', labelsize=3)
        ax.tick_params(axis='y', colors='white', labelsize=3)
        cmap = plt.cm.get_cmap(cmap_choice)
        norm = plt.Normalize(vmin=cm.min(), vmax=cm.max())
        for text_obj, val in zip(disp.text_.ravel(), cm.ravel()):
            text_obj.set_fontsize(3)
            cell_rgb = cmap(norm(val))
            luminance = (0.299 * cell_rgb[0] + 0.587 * cell_rgb[1] + 0.114 * cell_rgb[2])
            text_color = 'white' if luminance < 0.6 else 'black'
            text_obj.set_color(text_color)
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', transparent=True, dpi=600)
        buf.seek(0)
        
        # Center the image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(buf)
        plt.close(fig)



    # SHAP Summary for XGBoost
    st.subheader("🔍 SHAP Feature Importance - XGBoost")
    try:
        explainer = shap.TreeExplainer(models['XGBoost'])
        shap_values = explainer.shap_values(X)

        plt.style.use("dark_background")

        # SHAP Summary Bar Plot
        st.write("📊 SHAP Summary Plot (Bar)")
        # SHAP Summary Bar Plot with full label visible
        plt.style.use("dark_background")
        fig_bar, ax_bar = plt.subplots(figsize=(14, 6))  # Wider to prevent label cut-off
        shap.summary_plot(shap_values, X, plot_type="bar", show=False, plot_size=(14, 6))
        # Adjust spacing to prevent truncation
        plt.subplots_adjust(left=0.4, bottom=0.3, right=0.98)
        # Make background transparent
        fig_bar.patch.set_alpha(0)
        ax_bar.set_facecolor("none")
        # White labels for dark theme
        ax_bar.tick_params(colors='white')
        ax_bar.xaxis.label.set_color('white')
        ax_bar.yaxis.label.set_color('white')
        ax_bar.title.set_color('white')
        # Loop to set y-tick labels color
        for label in ax_bar.get_yticklabels():
            label.set_color('white')
        for label in ax_bar.get_xticklabels():
            label.set_color('white')
        st.pyplot(fig_bar)
        plt.clf()


        # Beeswarm plot
        st.write("🐝 SHAP Summary Plot (Beeswarm)")
        # Set figure size
        shap.summary_plot(shap_values, X, plot_size=(10, 4), show=False)
        fig_swarm = plt.gcf()
        ax_swarm = plt.gca()
        # Make background transparent
        fig_swarm.patch.set_alpha(0)
        ax_swarm.set_facecolor('none')
        ax_swarm.tick_params(colors='white', labelsize=6)  # Reduce x-axis label size
        # Reduce font size of y-axis labels (feature names)
        for label in ax_swarm.get_yticklabels():
            label.set_color('white')
            label.set_fontsize(10)  # Smaller font
        # Reduce font size of title and axis label
        ax_swarm.set_title(ax_swarm.get_title(), fontsize=8, color='white')
        ax_swarm.set_xlabel(ax_swarm.get_xlabel(), fontsize=8, color='white')
        st.pyplot(fig_swarm)
        plt.clf()
    except Exception as e:
        st.error(f"SHAP Plot Error: {e}")

    # Feature Importance
    st.subheader("📌 Feature Importance Visualizations")

    def plot_importance(title, series):
        fig, ax = plt.subplots(figsize=(14, 6))
        series.sort_values().plot(kind='barh', ax=ax, color='skyblue')
        # Increase font sizes
        ax.set_title(title, color='white', fontsize=16)
        ax.set_xlabel("Importance", color='white', fontsize=16)
        ax.set_ylabel("Features", color='white', fontsize=16)
        ax.tick_params(colors='white', labelsize=15)  # Increase tick label size
        # Optional: make background transparent
        ax.set_facecolor('none')
        fig.patch.set_alpha(0)
        st.pyplot(fig)


    st.markdown("**Logistic Regression**")
    lr = models['Logistic Regression']
    lr_importance = pd.Series(lr.coef_[0], index=X.columns)
    plot_importance("Logistic Regression Feature Weights", lr_importance)

    st.markdown("**Random Forest**")
    rf = models['Random Forest']
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    plot_importance("Random Forest Feature Importance", rf_importance)

    st.markdown("**XGBoost**")
    xgb = models['XGBoost']
    xgb_importance = pd.Series(xgb.feature_importances_, index=X.columns)
    plot_importance("XGBoost Feature Importance", xgb_importance)

    # Cross-validation
    try:
        cv_df = pd.read_csv("data/cv_scores.csv")
        st.markdown("#### 📊 Cross-validation Summary")

        for _, row in cv_df.iterrows():
            st.markdown(
                f"""
                <div style='padding: 4px 0; font-size: 15px; color: white;'>
                    <b>{row['Model']}</b>:  
                    <span style='color: lightgreen;'>Accuracy = {row['Mean Accuracy']:.4f}</span> ± 
                    <span style='color: orange;'>{row['Std Dev']:.4f}</span>
                </div>
                """, unsafe_allow_html=True
            )
    except FileNotFoundError:
        st.warning("⚠️ CV score file not found. Please ensure 'data/cv_scores.csv' exists.")

