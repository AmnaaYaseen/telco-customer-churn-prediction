# 📉 Telco Customer Churn Prediction

This is a machine learning project developed as part of the **Data Mining and Machine Learning** course. It aims to predict whether a customer will churn (i.e., leave the service) based on various features from the Telco dataset. The project covers the entire ML pipeline from data preprocessing to deployment through a user-friendly web application.

---

## 🎯 Objective

To build a predictive system that can help telecom companies identify customers likely to churn and take proactive measures to retain them, thus minimizing revenue loss.

---

## 🧠 Models Used

We trained and evaluated three machine learning models to find the best-performing one:

- ✅ **Logistic Regression** – Simple and interpretable
- ✅ **Random Forest** – Robust and handles feature interactions well
- ✅ **XGBoost** – High performance with gradient boosting

Each model was evaluated using standard metrics and tested with a custom threshold for improved recall on churn cases.

---

## 📊 Features of the Project

- 🔍 **Interactive Exploratory Data Analysis (EDA)**  
  Dynamic visualizations like histograms, pie charts, violin plots, and comparisons by churn status.

- ⚖️ **Model Comparison Page**  
  Compare different models using Accuracy, Precision, Recall, F1-Score, and AUC-ROC with performance tables.

- 🧠 **Feature Importance with SHAP**  
  Understand which features most influence customer churn with SHAP plots.

- 🤖 **Live Prediction Interface**  
  A form-based interface lets users input new customer details and get churn predictions instantly.

- 🎯 **Custom Thresholding**  
  Set a lower threshold (e.g., 0.3) to capture more churn cases and simulate profit-aware decision-making.

---

## 🧰 Tools & Libraries Used

- **Languages**: Python  
- **ML Libraries**: scikit-learn, XGBoost  
- **Visualization**: Matplotlib, Seaborn, SHAP  
- **Web App**: Streamlit  
- **Other**: NumPy, Pandas, OpenCV, Pillow

---

## 🚀 How to Run

Follow these steps to run the project locally:

```bash
# Step 1: Clone the repository
git clone https://github.com/AmnaaYaseen/telco-customer-churn-prediction.git

# Step 2: Navigate to the root project directory
cd telco_customer_churn_prediction

# Step 3 (Optional but recommended): Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # For Windows users

# Step 4: Install all required dependencies
pip install -r requirements.txt

# Step 5: Run the Streamlit app
cd churn_app
streamlit run app.py
