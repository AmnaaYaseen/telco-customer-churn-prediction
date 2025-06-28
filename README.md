# 📉 Telco Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#-license)  
[![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-ff4b4b?logo=streamlit)](https://streamlit.io)

A complete machine learning pipeline to predict customer churn in the telecom industry using data-driven techniques. The project includes data preprocessing, model training, performance evaluation, feature interpretation, and deployment via an interactive web application.

---

## 🎬 Demo

Watch a 2-minute walkthrough of the project in action:  
[![Telco Churn Prediction Demo](https://img.youtube.com/vi/-znVog3LSi8/0.jpg)](https://youtu.be/-znVog3LSi8)

> 📌 *Click the thumbnail or [this link](https://youtu.be/-znVog3LSi8) to watch the demo.*

---

## 🎯 Objective

To help telecom companies identify potential churners so they can take proactive measures for customer retention and minimize revenue loss. The system predicts churn probability based on account information and service usage.

---

## 🧠 Models Used

- ✅ **Logistic Regression** – Interpretable and easy to deploy  
- ✅ **Random Forest** – Handles non-linearity and feature interactions well  
- ✅ **XGBoost** – Powerful gradient boosting algorithm for high accuracy  

> 🟢 **Random Forest is used in the live prediction interface** because it achieved the highest recall, helping to identify more churn-prone customers.

Custom thresholding (set to 0.3) was applied to improve recall and identify more potential churn cases.

---

## 📊 Key Features

- 🔍 **Interactive EDA** with histograms, pie charts, violin plots, and comparison by churn  
- ⚖️ **Model Evaluation** using Accuracy, Precision, Recall, F1-Score, and AUC-ROC  
- 🧠 **SHAP Feature Importance** for explainability  
- 🤖 **Live Prediction Interface** using user inputs  
- 🎯 **Custom Thresholding** to prioritize churn capture for profit-driven strategy  

---

## 🧰 Tech Stack

| Category         | Tools & Libraries                                     |
|------------------|--------------------------------------------------------|
| **Language**     | Python                                                 |
| **ML Libraries** | scikit-learn, XGBoost                                  |
| **EDA & Plots**  | Pandas, Seaborn, Matplotlib, SHAP                      |
| **Web App**      | Streamlit                                              |
| **Other**        | NumPy, OpenCV, Pillow                                  |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/AmnaaYaseen/telco-customer-churn-prediction.git

# 2. Navigate to the project folder
cd telco_customer_churn_prediction

# 3. (Optional) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # For Windows

# 4. Install the required dependencies
pip install -r requirements.txt

# 5. Launch the Streamlit application
cd churn_app
streamlit run app.py
```

---

## 📁 Project Structure

```
<pre>
├── churn_app/
│   ├── app_pages/
│   │   ├── about.py
│   │   ├── eda_app.py
│   │   ├── home.py
│   │   ├── live_prediction.py
│   │   └── model_comparison.py
│   ├── assets/
│   │   ├── bg.png
│   │   ├── image1.png
│   │   └── styles.css
│   ├── data/
│   │   ├── cv_scores.csv
│   │   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   │   ├── X_test.csv
│   │   ├── X_test_top.csv
│   │   ├── y_test.csv
│   │   └── y_test_top.csv
│   ├── models/
│   │   ├── columns.pkl
│   │   ├── logistic.pkl
│   │   ├── rf.pkl
│   │   └── xgb.pkl
│   ├── models_train_all_features/
│   │   ├── columns.pkl
│   │   ├── logistic.pkl
│   │   ├── rf.pkl
│   │   └── xgb.pkl
│   ├── app.py
│   ├── retrain_top_features.py
│   ├── train_all_features.ipynb
│   └── utils.py
├── requirements.txt
├── README.md
├── Project Proposal.pdf
├── .gitignore
└── LICENSE
</pre>


```

---

## 👩‍💻 Developer

**Amna Yaseen**  
[GitHub](https://github.com/AmnaaYaseen) • [LinkedIn](https://linkedin.com/in/amnaa-yaseen)

---

## 📄 License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute it with proper credit.  

© 2025 Amna Yaseen. All rights reserved.
