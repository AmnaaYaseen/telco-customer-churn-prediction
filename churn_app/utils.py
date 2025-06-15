import joblib
import pandas as pd

# Load trained models
def load_model(model_dir="models/"):
    return {
        "Logistic Regression": joblib.load(model_dir + "logistic.pkl"),
        "Random Forest": joblib.load(model_dir + "rf.pkl"),
        "XGBoost": joblib.load(model_dir + "xgb.pkl")
    }

# Preprocess input for live prediction
def preprocess_input(df):
    required_columns = joblib.load("models/columns.pkl")
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=required_columns, fill_value=0)
    return df_encoded

# Run prediction
def run_prediction(models, df_proc):
    model = models["Random Forest"]  # Or use a selector if needed
    prob = model.predict_proba(df_proc)[:, 1][0]
    label = "Churn" if prob > 0.3 else "Not Churn"
    return {"label": label, "prob": prob}
