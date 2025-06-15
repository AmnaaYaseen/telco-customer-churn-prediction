import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Step 1: Load original data (not encoded)
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Step 2: Basic preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Step 3: Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Step 4: One-hot encode the features
df_encoded = pd.get_dummies(df)

# Step 5: Define top features (must match column names from get_dummies)
top_features = [
    'tenure',
    'Contract_Month-to-month',
    'TotalCharges',
    'InternetService_Fiber optic',
    'MonthlyCharges',
    'Contract_Two year',
    'InternetService_DSL',
    'OnlineSecurity_No',
    'TechSupport_No',
    'PaymentMethod_Electronic check'
]

# Step 6: Split into train/test
X = df_encoded[top_features]
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ✅ Save test data with top features for app evaluation
X_test.to_csv("data/X_test_top.csv", index=False)
y_test.to_csv("data/y_test_top.csv", index=False)

# Step 7: Train models
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Step 8: Save models and columns
joblib.dump(lr, "models/logistic.pkl")
joblib.dump(rf, "models/rf.pkl")
joblib.dump(xgb, "models/xgb.pkl")
joblib.dump(top_features, "models/columns.pkl")

# ✅ Step 9: Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_cv = cross_val_score(lr, X_train, y_train, cv=cv, scoring='accuracy')
rf_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
xgb_cv = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='accuracy')

# ✅ Step 10: Create and save CV results
cv_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Mean Accuracy": [lr_cv.mean(), rf_cv.mean(), xgb_cv.mean()],
    "Std Dev": [lr_cv.std(), rf_cv.std(), xgb_cv.std()]
})

cv_df.to_csv("data/cv_scores.csv", index=False)

print("✅ Models retrained and saved successfully using top 10 features!")
print("📄 Cross-validation scores saved to data/cv_scores.csv")
