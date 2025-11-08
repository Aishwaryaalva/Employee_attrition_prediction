# app.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ----------------------------
# Train & Save Model (runs once on first load)
# ----------------------------
MODEL_PATH = "hr_model.pkl"
ENC_SALARY = "le_salary.pkl"
ENC_DEPT = "le_dept.pkl"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Training model... (takes ~10 seconds)"):
        # Load data — your CSV has headers!
        df = pd.read_csv("HR_comma_sep.csv")

        # Encode categorical columns
        le_dept = LabelEncoder()
        le_salary = LabelEncoder()
        df["Department"] = le_dept.fit_transform(df["Department"])
        df["salary"] = le_salary.fit_transform(df["salary"])

        # Features and target
        X = df.drop("left", axis=1)
        y = df["left"]

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save model and encoders
        joblib.dump(model, MODEL_PATH)
        joblib.dump(le_dept, ENC_DEPT)
        joblib.dump(le_salary, ENC_SALARY)
        st.success("✅ Model trained and saved!")

# ----------------------------
# Load Model & Encoders
# ----------------------------
model = joblib.load(MODEL_PATH)
le_dept = joblib.load(ENC_DEPT)
le_salary = joblib.load(ENC_SALARY)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="HR Attrition Predictor", layout="centered")
st.title("👥 HR Employee Attrition Predictor")
st.markdown("Predict whether an employee is likely to **leave** the company.")

# Input form
col1, col2 = st.columns(2)

with col1:
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.6)
    evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.7)
    projects = st.number_input("Number of Projects", min_value=2, max_value=7, value=4)
    hours = st.number_input("Avg. Monthly Hours", min_value=90, max_value=310, value=200)
    years = st.number_input("Years at Company", min_value=1, max_value=10, value=3)

with col2:
    accident = st.selectbox("Work Accident?", ["No", "Yes"])
    promotion = st.selectbox("Promotion in Last 5 Years?", ["No", "Yes"])
    dept = st.selectbox("Department", le_dept.classes_)
    salary = st.selectbox("Salary Level", ["low", "medium", "high"])

# Encode inputs
accident_val = 1 if accident == "Yes" else 0
promotion_val = 1 if promotion == "Yes" else 0
dept_encoded = le_dept.transform([dept])[0]
salary_encoded = le_salary.transform([salary])[0]

# Create input DataFrame (same order as training)
input_df = pd.DataFrame([[
    satisfaction, evaluation, projects, hours, years,
    accident_val, promotion_val, dept_encoded, salary_encoded
]], columns=[
    "satisfaction_level", "last_evaluation", "number_project",
    "average_montly_hours", "time_spend_company", "Work_accident",
    "promotion_last_5years", "Department", "salary"
])

# Predict
if st.button("🔍 Predict Attrition Risk"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # Probability of leaving

    if pred == 1:
        st.error(f"⚠️ **High Risk**: Employee is likely to **leave** (Probability: {prob:.2%})")
    else:
        st.success(f"✅ **Low Risk**: Employee is likely to **stay** (Leaving probability: {prob:.2%})")