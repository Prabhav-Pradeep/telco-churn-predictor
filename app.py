import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# -------------------------------------------------------------------
# Load artifacts (these must be in the same folder as this script,
# or update the paths below)
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    pipe = joblib.load("churn_model_pipeline.joblib")
    threshold = joblib.load("best_threshold.joblib")
    template_row = joblib.load("template_row.joblib")
    return pipe, threshold, template_row

pipe, BEST_THRESHOLD, template_row = load_artifacts()

st.title("Customer Churn Predictor")
st.caption("LightGBM model trained on the IBM Telco Customer Churn dataset. "
            "Decision threshold optimized for business cost (false negatives "
            "cost more than false positives) rather than raw accuracy.")

# -------------------------------------------------------------------
# Input form
# -------------------------------------------------------------------
st.subheader("Customer details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                     value=float(tenure) * monthly_charges, step=1.0)

# -------------------------------------------------------------------
# Build a single-row DataFrame matching the training schema exactly
# -------------------------------------------------------------------
input_dict = {
    "gender": gender,
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

input_df = pd.DataFrame([input_dict])

# Reindex to match the exact column order/dtypes the pipeline was trained on
input_df = input_df.reindex(columns=template_row.columns)
for col in template_row.columns:
    input_df[col] = input_df[col].astype(template_row[col].dtype)

# -------------------------------------------------------------------
# Predict
# -------------------------------------------------------------------
if st.button("Predict churn risk", type="primary"):
    proba = pipe.predict_proba(input_df)[0, 1]
    pred_class = int(proba >= BEST_THRESHOLD)

    st.divider()
    st.subheader("Result")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Churn probability", f"{proba:.1%}")
    with c2:
        if pred_class == 1:
            st.error("⚠️ High churn risk")
        else:
            st.success("✅ Low churn risk")

    st.caption(f"Decision threshold: {BEST_THRESHOLD:.2f} "
               f"(chosen to minimize business cost — missing a churner is "
               f"assumed costlier than a false alarm)")

    # -------------------------------------------------------------------
    # SHAP explanation for this specific prediction
    # -------------------------------------------------------------------
    st.subheader("Why this prediction?")

    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["clf"]

    input_transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_transformed)

    # shap_values can be a list (per-class) or array depending on lightgbm/shap version
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        sv = shap_values[0]
        base_value = explainer.expected_value

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(
        shap.Explanation(
            values=sv,
            base_values=base_value,
            data=input_transformed[0],
            feature_names=feature_names,
        ),
        show=False,
        max_display=10,
    )
    st.pyplot(fig, bbox_inches="tight")

    st.caption("Red bars push the prediction toward higher churn risk, "
               "blue bars push it toward lower churn risk.")