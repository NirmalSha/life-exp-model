import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Life Expectancy Predictor",
    layout="centered"
)

st.title("🧬 Life Expectancy Prediction")
st.markdown("Fill patient health details to predict life expectancy")

# -------------------------------
# Load MLflow Model (alias-based, MLflow 3.x)
# -------------------------------
@st.cache_resource
def load_model_with_version():
    mlflow.set_tracking_uri("http://localhost:5000")

    client = MlflowClient()
    model_name = "life-expectancy-model"
    alias = "production"

    try:
        # Preferred: load via alias (MLflow 3.x best practice)
        mv = client.get_model_version_by_alias(model_name, alias)
        model_version = mv.version
        model_uri = f"models:/{model_name}@{alias}"

    except Exception:
        # Fallback: latest version (never crash UI)
        latest = client.get_latest_versions(model_name)
        if not latest:
            raise RuntimeError("No model versions found in registry")

        model_version = latest[0].version
        model_uri = f"models:/{model_name}/{model_version}"

    model = mlflow.sklearn.load_model(model_uri)
    return model, model_version




# ✅ CALL FUNCTION ONLY AFTER IT IS DEFINED
model, model_version = load_model_with_version()

# Show model info
st.info(
    f"🧠 Using MLflow Model: life-expectancy-model | "
    f"Alias: production | Version: v{model_version}"
)

# -------------------------------
# Input Form
# -------------------------------
with st.form("prediction_form"):

    age = st.slider("Age", 18, 90, 40)
    gender = st.selectbox("Gender", ["male", "female"])
    bmi = st.slider("BMI", 15.0, 40.0, 25.0)

    smoker = st.selectbox("Smoker", [0, 1])
    alcohol = st.selectbox("Alcohol Consumption", [0, 1])

    diabetes = st.selectbox("Diabetes", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    cancer = st.selectbox("Cancer", [0, 1])
    hypertension = st.selectbox("Hypertension", [0, 1])
    asthma = st.selectbox("Asthma", [0, 1])

    region = st.selectbox(
        "Region",
        ["asia", "europe", "africa", "americas"]
    )

    healthcare_access = st.selectbox(
        "Healthcare Access",
        ["poor", "average", "good"]
    )

    submit = st.form_submit_button("🔮 Predict Life Expectancy")

# -------------------------------
# Prediction
# -------------------------------
if submit:

    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "smoker": smoker,
        "alcohol": alcohol,
        "diabetes": diabetes,
        "heart_disease": heart_disease,
        "cancer": cancer,
        "hypertension": hypertension,
        "asthma": asthma,
        "region": region,
        "healthcare_access": healthcare_access
    }])

    prediction = model.predict(input_df)[0]

    st.success(
        f"🧠 Predicted Life Expectancy: **{round(prediction, 2)} years**"
    )

    st.markdown("---")
    st.caption(
        f"Powered by MLflow | Alias: production | Version: v{model_version}"
    )

