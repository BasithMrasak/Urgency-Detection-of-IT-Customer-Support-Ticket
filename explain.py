import shap
import joblib
import numpy as np
from predict import predict_urgency
from preprocessing import get_features, preprocess_text

# Load model
model = joblib.load("models/xgb_phase1_balanced.pkl")

# Sample ticket (the one predicted wrong)
subject = "Issue with report generation delay"
body = """The automated report generation feature is not running as scheduled since yesterday.
It is causing delays in client deliverables. Please fix this soon."""

# Preprocess and extract features
subject_clean = preprocess_text(subject)
body_clean = preprocess_text(body)
X = get_features(subject_clean, body_clean)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Display summary plot
shap.initjs()
shap.plots.waterfall(shap_values[0], show=True)
