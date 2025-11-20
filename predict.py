"""
predict.py
Load the trained model and provide prediction functionality
"""

import joblib
import os
from preprocessing import get_features, preprocess_text
import numpy as np

# Load the model
MODEL_PATH = os.path.join("models", "xgboost_model.pkl")
model = joblib.load(MODEL_PATH)
le = joblib.load("models/label_encoder.pkl")
tfidf_subject = joblib.load("models/tfidf_subject.pkl")
tfidf_body = joblib.load("models/tfidf_body.pkl")

def predict_urgency(subject, body):
    """
    Predict urgency of a support ticket.

    Args:
        subject (str): Ticket subject
        body (str): Ticket body

    Returns:
        dict: { 'class': 'High'/'Medium'/'Low', 'probabilities': [prob_H, prob_M, prob_L] }
    """

    # Preprocess input
    subject_clean = preprocess_text(subject)
    body_clean = preprocess_text(body)

    # Extract features
    X = get_features(subject_clean, body_clean,tfidf_subject, tfidf_body)
    X.sum()          # how many non-zero TF-IDF features
    X_nonzero = np.nonzero(X)[1]
    print("Number of non-zero features:", len(X_nonzero))

    # Get prediction
    pred_class_idx = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    

    # Map class index to label (adjust according to your training)
    pred_class = le.inverse_transform([pred_class_idx])[0]
    print(pred_class,pred_proba)

    return {
        "class": pred_class,
        "probabilities": pred_proba.tolist()
    }


