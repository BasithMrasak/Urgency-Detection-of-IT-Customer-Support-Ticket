"""
preprocessing.py
Feature extraction and preprocessing utilities for support ticket urgency prediction
"""

import re
import numpy as np
import pandas as pd


def get_features(subject, body, tfidf_subject, tfidf_body, language='en'):
    subject = str(subject) if subject else "no subject"
    body = str(body) if body else ""

    # TF-IDF features
    from scipy.sparse import hstack
    X_subj = tfidf_subject.transform([subject])
    X_body = tfidf_body.transform([body])
    X = hstack([X_subj, X_body]).toarray()


    return X

def validate_input(subject, body):
    if not subject or not subject.strip():
        return False, "Subject cannot be empty"
    if not body or not body.strip():
        return False, "Body cannot be empty"
    if len(subject) > 500:
        return False, "Subject is too long (max 500 chars)"
    if len(body) > 5000:
        return False, "Body is too long (max 5000 chars)"
    return True, None

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text
