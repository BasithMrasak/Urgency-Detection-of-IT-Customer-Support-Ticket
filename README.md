# Urgency Detection of IT Customer Support Tickets

A machine learning web application that automatically classifies IT customer support tickets into **High**, **Medium**, or **Low** urgency levels — helping support teams prioritize and respond faster.

## Overview

Support teams often receive hundreds of tickets daily. Manual triage is slow and error-prone. This system uses an XGBoost classifier trained on ticket subject and body text (via TF-IDF features) to instantly predict urgency and display class probabilities.

## Tech Stack

- **ML Model:** XGBoost + TF-IDF (separate vectorizers for subject and body)
- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS (Jinja2 templates)
- **Serialization:** joblib (model, label encoder, vectorizers)

## Project Structure
```
├── app.py               # Flask routes (home, predict)
├── predict.py           # Model loading and inference
├── preprocessing.py     # Text cleaning, validation, feature extraction
├── explain.py           # (Explainability utilities)
├── test_prediction.py   # Standalone test script
├── models/
│   ├── xgboost_model.pkl
│   ├── label_encoder.pkl
│   ├── tfidf_subject.pkl
│   └── tfidf_body.pkl
├── templates/           # HTML templates
└── static/css/          # Stylesheets
```
## How It Works

1. User submits a ticket subject and body via the web form
2. Input is validated and whitespace-normalized
3. Subject and body are independently vectorized using pre-fitted TF-IDF vectorizers
4. Feature vectors are concatenated and passed to the XGBoost model
5. The app returns the predicted urgency class along with probability scores for all three classes

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```


## Model Performance

- **Accuracy:** ~91.5% on IT support ticket dataset
- **Classes:** High | Medium | Low

## Author

[BasithMrasak](https://github.com/BasithMrasak)
