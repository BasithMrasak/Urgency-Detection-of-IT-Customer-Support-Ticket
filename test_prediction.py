import joblib
from preprocessing import get_features  # make sure get_features is updated for TF-IDF only

# 1️⃣ Load model and vectorizers
model = joblib.load("models/xgboost_model.pkl")
tfidf_subject = joblib.load("models/tfidf_subject.pkl")
tfidf_body = joblib.load("models/tfidf_body.pkl")
le = joblib.load("models/label_encoder.pkl")

# 2️⃣ Prepare test ticket
subject = "Server down – immediate assistance"
body = "Users cannot access the application. Please resolve ASAP."

# 3️⃣ Extract features
X = get_features(subject, body, tfidf_subject, tfidf_body)

print("Shape of feature vector:", X.shape)
print("Non-zero features:", X.sum())

# 4️⃣ Predict class
pred_class_idx = model.predict(X)[0]
pred_class = le.inverse_transform([pred_class_idx])[0]

print("Predicted class:", pred_class)
