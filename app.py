from flask import Flask, render_template, request, redirect, url_for
from predict import predict_urgency
from preprocessing import validate_input, preprocess_text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Home Page
@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        subject = request.form.get("subject", "")
        body = request.form.get("body", "")

        # Validate input
        is_valid, error_msg = validate_input(subject, body)
        if not is_valid:
            return render_template("home.html", error=error_msg, subject=subject, body=body)

        # Preprocess input
        subject_clean = preprocess_text(subject)
        body_clean = preprocess_text(body)

        # Redirect to prediction route
        return redirect(url_for("predict", subject=subject_clean, body=body_clean))

    return render_template("home.html")


# Prediction Page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Handle POST request (from form submission)
        subject = request.form.get("subject", "")
        body = request.form.get("body", "")
    else:
        # Handle GET request (from redirect)
        subject = request.args.get("subject", "")
        body = request.args.get("body", "")

    # Get prediction
    result = predict_urgency(subject, body)
    prediction = result['class']
    probabilities = result['probabilities']

    return render_template(
        "result.html",
        subject=subject,
        body=body,
        prediction=prediction,
        probabilities=probabilities
    )

if __name__ == "__main__":
    app.run(debug=True)
