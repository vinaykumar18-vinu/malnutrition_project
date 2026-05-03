from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Get base directory (important for deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and encoder safely
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# About Route
@app.route('/about')
def about():
    return render_template('about.html')

# Explore Route
@app.route('/explore')
def explore():
    return render_template('explore.html')

# Predict Page Route
@app.route('/predict')
def predict():
    return render_template('predict.html')

# Prediction Logic
@app.route('/result', methods=['POST'])
def result():
    try:
        # Get input values
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        muac = float(request.form['muac'])

        # Basic validation
        if height <= 0 or weight <= 0:
            return "Invalid input: Height and Weight must be positive."

        # BMI calculation
        bmi = weight / ((height / 100) ** 2)

        # Prepare data (IMPORTANT: order must match training)
        data = np.array([[age, weight, height, muac, bmi]])

        # Predict
        prediction = model.predict(data)
        result_label = encoder.inverse_transform(prediction)[0]

        # Send result to HTML
        return render_template(
            'result.html',
            prediction=result_label,
            bmi=round(bmi, 2)
        )

    except Exception as e:
        return f"Error: {str(e)}"

# Run app locally (Render ignores this and uses gunicorn)
if __name__ == "__main__":
    app.run()