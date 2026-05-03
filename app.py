from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and encoder
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")


# Home → Predict Page
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


# Predict Route
@app.route('/predict')
def predict():
    return render_template('predict.html')


# Prediction Route
@app.route('/result', methods=['POST'])
def result():
    try:
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        muac = float(request.form['muac'])

        # Validation
        if height <= 0 or weight <= 0:
            return "Invalid input"

        # BMI Calculation
        bmi = weight / ((height / 100) ** 2)

        # Model Input (IMPORTANT ORDER)
        data = np.array([[age, weight, height, muac, bmi]])

        pred = model.predict(data)
        result_label = encoder.inverse_transform(pred)[0]

        return render_template(
            'result.html',
            prediction=result_label,
            bmi=round(bmi, 2)
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)