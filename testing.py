from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved Meta-Classifier and training feature list
bundle = joblib.load('depression_meta_model.pkl')
model = bundle['model']
training_columns = bundle['columns']

# Preprocessing mappings matching Task 2
BINARY_MAP = {'Yes': 1, 'No': 0}
SLEEP_MAP = {
    'Less than 5 hours': 1,
    '5-6 hours': 2,
    '7-8 hours': 3,
    'More than 8 hours': 4
}
DIET_MAP = {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3}

def run_prediction(data):
    input_df = pd.DataFrame([data])

    # Apply exact ordinal/binary encodings
    input_df['Have you ever had suicidal thoughts ?'] = input_df['Have you ever had suicidal thoughts ?'].map(BINARY_MAP)
    input_df['Family History of Mental Illness'] = input_df['Family History of Mental Illness'].map(BINARY_MAP)
    input_df['Sleep Duration'] = input_df['Sleep Duration'].map(SLEEP_MAP)
    input_df['Dietary Habits'] = input_df['Dietary Habits'].map(DIET_MAP)

    # One-hot encode and align columns with X_train
    input_encoded = pd.get_dummies(input_df)
    input_aligned = input_encoded.reindex(columns=training_columns, fill_value=0)

    prediction = int(model.predict(input_aligned)[0])
    probability = float(model.predict_proba(input_aligned)[0][1])
    return prediction, probability

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Depression Risk Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f7f6; margin: 0; padding: 30px; }
        .container { max-width: 650px; margin: auto; background: white; padding: 25px 35px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .form-group { display: flex; flex-direction: column; }
        label { font-weight: bold; font-size: 13px; margin-bottom: 5px; color: #34495e; }
        input, select { padding: 8px; border: 1px solid #ccc; border-radius: 5px; font-size: 14px; }
        button { width: 100%; margin-top: 20px; padding: 12px; background: #2980b9; color: white; border: none; border-radius: 5px; font-size: 16px; font-weight: bold; cursor: pointer; }
        button:hover { background: #1f6391; }
        .result { margin-top: 25px; padding: 15px; border-radius: 6px; text-align: center; font-size: 18px; }
        .high-risk { background: #fdecea; color: #c0392b; border: 1px solid #e74c3c; }
        .low-risk { background: #eafaf1; color: #1e8449; border: 1px solid #2ecc71; }
    </style>
</head>
<body>
<div class="container">
    <h2>Student Depression Prediction (Meta-Classifier)</h2>
    <form method="POST" action="/">
        <div class="grid">
            <div class="form-group">
                <label>Gender</label>
                <select name="Gender">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>
            <div class="form-group">
                <label>Age</label>
                <input type="number" name="Age" min="15" max="60" value="21" required>
            </div>
            <div class="form-group">
                <label>City</label>
                <input type="text" name="City" value="Visakhapatnam" required>
            </div>
            <div class="form-group">
                <label>Degree</label>
                <input type="text" name="Degree" value="B.Tech" required>
            </div>
            <div class="form-group">
                <label>Academic Pressure (1 - 5)</label>
                <input type="number" step="0.5" min="0" max="5" name="Academic Pressure" value="3.0" required>
            </div>
            <div class="form-group">
                <label>CGPA (0 - 10)</label>
                <input type="number" step="0.1" min="0" max="10" name="CGPA" value="7.5" required>
            </div>
            <div class="form-group">
                <label>Study Satisfaction (1 - 5)</label>
                <input type="number" step="0.5" min="0" max="5" name="Study Satisfaction" value="3.0" required>
            </div>
            <div class="form-group">
                <label>Work/Study Hours per Day</label>
                <input type="number" step="0.5" min="0" max="24" name="Work/Study Hours" value="6.0" required>
            </div>
            <div class="form-group">
                <label>Sleep Duration</label>
                <select name="Sleep Duration">
                    <option value="Less than 5 hours">Less than 5 hours</option>
                    <option value="5-6 hours">5-6 hours</option>
                    <option value="7-8 hours" selected>7-8 hours</option>
                    <option value="More than 8 hours">More than 8 hours</option>
                </select>
            </div>
            <div class="form-group">
                <label>Dietary Habits</label>
                <select name="Dietary Habits">
                    <option value="Healthy">Healthy</option>
                    <option value="Moderate" selected>Moderate</option>
                    <option value="Unhealthy">Unhealthy</option>
                </select>
            </div>
            <div class="form-group">
                <label>Financial Stress (1 - 5)</label>
                <input type="number" step="1" min="1" max="5" name="Financial Stress" value="3" required>
            </div>
            <div class="form-group">
                <label>Family History of Mental Illness</label>
                <select name="Family History of Mental Illness">
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>
            <div class="form-group" style="grid-column: span 2;">
                <label>Have you ever had suicidal thoughts?</label>
                <select name="Have you ever had suicidal thoughts ?">
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>
        </div>
        <button type="submit">Predict Depression Risk</button>
    </form>

    {% if result %}
    <div class="result {{ 'high-risk' if prediction == 1 else 'low-risk' }}">
        <strong>Prediction: {{ result }}</strong><br>
        Estimated Depression Probability: {{ probability }}%
    </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        form_data = {
            'Gender': request.form['Gender'],
            'Age': float(request.form['Age']),
            'City': request.form['City'],
            'Degree': request.form['Degree'],
            'Academic Pressure': float(request.form['Academic Pressure']),
            'CGPA': float(request.form['CGPA']),
            'Study Satisfaction': float(request.form['Study Satisfaction']),
            'Work/Study Hours': float(request.form['Work/Study Hours']),
            'Sleep Duration': request.form['Sleep Duration'],
            'Dietary Habits': request.form['Dietary Habits'],
            'Financial Stress': float(request.form['Financial Stress']),
            'Family History of Mental Illness': request.form['Family History of Mental Illness'],
            'Have you ever had suicidal thoughts ?': request.form['Have you ever had suicidal thoughts ?']
        }
        pred, prob = run_prediction(form_data)
        label = "Depressed (High Risk)" if pred == 1 else "Not Depressed (Low Risk)"
        return render_template_string(HTML_TEMPLATE, result=label, prediction=pred, probability=round(prob * 100, 2))

    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        pred, prob = run_prediction(data)
        return jsonify({
            "prediction": pred,
            "result": "Depressed" if pred == 1 else "Not Depressed",
            "depression_probability": round(prob, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)