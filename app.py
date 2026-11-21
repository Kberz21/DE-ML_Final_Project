from flask import Flask, request, jsonify
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

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Student Depression Meta-Classifier API is running on AWS EC2"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Apply exact ordinal/binary encodings
        input_df['Have you ever had suicidal thoughts ?'] = input_df['Have you ever had suicidal thoughts ?'].map(BINARY_MAP)
        input_df['Family History of Mental Illness'] = input_df['Family History of Mental Illness'].map(BINARY_MAP)
        input_df['Sleep Duration'] = input_df['Sleep Duration'].map(SLEEP_MAP)
        input_df['Dietary Habits'] = input_df['Dietary Habits'].map(DIET_MAP)

        # One-hot encode and align columns with X_train
        input_encoded = pd.get_dummies(input_df)
        input_aligned = input_encoded.reindex(columns=training_columns, fill_value=0)

        # Predict class and probability
        prediction = int(model.predict(input_aligned)[0])
        probability = float(model.predict_proba(input_aligned)[0][1])

        return jsonify({
            "prediction": prediction,
            "result": "Depressed" if prediction == 1 else "Not Depressed",
            "depression_probability": round(probability, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # 0.0.0.0 allows external connections via the EC2 Public IP
    app.run(host='0.0.0.0', port=5000)