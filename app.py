from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
imputer = joblib.load("models/imputer.pkl")
selected_features = joblib.load("models/selected_features.pkl")


@app.route('/')
def home():
    return render_template('index.html', selected_features=selected_features)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = [data.get(feature, np.nan) for feature in selected_features]

    input_df = pd.DataFrame([input_data], columns=selected_features)
    input_filled = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    input_scaled = scaler.transform(input_filled)

    prediction = model.predict(input_scaled)

    return render_template('index.html', prediction=prediction[0], selected_features=selected_features)


if __name__ == "__main__":
    app.run(debug=True)
