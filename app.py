from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load pre-trained model and other necessary components
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
selected_features = joblib.load("selected_features.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])

    # Align the columns to match the training data
    input_df = input_df[selected_features].copy()

    # Impute missing values
    input_filled = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

    # Scale the data
    input_scaled = scaler.transform(input_filled)

    # Make predictions
    prediction = model.predict(input_scaled)

    return jsonify({'prediction': prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
