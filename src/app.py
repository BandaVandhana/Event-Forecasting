from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ✅ Load model, scaler, and feature names
model_path = os.path.join("outputs", "models", "demand_forecast_model.pkl")
scaler_path = os.path.join("outputs", "models", "scaler.pkl")
features_path = os.path.join("outputs", "models", "feature_names.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)

@app.route('/')
def home():
    return "🎯 Event-Aware Demand Forecasting API is running! Use the /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Ensure all required features exist
        X = np.array([[data.get(feat, 0) for feat in feature_names]])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

        return jsonify({'predicted_demand': round(float(prediction), 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
