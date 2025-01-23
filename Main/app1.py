from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)

# Global variables
model_path = "logi.pkl"
features_path = "feature_names.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None
feature_names = joblib.load(features_path) if os.path.exists(features_path) else None
data = None

@app.route('/upload', methods=['POST'])
def upload_data():
    global data
    file = request.files.get('file')
    if file:
        data = pd.read_csv(file)
        return jsonify({"message": "Data uploaded successfully", "columns": data.columns.tolist()})
    return jsonify({"error": "No file uploaded"}), 400

@app.route('/train', methods=['POST'])
def train_model():
    global model, feature_names, data
    if data is None:
        return jsonify({"error": "No data uploaded. Please upload data first."}), 400
    
    if 'Downtime_Flag' not in data.columns:
        return jsonify({"error": "Target column 'Downtime_Flag' is missing in the dataset."}), 400
    
    X = data.drop(columns=['Downtime_Flag'])
    y = data['Downtime_Flag']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and feature names
    joblib.dump(model, model_path)
    joblib.dump(feature_names, features_path)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return jsonify({"message": "Model trained successfully", "accuracy": accuracy, "f1_score": f1})

@app.route('/predict', methods=['POST'])
def predict():
    global model, feature_names
    if model is None:
        return jsonify({"error": "No trained model found. Train the model first."}), 400
    
    input_data = request.get_json()
    if not input_data:
        return jsonify({"error": "Invalid input data"}), 400
    
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure all features are present
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0  # Add missing features with default value
        
        # Ensure column order matches
        df = df[feature_names]
        
        # Predict
        prediction = model.predict(df)
        confidence = model.predict_proba(df).max()
        return jsonify({"Downtime": "Yes" if prediction[0] else "No", "Confidence": round(confidence, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
