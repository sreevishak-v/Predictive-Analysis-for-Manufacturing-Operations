
# Predictive Analysis for Manufacturing Operations

This project involves creating a RESTful API to predict machine downtime or production defects based on a manufacturing dataset. The API provides endpoints for uploading data, training a machine learning model, and making predictions in postman.

## Features
- Upload a dataset to the API.
- Train a Logistic Regression model using the uploaded dataset.
- Predict machine downtime using input parameters such as temperature and runtime.

## Requirements


### Install Dependencies
Run the following command to install the required Python packages:
```bash
  pip install flask scikit-learn joblib pandas
```

## Usage
### Running the API
#### 1 . Place the following files in the same directory:
- app.py (the Flask application file).

- logi.pkl (pre-trained Logistic Regression model).

- feature_names.pkl (feature names used during training).

#### 2. Start the Flask app:
 ```bash
  python app.py
```

#### 2. The app will run locally at http://127.0.0.1:5000.

## API Endpoints

### 1. Upload Dataset
 - Endpoint: /upload
 - Method: POST
 - Description: Upload a dataset for training or retraining.
 - Request: Form-data: Key = file, Value = [CSV file]
 - Response:
 ```bash
 {
  "message": "Data uploaded successfully",
  "columns": ["Machine_ID", "Temperature", "Run_Time", "Downtime_Flag"]
}
```
### 2. Train Model
- Endpoint: /train
- Method: POST
- Description: Train the Logistic Regression model using the uploaded dataset.
- Request: None

- Response:
 ```bash
{
  "message": "Model trained successfully",
  "accuracy": 0.85,
  "f1_score": 0.82
}
```
### 3. Train Model
- Endpoint: /predict

- Method: POST

- Description: Predict machine downtime based on input parameters.

- Request:
- json body 
 ```bash
{
  "message": "Model trained successfully",
  "accuracy": 0.85,
  "f1_score": 0.82
}
```
- response 
 ```bash
{
  "Downtime": "No",
  "Confidence": 0.81
}
```

## Files in the Repository
1. app.py: Flask application code.

2.  logi.pkl: Pre-trained Logistic Regression model.

3.  feature_names.pkl: List of feature names used during training.

4. test_manufacturing_data.csv: Example dataset for testing.
