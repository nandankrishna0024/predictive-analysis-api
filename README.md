# Predictive Analysis API for Manufacturing Operations

## Overview
This API predicts machine downtime using a simple Logistic Regression model trained on manufacturing data. The API includes endpoints to upload data, train a model, and make predictions.

## Features
- Upload manufacturing data via CSV
- Train a logistic regression model
- Predict machine downtime based on input features

## Requirements
- Python 3.8+
- FastAPI
- Uvicorn
- Pandas
- NumPy
- Scikit-learn
- Pydantic

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/predictive-analysis-api.git
   cd predictive-analysis-api
   ```
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn pandas numpy scikit-learn pydantic
   ```

## Running the API
Start the FastAPI server:
```bash
uvicorn filename:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

## API Endpoints

### 1. Upload Dataset
- **Endpoint:** `POST /upload`
- **Description:** Upload a CSV file containing manufacturing data.
- **Request:** Multipart Form Data (`file`)
- **Response:**
  ```json
  { "message": "File uploaded successfully" }
  ```

### 2. Train Model
- **Endpoint:** `POST /train`
- **Description:** Train the model using the uploaded dataset.
- **Response:**
  ```json
  { "accuracy": 0.85, "f1_score": 0.75 }
  ```

### 3. Predict Downtime
- **Endpoint:** `POST /predict`
- **Description:** Predict machine downtime based on temperature and run time.
- **Request:**
  ```json
  { "Temperature": 80, "Run_Time": 120 }
  ```
- **Response:**
  ```json
  { "Downtime": "Yes", "Confidence": 0.85 }
  ```

## Testing the API
Use **Postman** or **cURL** to test endpoints.
- Example cURL request:
  ```bash
  curl -X POST "http://127.0.0.1:8000/predict" \
       -H "Content-Type: application/json" \
       -d '{"Temperature": 80, "Run_Time": 120}'
  ```

## License
This project is licensed under the MIT License.

