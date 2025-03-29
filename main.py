import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import os

app = FastAPI()

dataset_path = "manufacturing_data.csv"
model_path = "model.pkl"

# Generate synthetic dataset
def generate_synthetic_data():
    np.random.seed(42)
    data = {
        "Machine_ID": np.arange(1, 101),
        "Temperature": np.random.randint(50, 100, 100),
        "Run_Time": np.random.randint(100, 500, 100),
        "Downtime_Flag": np.random.choice([0, 1], 100, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)

generate_synthetic_data()

# Upload CSV endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df.to_csv(dataset_path, index=False)
        return {"message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Train model endpoint
@app.post("/train")
def train_model():
    df = pd.read_csv(dataset_path)
    X = df[["Temperature", "Run_Time"]]
    y = df["Downtime_Flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    return {"accuracy": accuracy, "f1_score": f1}

# Prediction endpoint
class InputData(BaseModel):
    Temperature: float
    Run_Time: float

@app.post("/predict")
def predict_downtime(data: InputData):
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Model not trained. Please train first.")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    input_data = np.array([[data.Temperature, data.Run_Time]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].max()
    
    return {"Downtime": "Yes" if prediction == 1 else "No", "Confidence": round(probability, 2)}

# Run FastAPI with: uvicorn filename:app --reload
