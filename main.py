# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load('fraud_model.joblib')

# Define the structure of the request body using Pydantic
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    """
    Accepts transaction data and returns a fraud prediction.
    """
    # Convert the input data into a pandas DataFrame
    input_data = pd.DataFrame([transaction.dict()])
    
    # Make a prediction
    prediction = model.predict(input_data)[0]
    
    # Get the prediction probability
    probability = model.predict_proba(input_data)[0].max()
    
    return {
        "prediction": int(prediction),
        "probability_score": float(probability)
    }

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running"}
