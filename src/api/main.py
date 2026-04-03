from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import pandas as pd
import yaml
import uvicorn
from src.models.predict_model import predict_churn

app = FastAPI(title="Churn Prediction API")

class CustomerData(BaseModel):
    data: Dict[str, Any]

@app.post("/predict")
async def predict(customer: CustomerData):
    try:
        prob = predict_churn(customer.data)
        return {"churn_probability": float(prob)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
