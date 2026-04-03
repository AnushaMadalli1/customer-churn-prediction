from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from src.models.predict import predict_churn

app = FastAPI(title="Churn Prediction API", docs_url="/docs")

class PredictionRequest(BaseModel):
    data: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    preds, probs = predict_churn(request.data)
    return {"predictions": preds, "probabilities": probs}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    preds, probs = predict_churn(df)
    return {"predictions": preds, "probabilities": probs}

if __name__ == "__main__":
    import uvicorn
    config = {'host': '127.0.0.1', 'port': 5000}
    uvicorn.run(app, **config)

