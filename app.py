from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import os

app = FastAPI(title="California Housing API")
model = None

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "./model")
    model = mlflow.xgboost.load_model(model_path)
    print("Model loaded successfully!")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(input_data: HousingInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        input_df = pd.DataFrame([input_data.dict()])
        dmatrix_input = xgb.DMatrix(input_df)
        raw_prediction = float(model.predict(dmatrix_input)[0])
        actual_price = raw_prediction * 100000.0
        return {"prediction": f"${actual_price:,.2f}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))