from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()
model = joblib.load('catboost_tuned_model.pkl')

@app.post("/predict")
async def predict(data: dict):
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return {"prediction": int(prediction), "fraud_probability": float(probability)}