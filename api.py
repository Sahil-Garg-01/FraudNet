from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
import pandas as pd
import numpy as np
import joblib
from auth import create_access_token, get_current_user, oauth2_scheme

app = FastAPI()
model = joblib.load('catboost_tuned_model.pkl')

# Token endpoint for login
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    from auth import users_db, pwd_context
    user = users_db.get(form_data.username)
    if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Secure prediction endpoint
@app.post("/predict")
async def predict(data: dict, current_user: str = Depends(get_current_user)):
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return {"prediction": int(prediction), "fraud_probability": float(probability)}