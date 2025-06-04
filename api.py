from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
import pandas as pd
import numpy as np
import joblib
from auth import create_access_token, get_current_user, oauth2_scheme
from confluent_kafka import Consumer, KafkaError
import json
import threading
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
model = joblib.load('catboost_tuned_model.pkl')

# Kafka consumer configuration for Aiven with client authentication
conf = {
    'bootstrap.servers': f"{os.getenv('HOST')}:{os.getenv('PORT')}",  
    'security.protocol': 'SSL',
    'ssl.ca.location': os.path.join(os.getcwd(), 'ca.pem'),
    'ssl.certificate.location': os.path.join(os.getcwd(), 'service.cert'),
    'ssl.key.location': os.path.join(os.getcwd(), 'service.key'),
    'group.id': 'fraud-detection-group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(conf)
consumer.subscribe(['transactions'])

# Background task to process Kafka messages
def consume_transactions():
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(f'Consumer error: {msg.error()}')
                break
        transaction = json.loads(msg.value().decode('utf-8'))
        input_data = pd.DataFrame([transaction])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        print(f"Processed transaction: Prediction={prediction}, Probability={probability:.2%}")

# Start consumer in background
threading.Thread(target=consume_transactions, daemon=True).start()

# Token endpoint for login
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    from auth import users_db, pwd_context
    user = users_db.get(form_data.username)
    if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Secure prediction endpoint (manual input for testing)
@app.post("/predict")
async def predict(data: dict, current_user: str = Depends(get_current_user)):
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return {"prediction": int(prediction), "fraud_probability": float(probability)}