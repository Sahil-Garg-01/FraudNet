import json
from confluent_kafka import Producer
import pandas as pd
import time
import os

# Kafka configuration 
conf = {
    'bootstrap.servers': f"{os.getenv('HOST')}:{os.getenv('PORT')}",
    'security.protocol': 'SSL',
    'ssl.ca.location': os.path.join(os.getcwd(), 'ca.pem'),
    'ssl.certificate.location': os.path.join(os.getcwd(), 'service.cert'),
    'ssl.key.location': os.path.join(os.getcwd(), 'service.key')
}

# Create Producer instance
producer = Producer(conf)

# Load sample test data
X_test_scaled = pd.read_csv('X_test_scaled.csv')
sample_transactions = X_test_scaled.iloc[:5].to_dict('records')  # Use 5 test samples

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()}')

# Produce transactions to Kafka topic
topic = 'transactions'
for transaction in sample_transactions:
    producer.produce(topic, json.dumps(transaction).encode('utf-8'), callback=delivery_report)
    producer.flush()
    time.sleep(1)  # Simulate real-time streaming