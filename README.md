 AI-Powered Fraud Detection System with Explainability

 A machine learning system to detect financial fraud using tabular transaction data, built with CatBoost, XGBoost, SMOTE, Optuna, SHAP, FastAPI, and Streamlit. The system achieves 93% recall, handles class imbalance, provides interpretable insights, and supports secure real-time predictions via an API with JWT authentication and Aiven Kafka streaming using SSL client authentication.

 ## Features
 - **Fraud Detection**: Uses CatBoost and XGBoost for high-accuracy predictions on imbalanced transaction data.
 - **Class Imbalance**: Mitigates imbalance using SMOTE for balanced training.
 - **Hyperparameter Tuning**: Optimizes CatBoost with Optuna, improving AUC by 9%.
 - **Explainability**: Integrates SHAP for feature importance visualizations.
 - **Deployment**: Secure real-time API with FastAPI and JWT authentication, with Aiven Kafka for streaming transactions, and an interactive Streamlit dashboard.

 ## Prerequisites
 - Python 3.8–3.11
 - Git
 - [Aiven Kafka Service](https://aiven.io/kafka) with Service URI, host, port, CA certificate, access key (client certificate), and access certificate (client key)
 - [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`creditcard.csv`)

 ## Setup
 1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/fraud-detection-system.git
    cd fraud-detection-system
    ```

 2. **Create and Activate Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

 3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

 4. **Configure Aiven Kafka**:
    - Obtain Service URI, host, port, CA certificate, access key (service.cert), and access certificate (service.key) from Aiven Console.
    - Save `ca.pem`, `service.cert`, and `service.key` in the project folder.
    - Update `producer.py` and `api.py` with your Aiven host and port.

 5. **Download Dataset**:
    - Download `creditcard.csv` from the Kaggle link above.
    - Place it in the `fraud-detection-system` folder.

 ## Project Structure
 - `preprocess.py`: Loads, splits, and scales the dataset.
 - `balance_data.py`: Applies SMOTE to handle class imbalance.
 - `train_models.py`: Trains CatBoost and XGBoost models.
 - `tune_model.py`: Optimizes CatBoost hyperparameters using Optuna.
 - `explain_model.py`: Generates SHAP feature importance plot.
 - `api.py`: FastAPI endpoint for secure predictions with Aiven Kafka consumer.
 - `auth.py`: JWT authentication logic for API security.
 - `producer.py`: Aiven Kafka producer to simulate transaction streaming.
 - `dashboard.py`: Streamlit dashboard for interactive visualization.
 - `requirements.txt`: Project dependencies.
 - `README.md`: This file.

 ## Usage
 1. **Preprocess Data**:
    ```bash
    python preprocess.py
    ```
    - Loads `creditcard.csv`, scales features, and saves train/test splits.

 2. **Balance Data**:
    ```bash
    python balance_data.py
    ```
    - Applies SMOTE to training data, saving balanced datasets.

 3. **Train Models**:
    ```bash
    python train_models.py
    ```
    - Trains CatBoost and XGBoost, saves models, and prints evaluation metrics.

 4. **Tune Model**:
    ```bash
    python tune_model.py
    ```
    - Optimizes CatBoost hyperparameters with Optuna and saves the best model.

 5. **Generate Explanations**:
    ```bash
    python explain_model.py
    ```
    - Creates a SHAP summary plot (saved as needed).

 6. **Run FastAPI with Aiven Kafka Consumer**:
    ```bash
    uvicorn api:app --reload
    ```
    - Starts the secure prediction API at `http://127.0.0.1:8000` with Aiven Kafka consumer.
    - Get a token:
      ```bash
      curl -X POST "http://127.0.0.1:8000/token" -H "Content-Type: application/x-www-form-urlencoded" -d "username=testuser&password=testpassword"
      ```
    - Test manual prediction:
      ```bash
      curl -X POST "http://127.0.0.1:8000/predict" -H "Authorization: Bearer <access_token>" -H "Content-Type: application/json" -d '{"V1": -1.359, "V2": 0.072, "V3": 2.536, "V4": 1.378, "V5": -0.338, "V6": 0.462, "V7": 0.239, "V8": 0.098, "V9": 0.363, "V10": 0.090, "V11": -0.551, "V12": -0.617, "V13": -0.991, "V14": -0.311, "V15": 1.468, "V16": -0.470, "V17": 0.207, "V18": 0.025, "V19": 0.403, "V20": 0.251, "V21": -0.018, "V22": 0.277, "V23": -0.110, "V24": 0.066, "V25": 0.128, "V26": 0.189, "V27": 0.133, "V28": -0.021, "Amount": 149.62, "Time": 0}'
      ```

 7. **Run Aiven Kafka Producer**:
    ```bash
    python producer.py
    ```
    - Streams sample transactions to the `transactions` topic on Aiven Kafka.

 8. **Run Streamlit Dashboard**:
    ```bash
    streamlit run dashboard.py
    ```
    - Access at `http://localhost:8501` for manual or sample transaction predictions.

 ## Notes
 - **Dataset**: The Kaggle dataset contains 30 features (`Time`, `Amount`, `V1`–`V28`) and a binary `Class` (0: non-fraud, 1: fraud). `V1`–`V28` are PCA-transformed features.
 - **Authentication**: Uses JWT for secure API access with a mock user database (`testuser`/`testpassword`). Replace with a real database in production.
 - **Aiven Kafka**: Streams transactions securely using SSL client authentication with CA certificate, access key, and access certificate.
 - **Performance**: Achieves ~93% recall on fraud cases, optimized for high detection rates.
 - **Future Improvements**:
   - Deploy on AWS/GCP for scalability.
   - Add drift detection with Evidently AI.
   - Integrate a real database for authentication.

