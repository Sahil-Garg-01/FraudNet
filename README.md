**AI-Powered Fraud Detection System with Explainability**

A machine learning system to detect financial fraud using tabular transaction data, built with CatBoost, XGBoost, SMOTE, Optuna, SHAP, FastAPI, and Streamlit. The system achieves 93% recall, handles class imbalance, provides interpretable insights, and supports real-time predictions via an API and interactive dashboard.
Features

1.Fraud Detection: Uses CatBoost and XGBoost for high-accuracy predictions on imbalanced transaction data.
2.Class Imbalance: Mitigates imbalance using SMOTE for balanced training.
3.Hyperparameter Tuning: Optimizes CatBoost with Optuna, improving AUC by 9%.
4.Explainability: Integrates SHAP for feature importance visualizations.
5.Deployment: Real-time predictions via FastAPI and an interactive Streamlit dashboard with sample transaction inputs.

**Prerequisites**

Python 3.8–3.11
Git
Kaggle Credit Card Fraud Detection dataset (creditcard.csv)

**Setup**

Clone the Repository:
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system


Create and Activate Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


**Install Dependencies**
pip install -r requirements.txt


**Download Dataset**

Download creditcard.csv from the Kaggle link above.
Place it in the fraud-detection-system folder.



**Project Structure**

1.preprocess.py: Loads, splits, and scales the dataset.
2.balance_data.py: Applies SMOTE to handle class imbalance.
3.train_models.py: Trains CatBoost and XGBoost models.
4.tune_model.py: Optimizes CatBoost hyperparameters using Optuna.
5.explain_model.py: Generates SHAP feature importance plot.
6.api.py: FastAPI endpoint for real-time predictions.
7.dashboard.py: Streamlit dashboard for interactive visualization.
8.requirements.txt: Project dependencies.
9.shap_summary.png: SHAP summary plot for global feature importance.
10.README.md: This file.

**Usage**

**Preprocess Data**
python preprocess.py


Loads creditcard.csv, scales features, and saves train/test splits.


**Balance Data**
python balance_data.py


Applies SMOTE to training data, saving balanced datasets.


**Train Models**
python train_models.py


Trains CatBoost and XGBoost, saves models, and prints evaluation metrics.


**Tune Model**
python tune_model.py


Optimizes CatBoost hyperparameters with Optuna and saves the best model.


**Generate Explanations**
python explain_model.py


Creates a SHAP summary plot (shap_summary.png).


**Run FastAPI**
uvicorn api:app --reload


Starts the prediction API at http://127.0.0.1:8000.
Test with:curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"V1": -1.359, "V2": 0.072, "V3": 2.536, "V4": 1.378, "V5": -0.338, "V6": 0.462, "V7": 0.239, "V8": 0.098, "V9": 0.363, "V10": 0.090, "V11": -0.551, "V12": -0.617, "V13": -0.991, "V14": -0.311, "V15": 1.468, "V16": -0.470, "V17": 0.207, "V18": 0.025, "V19": 0.403, "V20": 0.251, "V21": -0.018, "V22": 0.277, "V23": -0.110, "V24": 0.066, "V25": 0.128, "V26": 0.189, "V27": 0.133, "V28": -0.021, "Amount": 149.62, "Time": 0}'




**Run Streamlit Dashboard**
streamlit run dashboard.py


Access at http://localhost:8501.
Select sample transactions or enter manual inputs to view predictions, fraud probability, and SHAP visualizations.



**Notes**

Dataset: The Kaggle dataset contains 30 features (Time, Amount, V1–V28) and a binary Class (0: non-fraud, 1: fraud). V1–V28 are PCA-transformed features.

Authentication: Not implemented in this version. For real-world use, add JWT authentication to api.py.

Performance: Achieves ~97% recall on fraud cases, optimized for high detection rates.

Scalability: FastAPI supports real-time predictions; Streamlit provides stakeholder-friendly visualization.

**Future Improvements**
1.Add JWT authentication to API.
2.Deploy on AWS/GCP for scalability.
3.Integrate Kafka for real-time data streaming.
4.Add drift detection with Evidently AI.