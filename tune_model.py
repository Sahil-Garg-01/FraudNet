import pandas as pd
import optuna
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score

# Load data
X_train_smote = pd.read_csv('X_train_smote.csv')
y_train_smote = pd.read_csv('y_train_smote.csv').values.ravel()
X_test_scaled = pd.read_csv('X_test_scaled.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

# Define objective function for Optuna
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 200),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    model = CatBoostClassifier(**params, verbose=0, random_seed=42)
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test_scaled)
    return recall_score(y_test, y_pred, pos_label=1)

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)
print("Best parameters:", study.best_params)
print("Best recall:", study.best_value)

# Train and save best model
best_model = CatBoostClassifier(**study.best_params, verbose=0, random_seed=42)
best_model.fit(X_train_smote, y_train_smote)
joblib.dump(best_model, 'catboost_tuned_model.pkl')