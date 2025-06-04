import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load balanced training and test data
X_train_smote = pd.read_csv('X_train_smote.csv')
y_train_smote = pd.read_csv('y_train_smote.csv').values.ravel()
X_test_scaled = pd.read_csv('X_test_scaled.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

# Train CatBoost
catboost_model = CatBoostClassifier(iterations=50, depth=6, learning_rate=0.1, verbose=0, random_seed=42)
catboost_model.fit(X_train_smote, y_train_smote)
y_pred_catboost = catboost_model.predict(X_test_scaled)
print("CatBoost Classification Report:")
print(classification_report(y_test, y_pred_catboost))
print("CatBoost ROC-AUC:", roc_auc_score(y_test, catboost_model.predict_proba(X_test_scaled)[:, 1]))

# Train XGBoost
xgboost_model = XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42)
xgboost_model.fit(X_train_smote, y_train_smote)
y_pred_xgboost = xgboost_model.predict(X_test_scaled)
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgboost))
print("XGBoost ROC-AUC:", roc_auc_score(y_test, xgboost_model.predict_proba(X_test_scaled)[:, 1]))

# Save models
joblib.dump(catboost_model, 'catboost_model.pkl')
joblib.dump(xgboost_model, 'xgboost_model.pkl')