import pandas as pd
import shap
import joblib

# Load data and model
X_test_scaled = pd.read_csv('X_test_scaled.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()
model = joblib.load('catboost_tuned_model.pkl')

# Compute SHAP values (use a subset for speed)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled[:1000])

# Save SHAP summary plot
shap.summary_plot(shap_values, X_test_scaled[:1000], feature_names=X_test_scaled.columns, show=False)
import matplotlib.pyplot as plt
plt.savefig('shap_summary.png')
plt.close()