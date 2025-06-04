import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and test data
model = joblib.load('catboost_tuned_model.pkl')
X_test_scaled = pd.read_csv('X_test_scaled.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

st.title("Fraud Detection Dashboard")

# Option to select input method
input_method = st.radio("Choose Input Method", ["Select Sample Transaction", "Manual Input"])

if input_method == "Select Sample Transaction":
    # Dropdown for sample transactions (first 5 from test set)
    sample_indices = list(range(5))  # Use first 5 test samples
    sample_choice = st.selectbox("Select a Transaction", sample_indices, format_func=lambda x: f"Transaction {x+1}")
    
    # Get selected transaction
    input_data = X_test_scaled.iloc[sample_choice].to_dict()
    st.write("Selected Transaction Data (scaled):")
    st.json(input_data)

else:
    # Manual input form
    st.header("Enter Transaction Data")
    input_data = {}
    for col in X_test_scaled.columns:
        input_data[col] = st.number_input(col, value=0.0, step=0.01)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.write(f"Prediction: **{'Fraud' if prediction == 1 else 'Non-Fraud'}**")
    st.write(f"Fraud Probability: **{probability:.2%}**")

    # SHAP explanation for the prediction
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    st.subheader("Feature Importance for This Prediction")
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], input_df, matplotlib=True)
    plt.savefig('shap_force.png')
    st.image('shap_force.png')

# Display global SHAP summary plot
st.header("Global Feature Importance")
st.image('shap_summary.png')