import sys
import joblib
import yaml
import pandas as pd
import numpy as np
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from src.data.preprocess import preprocess_churn_df

def load_config():
    with open('../configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_user_input():
    print("Enter customer features for churn prediction:")
    tenure = int(input("Tenure (months): "))
    monthly_charges = float(input("Monthly Charges ($): "))
    total_charges = float(input("Total Charges ($): "))
    gender = input("Gender (Male/Female): ")
    partner = input("Partner (Yes/No): ")
    dependents = input("Dependents (Yes/No): ")
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check'
    }
    return data

if __name__ == '__main__':
    config = load_config()
    model_path = config['models']['model_path']
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run 'python src/models/train_model.py' first.")
        exit(1)
    
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    
    data = get_user_input()
    df = pd.DataFrame([data])
    
    # Preprocess (demo - full would save fitted preprocessor)
    # For simplicity, extract numerical as model expects scaled features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X = df[num_cols]
    
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0, 1]
    
    churn_status = 'Yes' if pred == 1 else 'No'
    print(f"\nPrediction: Customer will {'churn' if pred == 1 else 'NOT churn'} (probability: {prob:.2%})")
