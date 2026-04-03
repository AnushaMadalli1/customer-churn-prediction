import joblib
import pandas as pd
import yaml
import numpy as np

with open('../../configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def predict_churn(customer_data: dict) -> float:
    """Predict churn probability for a single customer."""
    model = joblib.load('../../models/churn_model.pkl')
    scaler = joblib.load('../../models/scaler.pkl')
    encoders = joblib.load('../../models/encoders.pkl')
    
    df = pd.DataFrame([customer_data])
    
    # Apply same preprocessing as training
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = scaler.transform(df[num_cols])
    
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = encoders[col].transform(df[col].astype(str))
    
    prob = model.predict_proba(df)[0][1]
    return prob

if __name__ == '__main__':
    # Example usage
    sample = {'tenure': 12, 'MonthlyCharges': 65.0, 'Contract': 'Month-to-month'}  # Adjust keys to dataset
    print(predict_churn(sample))

