import joblib
import yaml
import pandas as pd
import numpy as np

def load_config():
    try:
        with open('../configs/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Default config for demo
        return {'model': {'model_path': '../data/processed/model.joblib'}}

def predict_churn(data):
    config = load_config()
    model_path = config['model']['model_path']
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        # Demo model - simple logistic for demo data
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        X_demo = np.array([[1,30,30], [60,100,6000]])
        y_demo = np.array([1,0])
        model.fit(X_demo, y_demo)

    # Assume data is df or list[dict], preprocess same way as training
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X = df[num_cols].fillna(df[num_cols].mean())
    
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return preds.tolist(), probs.tolist()

if __name__ == '__main__':
    # Test
    sample = [{'tenure': 12, 'MonthlyCharges': 29.85, 'TotalCharges': 29.85}]
    preds, probs = predict_churn(sample)
    print(f"Predictions: {preds}, Probs: {probs}")

