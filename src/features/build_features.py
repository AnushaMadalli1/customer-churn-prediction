import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
import joblib

with open('../../configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def build_features(X_train, X_test):
    """Feature engineering for churn prediction.
    Handles scaling and encoding for typical churn features like tenure, charges, contract.
    """
    # Numerical scaling
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[num_cols] = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]), 
        columns=num_cols, 
        index=X_train.index
    )
    X_test[num_cols] = pd.DataFrame(
        scaler.transform(X_test[num_cols]), 
        columns=num_cols, 
        index=X_test.index
    )
    
    # Categorical label encoding
    cat_cols = X_train.select_dtypes(include='object').columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
    
    joblib.dump(scaler, '../../models/scaler.pkl')
    joblib.dump(encoders, '../../models/encoders.pkl')
    
    return X_train, X_test

if __name__ == '__main__':
    X_train = pd.read_csv('../../data/processed/X_train.csv')
    X_test = pd.read_csv('../../data/processed/X_test.csv')
    X_train_proc, X_test_proc = build_features(X_train, X_test)
    X_train_proc.to_csv('../../data/processed/X_train_features.csv', index=False)
    X_test_proc.to_csv('../../data/processed/X_test_features.csv', index=False)

