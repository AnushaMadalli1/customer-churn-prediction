import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV

from src.data.preprocess import preprocess_churn_df

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_sample_df():
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'customerID': [f'cust_{i}' for i in range(n_samples)],
        'tenure': np.random.randint(0, 73, n_samples),
        'MonthlyCharges': np.random.uniform(18, 120, n_samples),
        'TotalCharges': np.random.uniform(0, 9000, n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73]),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples)
    })
    df['TotalCharges'] = np.where(df['tenure'] == 0, 0, df['MonthlyCharges'] * df['tenure'])
    return df

def train_model():
    config = load_config()
    try:
        df = pd.read_csv(config['data']['raw_path'])
        X_train, X_test, y_train, y_test = preprocess_churn_df(df)
    except FileNotFoundError:
        print("No data file found, using sample data")
        df = generate_sample_df()
        X_train, X_test, y_train, y_test = preprocess_churn_df(df)
    except:
        print("Error in preprocessing, check data")
        return
    
    # Handle sparse
    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray()
        X_test = X_test.toarray()
    
    # RF Hyperparam tuning
    rf_tuned_params = {
        'n_estimators': [100],
        'max_depth': [10]
    }
    rf_base = RandomForestClassifier(random_state=config['random_state'])
    f1_scorer = make_scorer(f1_score, average='macro')
    rf_grid = GridSearchCV(rf_base, rf_tuned_params, cv=config.get('cv_folds', 3), scoring=f1_scorer, n_jobs=-1)
    print("Tuning RF hyperparameters...")
    rf_grid.fit(X_train, y_train)
    tuned_rf = rf_grid.best_estimator_
    print(f"RF Best Params: {rf_grid.best_params_}")
    print(f"RF Best CV F1: {rf_grid.best_score_:.3f}")
    
    # Test tuned RF
    y_pred_rf = tuned_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf, average='macro')
    rec_rf = recall_score(y_test, y_pred_rf, average='macro')
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    print("\nTuned RF Test Results:")
    print(classification_report(y_test, y_pred_rf))
    
    # Base models
    rf_base_mdl = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=config['random_state'])
    lr_mdl = LogisticRegression(random_state=config['random_state'])
    
    rf_base_mdl.fit(X_train, y_train)
    lr_mdl.fit(X_train, y_train)
    
    y_pred_base_rf = rf_base_mdl.predict(X_test)
    y_pred_lr = lr_mdl.predict(X_test)
    
    # Metrics
    metrics = [
        {'model': 'Base RF', 'accuracy': accuracy_score(y_test, y_pred_base_rf), 'precision': precision_score(y_test, y_pred_base_rf, average='macro'), 'recall': recall_score(y_test, y_pred_base_rf, average='macro'), 'f1': f1_score(y_test, y_pred_base_rf, average='macro')},
        {'model': 'Logistic Regression', 'accuracy': accuracy_score(y_test, y_pred_lr), 'precision': precision_score(y_test, y_pred_lr, average='macro'), 'recall': recall_score(y_test, y_pred_lr, average='macro'), 'f1': f1_score(y_test, y_pred_lr, average='macro')},
        {'model': 'Tuned RF', 'accuracy': acc_rf, 'precision': prec_rf, 'recall': rec_rf, 'f1': f1_rf}
    ]
    
    metrics_df = pd.DataFrame(metrics)
    print("\nModel Comparison (Test Set):")
    print(metrics_df.round(3))
    
    # Save
    model_path = config['model']['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(tuned_rf, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == '__main__':
    train_model()

