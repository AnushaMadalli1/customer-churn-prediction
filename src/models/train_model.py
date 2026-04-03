import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('../../configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def train_model():
    \"\"\"Train XGBoost model on churn data.\"\"\"
    X_train = pd.read_csv('../../data/processed/X_train_features.csv')
    y_train = pd.read_csv('../../data/processed/y_train.csv').squeeze()
    
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=config['model']['random_state'],
        eval_metric='logloss'
    )
    
    # MLflow tracking
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    with mlflow.start_run():
        model.fit(X_train, y_train)
        joblib.dump(model, '../../models/churn_model.pkl')
        
        # Log params/metrics
        mlflow.log_params(model.get_params())
        mlflow.xgboost.log_model(model, "model")
        logger.info("Model trained and saved.")
    
    logger.info("Training complete.")

if __name__ == '__main__':
    train_model()

