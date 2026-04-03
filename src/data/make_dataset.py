import os
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

# Load config
with open('../../configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_dataset():
    """Load raw data, preprocess, split train/test."""
    raw_path = Path(config['data']['raw_path'])
    if not raw_path.exists():
        logger.error(f"Raw data not found at {raw_path}")
        return
    
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Basic preprocessing (customize for churn data)
    df = df.dropna()
    # Encode categoricals, etc.
    
    target = config['model']['target']
    X = df.drop(target, axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['model']['test_size'], 
        random_state=config['model']['random_state']
    )
    
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    logger.info("Dataset split and saved.")

if __name__ == '__main__':
    make_dataset()

