import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ---------------- CONFIG LOADER ---------------- #
def load_config():
    try:
        with open('../configs/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        print("Config not found, using default paths")
        return {
            "data": {
                "raw_path": "data/churn.csv",
                "test_size": 0.2,
                "random_state": 42
            }
        }


# ---------------- BASIC PREPROCESS (OPTIONAL) ---------------- #
def preprocess_data(config):
    df = pd.read_csv(config['data']['raw_path'])

    # Fix TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Convert target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Features & target
    X = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
    y = df['Churn']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    print("Basic preprocessing done")
    return X_train, X_test, y_train, y_test


# ---------------- MAIN PIPELINE FUNCTION ---------------- #
def preprocess_churn_df(df):
    """
    Full preprocessing pipeline:
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features
    - Split into train/test
    """

    # Fix TotalCharges column
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop ID column
    df = df.drop(['customerID'], axis=1, errors='ignore')

    # Convert target
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    # Features
    X = df.drop(['Churn'], axis=1)

    # Auto-detect columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    # Pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    # Transform data
    X_processed = preprocessor.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ---------------- MAIN EXECUTION ---------------- #
if __name__ == '__main__':
    # Load dataset directly
    df = pd.read_csv("data/churn.csv.csv")

    # Run full preprocessing
    X_train, X_test, y_train, y_test = preprocess_churn_df(df)