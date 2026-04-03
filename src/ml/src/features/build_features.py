import pandas as pd
import numpy as np

def build_features(df):
    """Feature engineering for churn prediction."""
    # Example features
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, np.inf], labels=['0-1y', '1-2y', '2-4y', '4y+'])
    df['charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
    return df

# Usage in other scripts

