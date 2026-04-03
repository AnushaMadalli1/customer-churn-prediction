import pytest
import pandas as pd
from src.data.preprocess import preprocess_data
import yaml

def test_preprocess():
    # Mock config
    config = {
        'data': {
            'raw_path': 'tests/sample.csv',  # Create sample
            'processed_path': 'tests/processed.csv',
            'test_size': 0.2,
            'random_state': 42
        }
    }
    
    # Create sample data
    sample_df = pd.DataFrame({
        'customerID': ['A', 'B'],
        'tenure': [1, 2],
        'MonthlyCharges': [29.85, 70.0],
        'TotalCharges': [29.85, 140.0],
        'Churn': ['No', 'Yes']
    })
    sample_df.to_csv('tests/sample.csv', index=False)
    
    X_train, X_test, y_train, y_test = preprocess_data(config)
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    os.remove('tests/sample.csv')
    # Cleanup processed

if __name__ == '__main__':
    pytest.main([__file__])

