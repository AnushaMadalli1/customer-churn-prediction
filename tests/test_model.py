import pytest
import joblib
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predict_model import predict_churn

def test_predict_churn():
    """Test model prediction."""
    model_path = '../models/churn_model.pkl'
    if os.path.exists(model_path):
        # Load model and test
        sample_data = {'tenure': 12, 'MonthlyCharges': 65.0, 'Contract': 'Month-to-month'}
        prob = predict_churn(sample_data)
        assert 0 <= prob <= 1
    else:
        pytest.skip("Model not trained yet.")

if __name__ == '__main__':
    pytest.main([__file__])

