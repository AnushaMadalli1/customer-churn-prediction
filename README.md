# Customer Churn Prediction ML Project

End-to-end ML project for predicting customer churn using Python.

## Folder Structure
```
customer-churn-prediction/
├── src/
│   ├── data/       # Data loading & preprocessing
│   ├── features/   # Feature engineering
│   ├── models/     # Training & prediction
│   └── api/        # FastAPI inference server
├── app/            # Streamlit dashboard
├── notebooks/      # EDA & experiments
├── configs/        # YAML configs
├── tests/          # Unit tests
├── requirements.txt
├── README.md
└── .gitignore
```

## Quick Start
1. `cd customer-churn-prediction`
2. `python -m venv .venv`
3. `.venv/Scripts/activate` (Windows)
4. `pip install -r requirements.txt`
5. Download dataset (e.g., telco-customer-churn.csv from Kaggle) to `data/raw/`
6. Train: `python src/models/train_model.py`
7. API: `uvicorn src.api.main:app --reload --port 8000`
8. App: `streamlit run app/app.py`

## Usage
- EDA: `jupyter notebook notebooks/eda.ipynb`
- Predict via API: POST to `/predict` with customer JSON.
- Tests: `pytest tests/`

