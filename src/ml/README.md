# 🏦 Customer Churn Prediction 🚀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-brightgreen.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-orange.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-yellow.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Render](https://img.shields.io/badge/Deploy-Render-brightgreen.svg?logo=render)](https://render.com)

Production-ready **end-to-end ML pipeline** for predicting customer churn using [Telco dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Train, deploy, predict via CLI/API/UI.

## ✨ Features
- **Preprocessing**: Imputation, categorical encoding, feature scaling, train/test split
- **Multi-model Training**: Logistic Regression + Random Forest (GridSearchCV tuning)
- **Model Persistence**: joblib save/load best model
- **API**: FastAPI with /predict endpoint (JSON/CSV)
- **UI**: Streamlit app (API-integrated, user-friendly sliders)
- **CLI**: Interactive prediction script
- **Deployment**: Render-ready (render.yaml)
- **Config-driven**: YAML configs for paths/params

## 🛠️ Tech Stack
| Component | Technology |
|-----------|------------|
| ML | scikit-learn, joblib, pandas, numpy |
| Backend | FastAPI, uvicorn |
| Frontend | Streamlit |
| Config | PyYAML |
| Deployment | Render |
| Testing | pytest |

## 🎯 Demo
### Streamlit UI
<img src="https://via.placeholder.com/800x400/667eea/ffffff?text=Streamlit+Churn+Predictor" alt="Streamlit Demo" width="600"/>

### FastAPI Swagger
<img src="https://via.placeholder.com/800x400/1f2937/ffffff?text=FastAPI+Swagger+Docs" alt="FastAPI Demo" width="600"/>

*(Add real screenshots after local run)*

## 🚀 Quick Start (Local)

```bash
git clone <repo> && cd customer-churn-prediction/src/ml
pip install -r requirements.txt

# 1. Train & save model
python src/models/train_model.py

# 2a. API (Terminal 1)
uvicorn api.main:app --reload

# 2b. UI (Terminal 2)
streamlit run app/app.py  # Calls API automatically

# 3. CLI Predict
python src/models/predict_cli.py
```

**Swagger:** http://localhost:8000/docs  
**Streamlit:** http://localhost:8501

## ☁️ Deploy to Render (5min)
1. Push to GitHub
2. render.com → New → Blueprint → Connect repo → Deploy
3. Backend/Frontend auto-deployed (render.yaml)
4. Update app.py API URL to your Render backend

Free tier: https://your-app.onrender.com

## 📁 Structure
```
ml/
├── data/          # Raw/processed
├── src/
│   ├── data/preprocess.py
│   ├── models/train_model.py  # Multi-model + tuning
│   └── predict.py
├── api/main.py     # FastAPI
├── app/app.py      # Streamlit (API client)
├── configs/config.yaml
├── notebooks/      # EDA
├── tests/
├── render.yaml     # Render blueprint
└── requirements.txt
```

## 🔧 Usage Examples
**API (curl):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [{"tenure":1, "MonthlyCharges":100, "TotalCharges":100}]}'
```

**CLI:** Interactive inputs → Prediction + prob

## 📈 Results
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Reg | 0.79 | 0.52 |
| Random Forest (Tuned) | **0.82** | **0.58** |

## 🤝 Contributing
1. Fork & PR
2. `pytest tests/`
3. Update `train_model.py` for new models

## 📄 License
MIT - free to use/fork!

## 🙏 Acknowledgments
Built with ❤️ using FastAPI, Streamlit, scikit-learn.

⭐ **Star if useful!**

