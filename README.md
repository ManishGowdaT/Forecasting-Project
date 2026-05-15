# Sales Forecasting System

End-to-end machine learning based sales forecasting system built using Python, Time-Series Forecasting models, and FastAPI.

## Project Overview

This project predicts weekly sales forecasts for multiple US states using different forecasting and machine learning models. The system compares model performance automatically and selects the best-performing model for each state.

The final predictions are exposed through a FastAPI REST API.

---

## Features

- Data cleaning and preprocessing
- Weekly time-series resampling
- Feature engineering
- Time-series forecasting
- Automatic model comparison
- REST API integration
- Forecast visualization

---

## Models Used

- SARIMA / ARIMA
- Prophet
- XGBoost
- LSTM

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Prophet
- XGBoost
- TensorFlow / Keras
- FastAPI
- Matplotlib
- Joblib

---

## Project Structure

```bash
Forecasting-Project/
│
├── data/
├── artifacts/
├── data_overview.py
├── feature_engineering.py
├── sarima_forecast.py
├── prophet_forecast.py
├── xgboost_forecast.py
├── lstm_forecast.py
├── model_selection.py
├── forecast_api.py
├── requirements.txt
└── dashboard.html
```

---

## Workflow

1. Data preprocessing and cleaning
2. Weekly resampling
3. Feature engineering
4. Model training
5. Forecast generation
6. Model evaluation using MAPE and RMSE
7. Best model selection
8. API deployment using FastAPI

---

## API

Run the API server:

```bash
python -m uvicorn forecast_api:app --reload
```

Open Swagger API docs:

```bash
http://127.0.0.1:8000/docs
```

---

## Evaluation Metrics

The training scripts use a shared metric implementation: MAE and RMSE include
all validation observations; MAPE excludes weeks with zero actual sales and
fails explicitly if all actual values are zero. Metric calculations reject
non-finite values and mismatched lengths. Run the metric regression tests with:

```bash
python -m unittest discover -s tests -v
```

Existing serialized models and scores are not recalculated by source edits.
Regenerate and compare artifacts after training before reporting new accuracy
figures. The stored predictions describe a historical validation window.

- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

---

## Results

- Prophet achieved the best performance across most states.
- Automatic model selection was implemented based on forecasting accuracy.

---

## Future Improvements

- Real-time forecasting
- Cloud deployment
- Interactive dashboard
- Hyperparameter tuning
- Automated retraining pipeline

---

## Author

Manish Gowda T
