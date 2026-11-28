# Weekly Sales Forecasting

An end-to-end time-series forecasting case study for weekly beverage sales across 43 U.S. states. The repository contains data preparation, four model implementations, a FastAPI service, and a dashboard for exploring **stored historical validation results**.

> This is a reproducible portfolio project, not a live forecasting service. The API serves forecasts and evaluation data generated from a historical holdout. Model rankings are exploratory because the current selection step chooses the lowest error on that same holdout.

## What it does

- Resamples irregular source observations to weekly state-level series.
- Builds lag and calendar features for machine-learning models.
- Compares SARIMA, Prophet, XGBoost, and LSTM approaches over an eight-week holdout.
- Exposes stored predictions and metrics through FastAPI.
- Provides a browser dashboard for viewing state and model comparisons.

## Repository layout

```text
.
├── data/
│   └── sales_data.xlsx             # Source dataset
├── artifacts/                      # Saved evaluation results and model files
├── tests/
│   └── test_forecast_metrics.py
├── prepare_data.py                 # Source data → weekly series and features
├── sarima_forecast.py
├── prophet_forecast.py
├── xgboost_forecast.py
├── lstm_forecast.py
├── model_selection.py              # Selects per-state model from stored results
├── forecast_api.py
├── forecast_metrics.py
├── dashboard.html
└── requirements.txt
```

## Setup

Use Python 3.10 or newer. TensorFlow and Prophet may require platform-specific installation steps.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the existing dashboard and API

The API loads the committed serialized files in `artifacts/` at startup. Start it from the repository root:

```bash
python -m uvicorn forecast_api:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) for the dashboard and [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation.

The dashboard is served by FastAPI so its requests use the same origin. Do not open `dashboard.html` directly as a local file.

## Rebuild the saved results

Run commands from the repository root. Training scripts overwrite files under `artifacts/`.

```bash
python prepare_data.py
python sarima_forecast.py
python prophet_forecast.py
python xgboost_forecast.py
python lstm_forecast.py
python model_selection.py
```

Model training can take several minutes and requires the full dependency set. The committed results let reviewers inspect the API and dashboard without retraining.

## Evaluation

The current scripts use the final eight weekly observations as a holdout and report MAE, RMSE, and MAPE. MAPE omits zero-valued actuals; calculations reject non-finite values and mismatched series lengths.

The per-state “best model” is chosen using error on that holdout. Therefore those selected-model scores are not an unbiased estimate of future performance. A stronger comparison should use rolling-origin backtesting, keep model selection separate from final evaluation, and compare against a seasonal-naive baseline before making performance claims.

Run the metric regression tests with:

```bash
python -m unittest discover -s tests -v
```

## Data and limitations

The source workbook is included in `data/sales_data.xlsx`. Review its source and license before reuse or redistribution. The current API returns stored historical validation forecasts; it does not retrain models or generate forecasts beyond the stored window at request time.

The scripts currently use simplified preprocessing and calendar features. Results should be treated as a case study, not a production demand-planning recommendation. See the code and evaluation caveats above before interpreting model rankings.

## License

No license file is currently provided. Add a license before inviting reuse or redistribution.
