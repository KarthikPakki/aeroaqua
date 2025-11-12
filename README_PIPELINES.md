Two pipeline variants are provided to compute daily solar energy and predict water yield.

1) PVLIB pipeline (deterministic clearsky-based)

- Script: `pipeline_pvlib.py`
- Usage example (from repo root):

```powershell
python -c "from pipeline_pvlib import run_pipeline_pvlib; print(run_pipeline_pvlib('2025-11-04', rh_percent=50.0))"
```

This pipeline uses pvlib clearsky GHI integrated over the day to compute kWh/m^2 and then calls the linear regression in `baselinesorption` to predict liters/day.

2) RandomForest pipeline (data-driven GHI predictor)

- Training script: `model/train_rf_model.py`
- Evaluation script: `model/evaluate_rf_model.py`
- Pipeline: `pipeline_rf.py`

Training with cross-validation:

```powershell
python -m aeroaqua.model.train_rf_model --csv path\to\usaWithWeather.csv --cv-folds 5 --test-size 0.2
```

This will:
- Split data into train/test sets (80/20 by default)
- Perform 5-fold cross-validation on training set
- Evaluate on held-out test set
- Report RMSE, MAE, R², and feature importances
- Save model to `model/solar_predictor_model.joblib`

Evaluate trained model on new data:

```powershell
python -m aeroaqua.model.evaluate_rf_model --model path\to\model.joblib --csv path\to\test_data.csv --output-dir evaluation_results
```

This will generate evaluation metrics and save diagnostic plots (predicted vs actual, residuals).

Run the RF pipeline after training:

```powershell
python -c "from pipeline_rf import run_pipeline_rf; print(run_pipeline_rf('2025-11-04', cloud_type=0, rh_percent=50.0, temperature_c=20.0))"
```

Or use the CLI wrapper:

```powershell
python -m aeroaqua.scripts.run_rf --date 2025-11-04 --cloud 0.0 --rh 50.0 --temp 20.0 --model path\to\model.joblib
```

## Validation

Validate trained RF model against Open-Meteo ground truth data:

```powershell
python -m aeroaqua.scripts.validate_openmeteo --start 2024-06-01 --end 2024-06-07 --model path\to\model.joblib
```

This will:
- Fetch hourly weather and GHI data from Open-Meteo API
- Use the same feature engineering as the pipeline
- Compare model predictions against actual GHI measurements
- Report R², RMSE, MAE, and MAPE metrics
- Generate scatter plot and residual plot

Notes and assumptions
- `baselinesorption.predict_water_yield(solar_energy_kwh_m2, rh_percent)` is used for both pipelines.
- The RF pipeline expects a trained model stored at `model/solar_predictor_model.joblib` (or pass `model_path`).
- Cross-validation helps assess model generalization before deployment.
- Test set evaluation provides unbiased performance estimates.

Requirements
- Install dependencies from `requirements.txt`.

```powershell
pip install -r requirements.txt
```

If you want, I can run a smoke test here (installing dependencies and running both pipelines).