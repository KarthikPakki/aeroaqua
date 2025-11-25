# AeroAqua - Atmospheric Water Generation with Explainable AI

Predictive modeling for atmospheric water generation (AWG) using solar energy predictions with **SHAP** and **LIME** explainability.

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Generate XAI Report
```bash
# Detailed analysis (15-20 min, recommended)
python scripts/generate_xai_report.py --model model/solar_predictor_model.joblib --csv usaWithWeather.csv --detailed

# Standard analysis (10-15 min)
python scripts/generate_xai_report.py --model model/solar_predictor_model.joblib --csv usaWithWeather.csv
```

**Output**: Timestamped folder with 20+ visualizations, SHAP analysis, LIME explanations, and comparison guide.

## Explainable AI

### SHAP (SHapley Additive exPlanations)
- Game theory-based, exact feature attribution
- Global importance + local explanations
- Validates model learned correct physics

### LIME (Local Interpretable Model-agnostic Explanations)  
- Simple linear approximations
- Intuitive stakeholder explanations
- Fast individual predictions

## Key Results

**Most Important Features (SHAP):**
1. Solar Zenith Angle (sun position)
2. Relative Humidity (moisture availability)
3. Temperature
4. Cloud Type

This validates the model learned correct physical relationships for solar energy prediction.

## Structure

```
aeroaqua/
├── model/                      # RF training & evaluation
├── explainability/             # SHAP, LIME, comparison
├── pipelines/                  # RF & PVLib pipelines
├── scripts/generate_xai_report.py  # Main XAI generator
└── usaWithWeather.csv          # Training data
```