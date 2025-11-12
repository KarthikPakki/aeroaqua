"""
Validate Random Forest GHI predictions against Open-Meteo ground truth.
"""
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from aeroaqua.pipelines.pipeline_rf import _find_model
from aeroaqua.solar import get_solar_positions_for_date, DEFAULT_LATITUDE, DEFAULT_LONGITUDE, DEFAULT_TZ
import joblib


def fetch_openmeteo_data(start_date: str, end_date: str):
    """Fetch hourly weather and solar data from Open-Meteo for Toronto."""
    params = {
        "latitude": DEFAULT_LATITUDE,
        "longitude": DEFAULT_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["shortwave_radiation", "temperature_2m", "relative_humidity_2m", "cloud_cover"],
        "timezone": DEFAULT_TZ
    }

    url = "https://archive-api.open-meteo.com/v1/archive"
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "shortwave_radiation": "GHI_true_Wm2",
        "temperature_2m": "Temperature",
        "relative_humidity_2m": "Relative Humidity",
        "cloud_cover": "Cloud Type"
    })
    return df


def validate_rf_openmeteo(start_date="2024-01-01", end_date="2024-12-31", model_path=None, sample_dates=None):
    """Run RF model vs. Open-Meteo GHI and compare.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        model_path: Optional path to trained model, otherwise uses _find_model()
        sample_dates: List of specific dates to sample (YYYY-MM-DD). If provided, fetches only these dates.
    
    Returns:
        dict with validation metrics
    """
    print("=" * 80)
    print("OPEN-METEO VALIDATION")
    print("=" * 80)
    
    # If sample_dates provided, fetch only those specific dates
    if sample_dates:
        print(f"Fetching Open-Meteo data for {len(sample_dates)} sampled dates...\n")
        all_dfs = []
        for date in sample_dates:
            try:
                df_single = fetch_openmeteo_data(date, date)
                all_dfs.append(df_single)
            except Exception as e:
                print(f"Warning: Failed to fetch {date}: {e}")
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        print(f"Fetching Open-Meteo data for {start_date} → {end_date}...\n")
        df = fetch_openmeteo_data(start_date, end_date)
    
    # Add solar geometry using existing pipeline function
    print("Computing solar positions...")
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H', tz=DEFAULT_TZ)
    
    all_solpos = []
    for ts in date_range:
        date_str = ts.strftime('%Y-%m-%d')
        hour_str = ts.strftime('%H')
        solpos = get_solar_positions_for_date(date_str=date_str, freq='1H')
        # Find the matching hour
        hour_match = solpos[solpos.index.hour == int(hour_str)]
        if len(hour_match) > 0:
            all_solpos.append({
                'time': ts,
                'Solar Zenith Angle': hour_match.iloc[0].get('apparent_zenith', hour_match.iloc[0].get('zenith'))
            })
    
    solpos_df = pd.DataFrame(all_solpos)
    solpos_df['time'] = pd.to_datetime(solpos_df['time'])
    
    # Ensure both dataframes have timezone-aware timestamps
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize(DEFAULT_TZ)
    
    # Merge with Open-Meteo data
    df = df.merge(solpos_df, on='time', how='inner')
    
    # Add temporal features
    df['Month'] = df['time'].dt.month
    df['Day'] = df['time'].dt.day
    df['Hour'] = df['time'].dt.hour
    
    # Normalize cloud cover to 0-1 range (Open-Meteo gives 0-100)
    # Training data has Cloud Type in 0-1 range
    df['Cloud Type'] = df['Cloud Type'] / 100.0
    
    # Load model
    print("Loading trained RF model...")
    found_model = _find_model(model_path)
    if not found_model:
        raise FileNotFoundError('RandomForest model not found. Train model first with train_rf_model.py')
    
    model = joblib.load(found_model)
    print(f"Model loaded from: {found_model}\n")

    # Predict GHI
    feature_cols = ["Cloud Type", "Solar Zenith Angle", "Relative Humidity", "Temperature", "Month", "Day", "Hour"]
    X = df[feature_cols]
    df["GHI_pred_Wm2"] = model.predict(X)

    # Compare
    valid = df.dropna(subset=["GHI_true_Wm2", "GHI_pred_Wm2"])
    
    if len(valid) == 0:
        print("ERROR: No valid data points for comparison")
        return None
    
    r2 = r2_score(valid["GHI_true_Wm2"], valid["GHI_pred_Wm2"])
    mae = mean_absolute_error(valid["GHI_true_Wm2"], valid["GHI_pred_Wm2"])
    rmse = np.sqrt(mean_squared_error(valid["GHI_true_Wm2"], valid["GHI_pred_Wm2"]))

    print("VALIDATION METRICS")
    print("=" * 80)
    print(f"Samples: {len(valid)}")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.2f} W/m²")
    print(f"MAE: {mae:.2f} W/m²")
    
    # Check if reasonable
    if r2 > 0.7:
        print("\n[PASS] Model shows good correlation with Open-Meteo data")
    elif r2 > 0.5:
        print("\n[WARN] Model shows moderate correlation with Open-Meteo data")
    else:
        print("\n[FAIL] Model shows poor correlation with Open-Meteo data")

    # Plot
    print("\nGenerating validation plot...")
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.scatter(valid["GHI_true_Wm2"], valid["GHI_pred_Wm2"], alpha=0.5, s=10)
    max_val = max(valid["GHI_true_Wm2"].max(), valid["GHI_pred_Wm2"].max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect prediction')
    plt.xlabel("Open-Meteo GHI (W/m²)")
    plt.ylabel("Predicted GHI (W/m²)")
    plt.title(f"RF Model vs. Open-Meteo ({start_date} to {end_date})\nR²={r2:.4f}, RMSE={rmse:.2f} W/m²")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    residuals = valid["GHI_pred_Wm2"] - valid["GHI_true_Wm2"]
    plt.scatter(valid["GHI_true_Wm2"], residuals, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Open-Meteo GHI (W/m²)")
    plt.ylabel("Residual (W/m²)")
    plt.title("Residual Plot")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('openmeteo_validation.png', dpi=150)
    print("Saved plot to: openmeteo_validation.png")
    plt.show()
    
    return {
        'n_samples': len(valid),
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate RF model against Open-Meteo data')
    parser.add_argument('--start', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--model', default=None, help='Path to trained model')
    parser.add_argument('--sample', action='store_true', help='Use diverse sample dates instead of continuous range')
    args = parser.parse_args()
    
    # Define diverse sample dates covering different seasons and conditions
    # Similar to TEST_SCENARIOS but spread across full year
    SAMPLE_DATES = [
        '2024-01-15',  # Winter - deep winter
        '2024-02-10',  # Winter - late winter
        '2024-03-20',  # Spring - spring equinox
        '2024-04-15',  # Spring - mid spring
        '2024-05-20',  # Spring - late spring
        '2024-06-21',  # Summer - summer solstice
        '2024-06-01',  # Summer - early summer
        '2024-07-15',  # Summer - mid summer
        '2024-08-10',  # Summer - late summer
        '2024-09-22',  # Fall - fall equinox
        '2024-10-15',  # Fall - mid fall
        '2024-11-04',  # Fall - late fall
        '2024-12-21',  # Winter - winter solstice
        # Add some random dates for variety
        '2024-03-05',  # Early spring
        '2024-04-25',  # Spring
        '2024-07-04',  # Mid summer
        '2024-08-25',  # Late summer
        '2024-09-10',  # Early fall
        '2024-10-31',  # Late fall
        '2024-12-10',  # Early winter
    ]
    
    if args.sample:
        validate_rf_openmeteo(sample_dates=SAMPLE_DATES, model_path=args.model)
    else:
        validate_rf_openmeteo(args.start, args.end, args.model)
