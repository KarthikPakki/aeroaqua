import os
import joblib
import pandas as pd
from aeroaqua.solar import get_solar_positions_for_date, DEFAULT_LATITUDE, DEFAULT_LONGITUDE, DEFAULT_ALTITUDE, DEFAULT_TZ
from aeroaqua.model import predict_water_yield


MODEL_FALLBACK_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', 'model', 'solar_predictor_model.joblib'),
    os.path.join(os.getcwd(), 'solarenergy', 'solar_predictor_model.joblib'),
    os.path.join(os.getcwd(), 'model', 'solar_predictor_model.joblib')
]


def _find_model(path_hint: str = None):
    if path_hint and os.path.exists(path_hint):
        return path_hint
    for p in MODEL_FALLBACK_PATHS:
        candidate = os.path.normpath(p)
        if os.path.exists(candidate):
            return candidate
    return None


def run_pipeline_rf(
    date_str: str = '2025-11-04',
    cloud_type: float = 0.0,
    rh_percent: float = 50.0,
    temperature_c: float = 20.0,
    model_path: str = None,
    freq: str = '10min',  # Changed from '10T' to '10min'
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    altitude: float = DEFAULT_ALTITUDE,
    timezone: str = DEFAULT_TZ,
):
    """Run the RF-based pipeline.

    Steps:
    1. Compute solar positions for the date to get Solar Zenith Angle and timestamps.
    2. Assemble feature dataframe expected by the RF model using provided scalars (cloud_type, RH, temperature)
       which are broadcast to every time sample.
    3. Load the trained RandomForest model and predict GHI (W/m^2) per sample.
    4. Integrate predicted GHI over the day to get daily solar energy (kWh/m^2).
    5. Feed daily solar energy and RH into baselinesorption.predict_water_yield to get liters/day.

    Returns a dict with keys: date, solar_energy_kwh_m2, rh_percent, predicted_lpd
    """
    solpos = get_solar_positions_for_date(date_str=date_str, freq=freq, latitude=latitude, longitude=longitude, altitude=altitude, timezone=timezone)
    times = solpos.index

    df_feat = pd.DataFrame(index=times)
    if 'apparent_zenith' in solpos.columns:
        df_feat['Solar Zenith Angle'] = solpos['apparent_zenith']
    elif 'zenith' in solpos.columns:
        df_feat['Solar Zenith Angle'] = solpos['zenith']
    else:
        raise RuntimeError('Solar position table does not contain zenith columns')

    df_feat['Cloud Type'] = float(cloud_type)
    df_feat['Relative Humidity'] = float(rh_percent)
    df_feat['Temperature'] = float(temperature_c)
    df_feat['Month'] = df_feat.index.month
    df_feat['Day'] = df_feat.index.day
    df_feat['Hour'] = df_feat.index.hour

    found = _find_model(model_path)
    if not found:
        raise FileNotFoundError('RandomForest model not found. Please run model/train_rf_model.py to create solar_predictor_model.joblib and pass its path via model_path.')

    model = joblib.load(found)

    input_features = ['Cloud Type', 'Solar Zenith Angle', 'Relative Humidity', 'Temperature', 'Month', 'Day', 'Hour']
    X = df_feat[input_features]

    ghi_pred = model.predict(X)
    ghi_series = pd.Series(ghi_pred, index=times)

    dt = times.to_series().diff().dt.total_seconds().div(3600).fillna(pd.Timedelta(freq).total_seconds() / 3600)
    wh_per_sample = ghi_series * dt.values
    total_kwh = wh_per_sample.sum() / 1000.0

    predicted = predict_water_yield(total_kwh, rh_percent)

    return {
        'date': pd.to_datetime(date_str).date(),
        'solar_energy_kwh_m2': float(total_kwh),
        'rh_percent': float(rh_percent),
        'predicted_liters_per_day': float(predicted),
    }


if __name__ == '__main__':
    try:
        out = run_pipeline_rf(date_str='2025-11-04', cloud_type=0.0, rh_percent=50.0, temperature_c=20.0)
        print('RF Pipeline result:')
        print(out)
    except Exception as e:
        print('Error running RF pipeline:')
        print(e)
        print('\nTo train a model:')
        print('python -m aeroaqua.model.train_rf_model --csv path/to/usaWithWeather.csv')
