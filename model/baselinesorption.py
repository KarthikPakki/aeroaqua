import pandas as pd
from sklearn.linear_model import LinearRegression
from io import StringIO


# The small experimental dataset from the paper is embedded below.
# We fit a LinearRegression at import time and expose a helper
# function `predict_water_yield` that returns liters/day given
# solar energy (kWh/m^2) and relative humidity (%).

_csv_data = """RH_Percent,Solar_Energy_kWhr_m2,Liters_Per_Day
20,5,2.5
30,5,3.0
40,5,3.5
50,5,4.0
60,5,4.0
70,5,4.5
20,5.41,3.0
30,5.41,3.5
40,5.41,4.0
50,5.41,4.0
60,5.41,4.5
70,5.41,5.0
20,5.83,3.0
30,5.83,3.5
40,5.83,4.0
50,5.83,4.5
60,5.83,5.0
70,5.83,5.5
20,6.25,3.0
30,6.25,4.0
40,6.25,4.5
50,6.25,5.0
60,6.25,5.5
70,6.25,6.0
20,6.66,3.5
30,6.66,4.0
40,6.66,5.0
50,6.66,5.5
60,6.66,6.0
70,6.66,6.0
"""

_df = pd.read_csv(StringIO(_csv_data))

_features = ['RH_Percent', 'Solar_Energy_KWhr_m2']
# Note: dataset header from original used 'Solar_Energy_kWhr_m2' lowercase 'kWh', keep compatibility
# We'll select by a case-insensitive match when building features for prediction.
_target = 'Liters_Per_Day'

# Normalize column names to a known form
_df.columns = [c.replace('Solar_Energy_kWhr_m2', 'Solar_Energy_kWhr_m2') for c in _df.columns]

_X = _df[['RH_Percent', 'Solar_Energy_kWhr_m2']]
_y = _df[_target]

_model = LinearRegression()
_model.fit(_X, _y)


def predict_water_yield(solar_energy_kwh_m2: float, rh_percent: float) -> float:
    """Predict liters per day given solar energy (kWh/m^2) and RH (%).

    Args:
        solar_energy_kwh_m2: daily solar energy in kWh per m^2
        rh_percent: relative humidity in percent (0-100)

    Returns:
        Predicted liters per day (float)
    """
    X_pred = pd.DataFrame([[rh_percent, solar_energy_kwh_m2]], columns=['RH_Percent', 'Solar_Energy_kWhr_m2'])
    pred = _model.predict(X_pred)[0]
    return float(pred)


if __name__ == '__main__':
    # simple verification printout
    print("Model Verification:")
    print(f"Intercept (B0): {_model.intercept_:.4f}")
    print(f"RH_Percent Coefficient (B1): {_model.coef_[0]:.4f}")
    print(f"Solar_Energy Coefficient (B2): {_model.coef_[1]:.4f}")
