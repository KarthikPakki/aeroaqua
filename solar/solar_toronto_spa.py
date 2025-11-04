import pandas as pd
import pvlib
from pvlib.location import Location


# Default location values (Toronto Harbourfront)
DEFAULT_LATITUDE = 43.64
DEFAULT_LONGITUDE = -79.39
DEFAULT_ALTITUDE = 76  # meters
DEFAULT_TZ = 'America/Toronto'


def get_solar_positions_for_date(
    date_str: str = '2025-11-04',
    freq: str = '10min',  # Changed from '10T' to '10min'
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    altitude: float = DEFAULT_ALTITUDE,
    timezone: str = DEFAULT_TZ,
):
    """Return a DataFrame of solar position values for the given date and location.

    Args:
        date_str: date string in 'YYYY-MM-DD' format.
        freq: pandas frequency string, e.g. '10min'.
        latitude, longitude, altitude, timezone: location parameters.

    Returns:
        pandas.DataFrame: solar position table (pvlib's get_solarposition output).
    """
    start = f"{date_str} 00:00:00"
    end = f"{date_str} 23:59:00"
    times = pd.date_range(start=start, end=end, freq=freq, tz=timezone)

    location = Location(latitude=latitude, longitude=longitude, tz=timezone, altitude=altitude)
    solpos = location.get_solarposition(times)
    return solpos


if __name__ == '__main__':
    # keep a simple CLI-like behavior for convenience
    solpos = get_solar_positions_for_date('2025-11-04')
    print("Solar Position for Toronto Harbourfront (first 10 results):")
    print(solpos[['apparent_zenith', 'zenith', 'apparent_elevation', 'azimuth']].head(10))
    # example value at noon (may raise if timezone mismatch)
    try:
        print("\n--- Example Data Point ---")
        print(f"Solar Zenith Angle at Noon: {solpos.at['2025-11-04 12:00:00-05:00', 'apparent_zenith']:.2f} degrees")
    except Exception:
        # best-effort: print a noon row if an exact tz-labeled index key fails
        noon = solpos.between_time('12:00', '12:10')
        if not noon.empty:
            print(f"Solar Zenith Angle at Noon: {noon.iloc[0]['apparent_zenith']:.2f} degrees")
        else:
            print("No exact noon row found.")
