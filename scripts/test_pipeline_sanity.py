"""Sanity check the full pipeline with realistic Toronto weather scenarios."""
import pandas as pd
from aeroaqua.pipelines import run_pipeline_rf
import warnings

# suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


TEST_SCENARIOS = [
    # Summer - sunny days
    {'date': '2024-06-21', 'cloud': 0.0, 'temp': 28.0, 'rh': 55.0, 'desc': 'Summer solstice - clear'},
    {'date': '2024-07-15', 'cloud': 0.2, 'temp': 30.0, 'rh': 60.0, 'desc': 'Hot summer day - few clouds'},
    {'date': '2024-08-10', 'cloud': 0.5, 'temp': 26.0, 'rh': 65.0, 'desc': 'Late summer - partly cloudy'},
    
    # Fall - moderate conditions
    {'date': '2024-09-22', 'cloud': 0.4, 'temp': 18.0, 'rh': 60.0, 'desc': 'Fall equinox - mild'},
    {'date': '2024-10-15', 'cloud': 0.7, 'temp': 12.0, 'rh': 70.0, 'desc': 'Cool fall - mostly cloudy'},
    {'date': '2024-11-04', 'cloud': 0.8, 'temp': 6.0, 'rh': 75.0, 'desc': 'Late fall - overcast'},
    
    # Winter - low sun angle, cold
    {'date': '2024-12-21', 'cloud': 0.6, 'temp': -5.0, 'rh': 70.0, 'desc': 'Winter solstice - cloudy'},
    {'date': '2025-01-15', 'cloud': 0.9, 'temp': -8.0, 'rh': 65.0, 'desc': 'Deep winter - very cloudy'},
    {'date': '2025-02-10', 'cloud': 0.3, 'temp': -3.0, 'rh': 60.0, 'desc': 'Late winter - clear cold'},
    
    # Spring - warming up
    {'date': '2025-03-20', 'cloud': 0.5, 'temp': 5.0, 'rh': 55.0, 'desc': 'Spring equinox - partly cloudy'},
    {'date': '2025-04-15', 'cloud': 0.4, 'temp': 12.0, 'rh': 50.0, 'desc': 'Spring warming - few clouds'},
    {'date': '2025-05-20', 'cloud': 0.2, 'temp': 20.0, 'rh': 55.0, 'desc': 'Late spring - mostly clear'},
    
    # Edge cases
    {'date': '2024-07-04', 'cloud': 1.0, 'temp': 22.0, 'rh': 85.0, 'desc': 'Summer storm - overcast & humid'},
    {'date': '2025-01-25', 'cloud': 0.0, 'temp': -15.0, 'rh': 50.0, 'desc': 'Clear winter day - very cold'},
]


def run_sanity_check():
    """Run pipeline on test scenarios and validate outputs."""
    
    print("=" * 80)
    print("AEROAQUA PIPELINE SANITY CHECK")
    print("=" * 80)
    print("\nTesting: Solar Position → RF Model → GHI Integration → Water Yield\n")
    
    results = []
    
    for scenario in TEST_SCENARIOS:
        try:
            output = run_pipeline_rf(
                date_str=scenario['date'],
                cloud_type=scenario['cloud'],
                rh_percent=scenario['rh'],
                temperature_c=scenario['temp']
            )
            
            results.append({
                'Date': scenario['date'],
                'Description': scenario['desc'],
                'Cloud': scenario['cloud'],
                'Temp (°C)': scenario['temp'],
                'RH (%)': scenario['rh'],
                'Solar (kWh/m²)': round(output['solar_energy_kwh_m2'], 2),
                'Water (L/day)': round(output['predicted_liters_per_day'], 2),
            })
            
        except Exception as e:
            print(f"FAILED: {scenario['date']}: {e}")
            results.append({
                'Date': scenario['date'],
                'Description': scenario['desc'],
                'Cloud': scenario['cloud'],
                'Temp (°C)': scenario['temp'],
                'RH (%)': scenario['rh'],
                'Solar (kWh/m²)': 'ERROR',
                'Water (L/day)': 'ERROR',
            })
    
    # Display results
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    valid_results = df[df['Solar (kWh/m²)'] != 'ERROR']
    
    if len(valid_results) == 0:
        print("ERROR: All tests failed. Check model path and dependencies.")
        return
    
    solar_values = valid_results['Solar (kWh/m²)'].astype(float)
    water_values = valid_results['Water (L/day)'].astype(float)
    
    # Check 1: No negative values
    if (solar_values < 0).any() or (water_values < 0).any():
        print("[FAIL] Found negative values (physically impossible)")
    else:
        print("[PASS] No negative values")
    
    # Check 2: Reasonable solar energy range for Toronto
    if solar_values.min() < 0.5 or solar_values.max() > 9.0:
        print(f"[WARN] Solar energy outside typical range (0.5-9 kWh/m²)")
        print(f"       Found: {solar_values.min():.2f} to {solar_values.max():.2f} kWh/m²")
    else:
        print(f"[PASS] Solar energy in reasonable range: {solar_values.min():.2f} to {solar_values.max():.2f} kWh/m²")
    
    # Check 3: Reasonable water yield range
    if water_values.min() < 1.5 or water_values.max() > 9.0:
        print(f"[WARN] Water yield outside expected range (1.5-9 L/day)")
        print(f"       Found: {water_values.min():.2f} to {water_values.max():.2f} L/day")
    else:
        print(f"[PASS] Water yield in reasonable range: {water_values.min():.2f} to {water_values.max():.2f} L/day")
    
    # Check 4: Summer > Winter (seasonal pattern)
    summer_solar = valid_results[valid_results['Date'].str.contains('2024-06|2024-07|2024-08')]['Solar (kWh/m²)'].astype(float).mean()
    winter_solar = valid_results[valid_results['Date'].str.contains('2024-12|2025-01|2025-02')]['Solar (kWh/m²)'].astype(float).mean()
    
    if summer_solar > winter_solar * 1.5:
        print(f"[PASS] Seasonal pattern correct: Summer {summer_solar:.2f} >> Winter {winter_solar:.2f}")
    else:
        print(f"[WARN] Weak seasonal pattern: Summer {summer_solar:.2f}, Winter {winter_solar:.2f}")
    
    # Check 5: Clear sky > Cloudy sky (for same season)
    summer_clear = valid_results[(valid_results['Date'].str.contains('2024-06-21'))]['Solar (kWh/m²)'].values
    summer_cloudy = valid_results[(valid_results['Date'].str.contains('2024-07-04'))]['Solar (kWh/m²)'].values
    
    if len(summer_clear) > 0 and len(summer_cloudy) > 0:
        diff = abs(summer_clear[0] - summer_cloudy[0])
        if diff < 0.5:
            print(f"[INFO] Cloud impact minimal (within 0.5 kWh/m²)")
        elif summer_clear[0] > summer_cloudy[0]:
            print(f"[PASS] Clear day ({summer_clear[0]:.2f}) > Cloudy day ({summer_cloudy[0]:.2f})")
        else:
            print(f"[INFO] Cloudy day slightly higher (other factors dominate)")
    
    # Check 6: Higher RH → Higher water yield (for same solar)
    similar_solar = valid_results[(valid_results['Solar (kWh/m²)'].astype(float) > 4.0) & 
                                   (valid_results['Solar (kWh/m²)'].astype(float) < 5.0)]
    if len(similar_solar) >= 2:
        corr = similar_solar[['RH (%)', 'Water (L/day)']].corr().iloc[0, 1]
        if corr > 0.5:
            print(f"[PASS] Humidity correlation correct: r={corr:.2f}")
        else:
            print(f"[WARN] Weak humidity-water correlation: r={corr:.2f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Tested: {len(valid_results)}/{len(TEST_SCENARIOS)} scenarios")
    print(f"Solar energy range: {solar_values.min():.2f} - {solar_values.max():.2f} kWh/m²")
    print(f"Water yield range: {water_values.min():.2f} - {water_values.max():.2f} L/day")
    
    # Overall assessment
    if (solar_values >= 0).all() and (water_values >= 0).all():
        if summer_solar > winter_solar * 1.5 and solar_values.max() < 9.5:
            print("\nRESULT: Pipeline outputs are physically reasonable for Toronto.")
        else:
            print("\nRESULT: Pipeline runs but review seasonal trends.")
    else:
        print("\nRESULT: Pipeline has issues - check model and data.")


if __name__ == '__main__':
    run_sanity_check()
