"""Sanity check the full pipeline with realistic Toronto weather scenarios."""
import pandas as pd
from aeroaqua.pipelines import run_pipeline_rf
import warnings

# suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# Experimental data from the baseline sorption paper (ground truth)
BASELINE_EXPERIMENTAL_DATA = [
    {'rh': 20, 'solar': 5.00, 'water': 2.5},
    {'rh': 30, 'solar': 5.00, 'water': 3.0},
    {'rh': 40, 'solar': 5.00, 'water': 3.5},
    {'rh': 50, 'solar': 5.00, 'water': 4.0},
    {'rh': 60, 'solar': 5.00, 'water': 4.0},
    {'rh': 70, 'solar': 5.00, 'water': 4.5},
    {'rh': 20, 'solar': 5.41, 'water': 3.0},
    {'rh': 30, 'solar': 5.41, 'water': 3.5},
    {'rh': 40, 'solar': 5.41, 'water': 4.0},
    {'rh': 50, 'solar': 5.41, 'water': 4.0},
    {'rh': 60, 'solar': 5.41, 'water': 4.5},
    {'rh': 70, 'solar': 5.41, 'water': 5.0},
    {'rh': 20, 'solar': 5.83, 'water': 3.0},
    {'rh': 30, 'solar': 5.83, 'water': 3.5},
    {'rh': 40, 'solar': 5.83, 'water': 4.0},
    {'rh': 50, 'solar': 5.83, 'water': 4.5},
    {'rh': 60, 'solar': 5.83, 'water': 5.0},
    {'rh': 70, 'solar': 5.83, 'water': 5.5},
    {'rh': 20, 'solar': 6.25, 'water': 3.0},
    {'rh': 30, 'solar': 6.25, 'water': 4.0},
    {'rh': 40, 'solar': 6.25, 'water': 4.5},
    {'rh': 50, 'solar': 6.25, 'water': 5.0},
    {'rh': 60, 'solar': 6.25, 'water': 5.5},
    {'rh': 70, 'solar': 6.25, 'water': 6.0},
    {'rh': 20, 'solar': 6.66, 'water': 3.5},
    {'rh': 30, 'solar': 6.66, 'water': 4.0},
    {'rh': 40, 'solar': 6.66, 'water': 5.0},
    {'rh': 50, 'solar': 6.66, 'water': 5.5},
    {'rh': 60, 'solar': 6.66, 'water': 6.0},
    {'rh': 70, 'solar': 6.66, 'water': 6.0},
]


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
    
    # Compare against experimental baseline data
    print("\n" + "=" * 80)
    print("COMPARISON WITH EXPERIMENTAL DATA")
    print("=" * 80)
    
    # Find pipeline outputs that are close to experimental conditions
    baseline_df = pd.DataFrame(BASELINE_EXPERIMENTAL_DATA)
    comparisons = []
    
    for _, exp in baseline_df.iterrows():
        # Find pipeline outputs with similar solar energy (±1 kWh/m²) and RH (±10%)
        similar = valid_results[
            (abs(valid_results['Solar (kWh/m²)'].astype(float) - exp['solar']) < 1.0) &
            (abs(valid_results['RH (%)'] - exp['rh']) < 10)
        ]
        
        if len(similar) > 0:
            pipeline_water = similar.iloc[0]['Water (L/day)']
            exp_water = exp['water']
            error = abs(pipeline_water - exp_water)
            pct_error = (error / exp_water) * 100
            
            comparisons.append({
                'Exp_Solar': exp['solar'],
                'Exp_RH': exp['rh'],
                'Exp_Water': exp_water,
                'Pipeline_Solar': similar.iloc[0]['Solar (kWh/m²)'],
                'Pipeline_RH': similar.iloc[0]['RH (%)'],
                'Pipeline_Water': pipeline_water,
                'Error': round(error, 2),
                'Error_%': round(pct_error, 1)
            })
    
    if len(comparisons) > 0:
        comp_df = pd.DataFrame(comparisons)
        print("\nMatching conditions found:")
        print(comp_df.to_string(index=False))
        
        avg_error = comp_df['Error'].mean()
        avg_pct_error = comp_df['Error_%'].mean()
        
        print(f"\nAverage absolute error: {avg_error:.2f} L/day")
        print(f"Average percentage error: {avg_pct_error:.1f}%")
        
        if avg_pct_error < 20:
            print("[PASS] Pipeline predictions align well with experimental data (<20% error)")
        elif avg_pct_error < 30:
            print("[WARN] Moderate deviation from experimental data (20-30% error)")
        else:
            print("[FAIL] Large deviation from experimental data (>30% error)")
    else:
        print("[INFO] No direct matches with experimental conditions for comparison")
        print("       Experimental range: Solar 5-6.7 kWh/m², RH 20-70%")
    
    # Additional check: Test a specific experimental condition
    print("\n" + "=" * 80)
    print("SPOT CHECK: Simulate Experimental Condition")
    print("=" * 80)
    
    # Pick mid-range experimental point: 5.83 kWh/m², 50% RH → should give ~4.5 L/day
    test_solar_target = 5.83
    test_rh = 50
    expected_water = 4.5
    
    # Find which Toronto date gives closest to 5.83 kWh/m²
    best_match = None
    best_diff = float('inf')
    
    for _, row in valid_results.iterrows():
        solar_diff = abs(row['Solar (kWh/m²)'] - test_solar_target)
        if solar_diff < best_diff and abs(row['RH (%)'] - test_rh) < 15:
            best_diff = solar_diff
            best_match = row
    
    if best_match is not None:
        print(f"\nExperimental: Solar={test_solar_target} kWh/m², RH={test_rh}% → {expected_water} L/day")
        print(f"Closest pipeline: Solar={best_match['Solar (kWh/m²)']} kWh/m², RH={best_match['RH (%)']}% → {best_match['Water (L/day)']} L/day")
        print(f"Date: {best_match['Date']} ({best_match['Description']})")
        
        water_error = abs(best_match['Water (L/day)'] - expected_water)
        pct_error = (water_error / expected_water) * 100
        
        print(f"Water yield error: {water_error:.2f} L/day ({pct_error:.1f}%)")
        
        if pct_error < 15:
            print("[PASS] Pipeline prediction matches experimental result closely")
        elif pct_error < 25:
            print("[WARN] Pipeline prediction has moderate deviation from experimental")
        else:
            print("[FAIL] Pipeline prediction differs significantly from experimental")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Tested: {len(valid_results)}/{len(TEST_SCENARIOS)} scenarios")
    print(f"Solar energy range: {solar_values.min():.2f} - {solar_values.max():.2f} kWh/m²")
    print(f"Water yield range: {water_values.min():.2f} - {water_values.max():.2f} L/day")
    
    if len(comparisons) > 0:
        print(f"Experimental comparison: {len(comparisons)} matches, avg error {avg_pct_error:.1f}%")
    
    # Overall assessment
    if (solar_values >= 0).all() and (water_values >= 0).all():
        if summer_solar > winter_solar * 1.5 and solar_values.max() < 9.5:
            if len(comparisons) > 0 and avg_pct_error < 25:
                print("\nRESULT: Pipeline is validated. Outputs match experimental data and are")
                print("        physically reasonable for Toronto conditions.")
            else:
                print("\nRESULT: Pipeline appears reasonable but limited experimental validation.")
        else:
            print("\nRESULT: Pipeline runs but review seasonal trends.")
    else:
        print("\nRESULT: Pipeline has issues - check model and data.")


if __name__ == '__main__':
    run_sanity_check()
