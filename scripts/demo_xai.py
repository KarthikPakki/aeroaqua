"""
Quick example script showing how to use SHAP and LIME programmatically.

Run this to see examples of both methods in action!
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from aeroaqua.explainability import explain_with_shap, explain_with_lime
import matplotlib.pyplot as plt


def quick_shap_demo(model_path: str, csv_path: str):
    """Quick SHAP demonstration."""
    print("=" * 80)
    print("SHAP QUICK DEMO")
    print("=" * 80)
    
    result = explain_with_shap(model_path, csv_path, n_samples=100)
    
    print("\n📊 Feature Importance (Mean |SHAP value|):")
    print("-" * 80)
    
    import numpy as np
    mean_abs_shap = np.abs(result['shap_values']).mean(axis=0)
    
    for feat_name, importance in zip(result['feature_names'], mean_abs_shap):
        bar = "█" * int(importance / 10)
        print(f"{feat_name:.<40} {importance:>8.2f} {bar}")
    
    print(f"\nBase Value (Expected prediction): {result['base_value']:.2f} W/m²")
    print("\n✓ SHAP analysis complete!")
    print("  For full visualizations, run: python scripts/generate_xai_report.py")


def quick_lime_demo(model_path: str, csv_path: str):
    """Quick LIME demonstration."""
    print("\n" + "=" * 80)
    print("LIME QUICK DEMO")
    print("=" * 80)
    
    result = explain_with_lime(model_path, csv_path, n_training_samples=500)
    
    # Explain one example
    explainer = result['lime_explainer']
    model = result['model']
    X_test = result['X_test']
    y_test = result['y_test']
    
    # Pick middle example
    idx = len(X_test) // 2
    instance = X_test.iloc[idx].values
    actual_ghi = y_test.iloc[idx]
    predicted_ghi = model.predict([instance])[0]
    
    print(f"\n📍 Explaining Example Prediction:")
    print("-" * 80)
    print(f"Actual GHI:    {actual_ghi:.2f} W/m²")
    print(f"Predicted GHI: {predicted_ghi:.2f} W/m²")
    print(f"Error:         {predicted_ghi - actual_ghi:.2f} W/m²")
    
    print("\n🔍 Generating LIME explanation (this may take 10-20 seconds)...")
    explanation = explainer.explain_instance(
        instance,
        model.predict,
        num_features=7,
        num_samples=1000
    )
    
    print("\n📊 Feature Contributions:")
    print("-" * 80)
    for feat_desc, weight in explanation.as_list():
        direction = "↑ increases" if weight > 0 else "↓ decreases"
        print(f"{feat_desc:.<60} {weight:>8.2f} {direction} GHI")
    
    print("\n✓ LIME analysis complete!")
    print("  For full visualizations, run: python scripts/generate_xai_report.py")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick XAI demo')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--csv', required=True, help='Path to CSV data')
    parser.add_argument('--method', choices=['shap', 'lime', 'both'], default='both',
                        help='Which method to demo')
    
    args = parser.parse_args()
    
    if args.method in ['shap', 'both']:
        quick_shap_demo(args.model, args.csv)
    
    if args.method in ['lime', 'both']:
        quick_lime_demo(args.model, args.csv)
    
    print("\n" + "=" * 80)
    print("For complete report with visualizations, run:")
    print("  python scripts/generate_xai_report.py --model <model> --csv <csv>")
    print("=" * 80)
