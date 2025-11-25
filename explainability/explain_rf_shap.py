"""
SHAP (SHapley Additive exPlanations) for Random Forest GHI Predictor

What is SHAP?
-------------
SHAP uses game theory (Shapley values) to explain predictions by computing
how much each feature contributes to the prediction. It provides:
- Feature importance across all predictions (global)
- Individual prediction explanations (local)
- Feature interaction effects
- Consistent and theoretically grounded explanations

Why use SHAP?
-------------
- Gold standard for model explainability
- Shows both positive and negative feature contributions
- Handles feature interactions well
- Works with any ML model
- Backed by solid mathematical foundation (Shapley values from cooperative game theory)

Best for: Understanding why the model makes certain predictions and
identifying which weather conditions drive solar energy predictions.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split


def explain_with_shap(model_path: str, csv_path: str, n_samples: int = 1000):
    """Generate SHAP explanations for the RF model.
    
    Args:
        model_path: Path to trained joblib model
        csv_path: Path to training/test data CSV
        n_samples: Number of samples to use for SHAP analysis (default: 1000)
                  Using all data can be slow; sampling is recommended
    
    Returns:
        dict with shap_values, explainer, X_sample, y_sample, feature_names
    """
    print("Loading model and data...")
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    
    input_features = [
        'Cloud Type',
        'Solar Zenith Angle',
        'Relative Humidity',
        'Temperature',
        'Month',
        'Day',
        'Hour'
    ]
    target_variable = 'GHI'
    
    X = df[input_features]
    y = df[target_variable]
    
    # Sample data if needed (SHAP can be computationally expensive)
    if len(X) > n_samples:
        print(f"Sampling {n_samples} from {len(X)} samples for SHAP analysis...")
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=n_samples, random_state=42, stratify=None
        )
    else:
        X_sample = X
        y_sample = y
    
    print(f"Creating SHAP TreeExplainer (optimized for Random Forest)...")
    # TreeExplainer is fast and exact for tree-based models
    explainer = shap.TreeExplainer(model)
    
    print(f"Computing SHAP values for {len(X_sample)} samples...")
    shap_values = explainer.shap_values(X_sample)
    
    print("SHAP analysis complete!")
    
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'X_sample': X_sample,
        'y_sample': y_sample,
        'feature_names': input_features,
        'base_value': explainer.expected_value
    }


def generate_shap_report(model_path: str, csv_path: str, output_dir: str, 
                         n_samples: int = 1000, n_examples: int = 5):
    """Generate comprehensive SHAP report with visualizations.
    
    Args:
        model_path: Path to trained joblib model
        csv_path: Path to training/test data CSV
        output_dir: Directory to save visualizations
        n_samples: Number of samples for SHAP analysis
        n_examples: Number of individual examples to explain in detail
    """
    os.makedirs(output_dir, exist_ok=True)
    
    result = explain_with_shap(model_path, csv_path, n_samples)
    
    shap_values = result['shap_values']
    X_sample = result['X_sample']
    y_sample = result['y_sample']
    feature_names = result['feature_names']
    base_value = result['base_value']
    
    print(f"\nGenerating SHAP visualizations in {output_dir}...")
    
    # 1. Summary Plot (Bar) - Global Feature Importance
    print("1. Creating feature importance bar plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance - GLOBAL ANALYSIS\nAverage impact on model output across all predictions", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_shap_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Summary Plot (Beeswarm) - Feature Impact Distribution
    print("2. Creating beeswarm summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary Plot - GLOBAL ANALYSIS\nFeature impact distribution (red=high feature value, blue=low)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_shap_summary_beeswarm.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Dependence Plots for top features
    print("3. Creating dependence plots for top features...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[::-1][:3]
    
    for i, feat_idx in enumerate(top_features_idx):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feat_idx, shap_values, X_sample, show=False)
        plt.title(f"SHAP Dependence Plot: {feature_names[feat_idx]}\nShows how feature value affects predictions", 
                  fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'3_{i+1}_dependence_{feature_names[feat_idx].replace(" ", "_")}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Waterfall plots for individual predictions
    print(f"4. Creating waterfall plots for {n_examples} example predictions...")
    
    # Select diverse examples (low, medium, high GHI)
    y_sorted_idx = np.argsort(y_sample.values)
    example_indices = [
        y_sorted_idx[0],  # Lowest GHI
        y_sorted_idx[len(y_sorted_idx)//4],  # 25th percentile
        y_sorted_idx[len(y_sorted_idx)//2],  # Median
        y_sorted_idx[3*len(y_sorted_idx)//4],  # 75th percentile
        y_sorted_idx[-1]  # Highest GHI
    ][:n_examples]
    
    # Define units for each feature
    feature_units = {
        'Cloud Type': '',
        'Solar Zenith Angle': '°',
        'Relative Humidity': '%',
        'Temperature': '°C',
        'Month': '',
        'Day': '',
        'Hour': ''
    }
    
    for i, idx in enumerate(example_indices):
        plt.figure(figsize=(12, 8))
        
        # Create custom feature names with values and units
        feature_vals = X_sample.iloc[idx]
        custom_names = []
        for feat in feature_names:
            val = feature_vals[feat]
            unit = feature_units.get(feat, '')
            custom_names.append(f"{feat}\n[{val:.1f}{unit}]")
        
        explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=base_value,
            data=X_sample.iloc[idx].values,
            feature_names=custom_names
        )
        shap.waterfall_plot(explanation, show=False, max_display=10)
        actual_ghi = y_sample.iloc[idx]
        plt.title(f"SHAP Waterfall Plot - LOCAL EXPLANATION (Example {i+1})\nActual GHI: {actual_ghi:.1f} W/m²", 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'4_{i+1}_waterfall_example_{i+1}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. Force plot for single prediction (save as image)
    print("5. Creating force plot...")
    idx = example_indices[len(example_indices)//2]  # Use median example
    plt.figure(figsize=(20, 3))
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=base_value,
        data=X_sample.iloc[idx].values,
        feature_names=feature_names
    )
    shap.plots.force(explanation, matplotlib=True, show=False)
    actual_ghi = y_sample.iloc[idx]
    plt.title(f"SHAP Force Plot - LOCAL EXPLANATION\nActual GHI: {actual_ghi:.1f} W/m²", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_shap_force_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Generate summary statistics
    print("6. Generating summary statistics...")
    summary_stats = {
        'base_value': float(base_value),
        'n_samples_analyzed': len(X_sample),
        'feature_importance': {}
    }
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    for feat_name, importance in zip(feature_names, mean_abs_shap):
        summary_stats['feature_importance'][feat_name] = float(importance)
    
    # Sort by importance
    summary_stats['feature_importance'] = dict(
        sorted(summary_stats['feature_importance'].items(), 
               key=lambda x: x[1], reverse=True)
    )
    
    # Save summary to text file
    summary_path = os.path.join(output_dir, 'shap_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SHAP ANALYSIS SUMMARY - AWG Solar Prediction Model\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Base Value (Expected prediction): {float(base_value):.2f} W/m²\n")
        f.write(f"Samples Analyzed: {len(X_sample):,}\n\n")
        
        f.write("FEATURE IMPORTANCE (Mean |SHAP value|):\n")
        f.write("-" * 80 + "\n")
        for feat_name, importance in summary_stats['feature_importance'].items():
            f.write(f"{feat_name:.<40} {importance:>10.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION GUIDE:\n")
        f.write("=" * 80 + "\n")
        f.write("- Base value: Average model prediction across all training data\n")
        f.write("- SHAP values: How much each feature pushes prediction up/down from base\n")
        f.write("- Positive SHAP: Feature increases predicted GHI\n")
        f.write("- Negative SHAP: Feature decreases predicted GHI\n")
        f.write("- Feature importance: Average magnitude of impact across all predictions\n\n")
        
        f.write("VISUALIZATION GUIDE:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Feature Importance Bar: Which features matter most overall\n")
        f.write("2. Beeswarm Plot: Distribution of feature impacts (color = feature value)\n")
        f.write("3. Dependence Plots: How feature values affect predictions\n")
        f.write("4. Waterfall Plots: Step-by-step explanation of individual predictions\n")
        f.write("5. Force Plot: Visual breakdown of a single prediction\n")
    
    print(f"\n✓ SHAP analysis complete!")
    print(f"✓ Generated {len(os.listdir(output_dir))} files in {output_dir}")
    print(f"✓ Summary saved to: {summary_path}")
    
    return summary_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SHAP explanations for RF model')
    parser.add_argument('--model', required=True, help='Path to trained model (joblib)')
    parser.add_argument('--csv', required=True, help='Path to training/test CSV data')
    parser.add_argument('--output-dir', default='shap_output', help='Output directory for visualizations')
    parser.add_argument('--n-samples', type=int, default=1000, 
                        help='Number of samples for SHAP analysis (default: 1000)')
    parser.add_argument('--n-examples', type=int, default=5,
                        help='Number of individual examples to explain (default: 5)')
    args = parser.parse_args()
    
    generate_shap_report(args.model, args.csv, args.output_dir, args.n_samples, args.n_examples)
