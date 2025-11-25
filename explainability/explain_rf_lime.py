"""
LIME (Local Interpretable Model-agnostic Explanations) for Random Forest GHI Predictor

What is LIME?
-------------
LIME explains individual predictions by fitting simple, interpretable models
(like linear regression) locally around specific predictions. It perturbs
the input and observes how predictions change.

Why use LIME?
-------------
- Model-agnostic (works with any black-box model)
- Creates human-interpretable explanations using simple models
- Good for explaining specific predictions to non-technical stakeholders
- Shows which features matter most for a particular prediction
- Provides intuitive "if feature X was different, prediction would change by Y"

Best for: Explaining individual predictions to stakeholders or customers,
especially when they want to understand "why did the model predict THIS value
for THIS specific day/time?"
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from lime import lime_tabular
from sklearn.model_selection import train_test_split


def explain_with_lime(model_path: str, csv_path: str, n_training_samples: int = 2000):
    """Create LIME explainer for the RF model.
    
    Args:
        model_path: Path to trained joblib model
        csv_path: Path to training/test data CSV
        n_training_samples: Number of samples for LIME training data
    
    Returns:
        dict with lime_explainer, model, X_train, X_test, y_train, y_test, feature_names
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
    
    # Split data for LIME
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_training_samples, random_state=42
    )
    
    print(f"Creating LIME explainer with {len(X_train)} training samples...")
    
    # Define uniform discretization bins for each feature (makes ranges cleaner)
    # These are interpretable, physics-based boundaries
    discretize_continuous = True
    categorical_features = [0]  # Cloud Type is categorical (0-10 scale)
    
    # Define custom bins for continuous features (uniform, meaningful ranges)
    discretizer = 'quartile'  # Use quartile-based discretization for uniform ranges
    
    # LIME needs training data to understand feature distributions
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=input_features,
        mode='regression',
        training_labels=y_train.values,
        categorical_features=categorical_features,
        discretize_continuous=discretize_continuous,
        discretizer=discretizer,
        verbose=False
    )
    
    print("LIME explainer created with quartile-based discretization!")
    
    return {
        'lime_explainer': explainer,
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': input_features
    }


def generate_lime_report(model_path: str, csv_path: str, output_dir: str,
                         n_training_samples: int = 2000, n_examples: int = 10,
                         n_features: int = 7):
    """Generate comprehensive LIME report with visualizations.
    
    Args:
        model_path: Path to trained joblib model
        csv_path: Path to training/test data CSV
        output_dir: Directory to save visualizations
        n_training_samples: Number of samples for LIME training
        n_examples: Number of individual examples to explain
        n_features: Number of features to show in explanations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    result = explain_with_lime(model_path, csv_path, n_training_samples)
    
    explainer = result['lime_explainer']
    model = result['model']
    X_test = result['X_test']
    y_test = result['y_test']
    feature_names = result['feature_names']
    
    print(f"\nGenerating LIME explanations for {n_examples} examples...")
    
    # Select diverse examples from test set
    y_sorted_idx = np.argsort(y_test.values)
    step = len(y_sorted_idx) // (n_examples + 1)
    example_indices = [y_sorted_idx[i * step] for i in range(1, n_examples + 1)]
    
    all_explanations = []
    
    for i, idx in enumerate(example_indices):
        print(f"Explaining example {i+1}/{n_examples}...")
        
        instance = X_test.iloc[idx].values
        actual_ghi = y_test.iloc[idx]
        predicted_ghi = model.predict([instance])[0]
        
        # Generate LIME explanation
        explanation = explainer.explain_instance(
            instance,
            model.predict,
            num_features=n_features,
            num_samples=5000  # More samples = more accurate local model
        )
        
        # Save as figure
        fig = explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation - LOCAL ANALYSIS (Example {i+1})\n"
                  f"Actual: {actual_ghi:.1f} W/m² | Predicted: {predicted_ghi:.1f} W/m²",
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'lime_example_{i+1}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store explanation details
        all_explanations.append({
            'example_num': i + 1,
            'actual_ghi': float(actual_ghi),
            'predicted_ghi': float(predicted_ghi),
            'features': dict(zip(feature_names, instance)),
            'lime_weights': explanation.as_list(),
            'local_pred': explanation.local_pred[0],
            'intercept': explanation.intercept[1]
        })
    
    # Generate summary report
    print("Generating summary report...")
    summary_path = os.path.join(output_dir, 'lime_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("LIME ANALYSIS SUMMARY - AWG Solar Prediction Model\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Number of Examples Explained: {n_examples}\n")
        f.write(f"Training Samples for LIME: {n_training_samples}\n")
        f.write(f"Features per Explanation: {n_features}\n\n")
        
        for exp in all_explanations:
            f.write("=" * 100 + "\n")
            f.write(f"EXAMPLE {exp['example_num']}\n")
            f.write("=" * 100 + "\n")
            f.write(f"Actual GHI:     {exp['actual_ghi']:>10.2f} W/m²\n")
            f.write(f"Predicted GHI:  {exp['predicted_ghi']:>10.2f} W/m²\n")
            f.write(f"Error:          {exp['predicted_ghi'] - exp['actual_ghi']:>10.2f} W/m²\n\n")
            
            # Define units for features
            feature_units = {
                'Cloud Type': '',
                'Solar Zenith Angle': '°',
                'Relative Humidity': '%',
                'Temperature': '°C',
                'Month': '',
                'Day': '',
                'Hour': ''
            }
            
            f.write("Feature Values:\n")
            f.write("-" * 100 + "\n")
            for feat_name, feat_val in exp['features'].items():
                unit = feature_units.get(feat_name, '')
                f.write(f"  {feat_name:.<40} {feat_val:>8.1f}{unit:>3}\n")
            
            f.write("\nLIME Feature Contributions (Local Linear Model):\n")
            f.write("-" * 100 + "\n")
            f.write("NOTE: LIME fits a simple linear model locally. Large contributions from minor features\n")
            f.write("      (Month, Day, Hour) are usually statistical artifacts, not real physics.\n")
            f.write("      Focus on Solar Zenith Angle and Cloud Type for physical interpretation.\n")
            f.write("-" * 100 + "\n")
            
            # Sort by absolute contribution
            sorted_weights = sorted(exp['lime_weights'], key=lambda x: abs(x[1]), reverse=True)
            
            # Calculate total absolute contribution for percentage
            total_abs = sum(abs(w[1]) for w in sorted_weights)
            
            for feat_desc, weight in sorted_weights:
                direction = "increases" if weight > 0 else "decreases"
                pct = (abs(weight) / total_abs * 100) if total_abs > 0 else 0
                
                # Visual indicator of magnitude
                if abs(weight) > 100:
                    magnitude = "[MAJOR]   "
                elif abs(weight) > 20:
                    magnitude = "[MODERATE]"
                elif abs(weight) > 5:
                    magnitude = "[MINOR]   "
                else:
                    magnitude = "[TINY]    "
                
                f.write(f"  {magnitude} {feat_desc:.<55} {weight:>8.2f} ({pct:>5.1f}%) {direction} GHI\n")
            
            f.write(f"\nLocal Model Intercept: {exp['intercept']:.2f} W/m²\n")
            f.write(f"Local Model Prediction: {exp['local_pred']:.2f} W/m²\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("INTERPRETATION GUIDE:\n")
        f.write("=" * 100 + "\n")
        f.write("LIME creates a simple linear model locally around each prediction.\n\n")
        f.write("How to read the results:\n")
        f.write("- [MAJOR] = Contribution > 100 W/m² (dominant factor)\n")
        f.write("- [MODERATE] = Contribution 20-100 W/m² (secondary factor)\n")
        f.write("- [MINOR] = Contribution 5-20 W/m² (small effect)\n")
        f.write("- [TINY] = Contribution < 5 W/m² (negligible, likely noise)\n\n")
        f.write("- Percentage shows relative contribution to the prediction\n")
        f.write("- Positive values: Feature increases predicted GHI\n")
        f.write("- Negative values: Feature decreases predicted GHI\n")
        f.write("- The feature ranges shown (e.g., '50 < Temp <= 70') indicate the value bin\n")
        f.write("- Local model is only valid near the specific prediction being explained\n\n")
        f.write("IMPORTANT - Understanding LIME Artifacts:\n")
        f.write("-" * 100 + "\n")
        f.write("LIME fits a LINEAR approximation in a small neighborhood. This can cause:\n")
        f.write("- Minor features (Month, Day, Hour) sometimes showing large contributions\n")
        f.write("- These are usually STATISTICAL ARTIFACTS, not real physical effects\n")
        f.write("- The model itself doesn't heavily use these features (see SHAP importance)\n")
        f.write("- Focus on Solar Zenith Angle and Cloud Type for physical interpretation\n\n")
        f.write("Key differences from SHAP:\n")
        f.write("- LIME uses local linear approximation (SHAP uses exact game theory)\n")
        f.write("- LIME is faster but less precise and can have artifacts\n")
        f.write("- LIME explanations are easier for non-technical stakeholders\n")
        f.write("- LIME can vary between runs (uses sampling), SHAP is deterministic\n")
        f.write("- Use SHAP for accurate feature importance, LIME for quick local explanations\n")
    
    # Create comparison plot of all examples
    print("Creating overview plot...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    example_nums = [exp['example_num'] for exp in all_explanations]
    actuals = [exp['actual_ghi'] for exp in all_explanations]
    predicteds = [exp['predicted_ghi'] for exp in all_explanations]
    errors = [p - a for p, a in zip(predicteds, actuals)]
    
    # Plot 1: Actual vs Predicted
    axes[0].plot(example_nums, actuals, 'o-', label='Actual GHI', linewidth=2, markersize=8)
    axes[0].plot(example_nums, predicteds, 's-', label='Predicted GHI', linewidth=2, markersize=8)
    axes[0].set_xlabel('Example Number', fontsize=12)
    axes[0].set_ylabel('GHI (W/m²)', fontsize=12)
    axes[0].set_title('LIME Examples - LOCAL ANALYSIS: Actual vs Predicted GHI', 
                      fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction Errors
    colors = ['red' if e < 0 else 'blue' for e in errors]
    axes[1].bar(example_nums, errors, color=colors, alpha=0.6)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Example Number', fontsize=12)
    axes[1].set_ylabel('Prediction Error (W/m²)', fontsize=12)
    axes[1].set_title('LIME Examples: Prediction Errors (Blue=Over-predicted, Red=Under-predicted)', 
                      fontsize=13, pad=15)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lime_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ LIME analysis complete!")
    print(f"✓ Generated {len(os.listdir(output_dir))} files in {output_dir}")
    print(f"✓ Summary saved to: {summary_path}")
    
    return all_explanations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate LIME explanations for RF model')
    parser.add_argument('--model', required=True, help='Path to trained model (joblib)')
    parser.add_argument('--csv', required=True, help='Path to training/test CSV data')
    parser.add_argument('--output-dir', default='lime_output', help='Output directory for visualizations')
    parser.add_argument('--n-training', type=int, default=2000,
                        help='Number of training samples for LIME (default: 2000)')
    parser.add_argument('--n-examples', type=int, default=10,
                        help='Number of examples to explain (default: 10)')
    parser.add_argument('--n-features', type=int, default=7,
                        help='Number of features to show in explanations (default: 7)')
    args = parser.parse_args()
    
    generate_lime_report(args.model, args.csv, args.output_dir, 
                         args.n_training, args.n_examples, args.n_features)
