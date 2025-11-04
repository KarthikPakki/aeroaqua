"""Evaluate a trained RandomForest model on test data."""
import argparse
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def evaluate_model(model_path: str, csv_path: str, output_dir: str = None):
    """Evaluate trained model on test dataset.

    Args:
        model_path: path to trained joblib model
        csv_path: path to CSV file with test data
        output_dir: optional directory to save evaluation plots

    Returns:
        dict with evaluation metrics
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading test data from: {csv_path}")
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

    missing = [c for c in input_features + [target_variable] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    X = df[input_features]
    y = df[target_variable]

    print(f"\nEvaluating on {len(X)} samples...")
    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"\nEvaluation Metrics:")
    print(f"  RMSE: {rmse:.2f} W/m²")
    print(f"  MAE: {mae:.2f} W/m²")
    print(f"  R²: {r2:.4f}")

    # Error analysis
    errors = y_pred - y
    print(f"\nError Statistics:")
    print(f"  Mean Error: {errors.mean():.2f} W/m²")
    print(f"  Std Error: {errors.std():.2f} W/m²")
    print(f"  Max Error: {errors.max():.2f} W/m²")
    print(f"  Min Error: {errors.min():.2f} W/m²")

    # Save plots if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Predicted vs Actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, alpha=0.3, s=1)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual GHI (W/m²)')
        plt.ylabel('Predicted GHI (W/m²)')
        plt.title(f'Predicted vs Actual GHI (R²={r2:.4f})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=150)
        print(f"\nSaved plot: {output_dir}/predicted_vs_actual.png")
        
        # Residual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, errors, alpha=0.3, s=1)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted GHI (W/m²)')
        plt.ylabel('Residual (W/m²)')
        plt.title('Residual Plot')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'residuals.png'), dpi=150)
        print(f"Saved plot: {output_dir}/residuals.png")
        
        plt.close('all')

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mean_error': float(errors.mean()),
        'std_error': float(errors.std()),
        'n_samples': len(X)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained RF model')
    parser.add_argument('--model', required=True, help='Path to trained model (joblib)')
    parser.add_argument('--csv', required=True, help='Path to test CSV data')
    parser.add_argument('--output-dir', help='Directory to save evaluation plots')
    args = parser.parse_args()

    evaluate_model(args.model, args.csv, args.output_dir)
