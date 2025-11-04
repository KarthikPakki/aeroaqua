import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np


def train_and_save(csv_path: str, model_path: str = None, test_size: float = 0.2, cv_folds: int = 5):
    """Train RandomForest on provided CSV and save model.

    Expects the CSV to contain these columns:
        'Cloud Type', 'Solar Zenith Angle', 'Relative Humidity', 'Temperature', 'Month', 'Day', 'Hour', 'GHI'

    Args:
        csv_path: path to the CSV file used to train the model.
        model_path: path to save the trained joblib model. If None, saves to model/solar_predictor_model.joblib
        test_size: fraction of data to use for testing (default: 0.2)
        cv_folds: number of cross-validation folds (default: 5)

    Returns:
        dict with model_path and evaluation metrics
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'solar_predictor_model.joblib')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    print("\nTraining RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Cross-validation on training set
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv_folds, 
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    cv_rmse = np.sqrt(-cv_scores)
    
    print(f"CV RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")

    # Test set evaluation
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test R²: {test_r2:.4f}")

    # Feature importances
    print("\nFeature Importances:")
    for feat, imp in zip(input_features, model.feature_importances_):
        print(f"  {feat}: {imp:.4f}")

    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return {
        'model_path': model_path,
        'cv_rmse_mean': float(cv_rmse.mean()),
        'cv_rmse_std': float(cv_rmse.std()),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'feature_importances': dict(zip(input_features, model.feature_importances_.tolist()))
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and save RandomForest GHI predictor')
    parser.add_argument('--csv', required=True, help='Path to usaWithWeather.csv (training data)')
    parser.add_argument('--out', required=False, help='Output model path (joblib)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction (default: 0.2)')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds (default: 5)')
    args = parser.parse_args()

    train_and_save(args.csv, args.out, args.test_size, args.cv_folds)
