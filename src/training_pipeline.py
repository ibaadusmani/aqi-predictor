import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def main():
    """Main function to run the training pipeline."""
    # 1. Load Data
    print("Loading processed data...")
    try:
        df = pd.read_parquet('data/processed/features.parquet')
    except FileNotFoundError:
        print("Error: features.parquet not found. Please run the feature pipeline first.")
        return

    # 2. Prepare Data for Training
    print("Preparing data for training...")
    target_cols = [f'pm25_t+{i}' for i in range(1, 73)]
    # Ensure all target columns exist, in case the feature engineering changed
    target_cols = [col for col in target_cols if col in df.columns]
    
    # Exclude non-numeric columns (timestamp, dt) and target columns
    exclude_cols = target_cols + ['pm2_5', 'timestamp', 'dt']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    y = df[target_cols]
    X = df[feature_cols]
    
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Targets: {len(target_cols)} columns")

    # 3. Split Data (Time-series split, no shuffle)
    print("Splitting data...")
    # First, remove rows where ALL target columns are NaN (last 72 hours)
    valid_rows = y.notna().any(axis=1)
    X_valid = X[valid_rows]
    y_valid = y[valid_rows]
    
    print(f"  Total rows: {len(X)}")
    print(f"  Valid rows (with at least one target): {len(X_valid)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, shuffle=False)

    # 4. Scale Data
    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    scaler_path = 'models/scaler.joblib'
    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)

    # 5. Train Model
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, max_depth=15, min_samples_leaf=5, min_samples_split=10)
    model.fit(X_train_scaled, y_train)

    # 6. Evaluate Model
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)

    # Remove rows with NaN in targets for evaluation
    # For each horizon, only evaluate rows that have valid target values
    def calc_metrics_for_horizon(y_true_col, y_pred_col, horizon_name):
        """Calculate metrics for a specific horizon, ignoring NaN values."""
        valid_mask = y_true_col.notna()
        if valid_mask.sum() == 0:
            return None, None
        y_true_valid = y_true_col[valid_mask]
        y_pred_valid = y_pred_col[valid_mask]
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        return rmse, mae
    
    # Calculate and print metrics for key horizons
    rmse_t1, mae_t1 = calc_metrics_for_horizon(y_test.iloc[:, 0], y_pred[:, 0], "t+1")
    rmse_t24, mae_t24 = calc_metrics_for_horizon(y_test.iloc[:, 23], y_pred[:, 23], "t+24")
    rmse_t72, mae_t72 = calc_metrics_for_horizon(y_test.iloc[:, 71], y_pred[:, 71], "t+72")

    print("\nModel Evaluation Results:")
    if rmse_t1:
        print(f"  - Horizon t+1:  RMSE = {rmse_t1:.4f}, MAE = {mae_t1:.4f}")
    if rmse_t24:
        print(f"  - Horizon t+24: RMSE = {rmse_t24:.4f}, MAE = {mae_t24:.4f}")
    if rmse_t72:
        print(f"  - Horizon t+72: RMSE = {rmse_t72:.4f}, MAE = {mae_t72:.4f}")
    print()

    # 7. Store Model
    model_path = 'models/aqi_model.joblib'
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
