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
    
    feature_cols = [col for col in df.columns if col not in target_cols and col != 'pm2_5']

    y = df[target_cols]
    X = df[feature_cols]

    # 3. Split Data (Time-series split, no shuffle)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

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

    # Calculate and print metrics for key horizons
    rmse_t1 = np.sqrt(mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]))
    mae_t1 = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
    
    rmse_t24 = np.sqrt(mean_squared_error(y_test.iloc[:, 23], y_pred[:, 23]))
    mae_t24 = mean_absolute_error(y_test.iloc[:, 23], y_pred[:, 23])

    rmse_t72 = np.sqrt(mean_squared_error(y_test.iloc[:, 71], y_pred[:, 71]))
    mae_t72 = mean_absolute_error(y_test.iloc[:, 71], y_pred[:, 71])

    print("\nModel Evaluation Results:")
    print(f"  - Horizon t+1:  RMSE = {rmse_t1:.4f}, MAE = {mae_t1:.4f}")
    print(f"  - Horizon t+24: RMSE = {rmse_t24:.4f}, MAE = {mae_t24:.4f}")
    print(f"  - Horizon t+72: RMSE = {rmse_t72:.4f}, MAE = {mae_t72:.4f}\n")

    # 7. Store Model
    model_path = 'models/aqi_model.joblib'
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
