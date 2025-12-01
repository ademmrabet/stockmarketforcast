import os
from pathlib import Path
import pandas as pd
from src.cleaning import(
    clean_percentage,
    clean_timestamp,
    clean_volume,
    drop_unused_columns
)
from src.features import (
    add_time_features,
    add_price_features,
    add_volume_features,
    build_features_dataset
)
from src.target import add_target_next_hour
from src.model import (
    load_feature_data,
    train_all_models
)

# Path config

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
PROCESSED_DATA_DIR = DATA_DIR / "02_processed"
STOCKS_RAW_FILE = RAW_DATA_DIR / "stocks.csv"
STOCKS_PROCESSED_FILE = PROCESSED_DATA_DIR / "stocks_clean.csv"
STOCK_FEATURES_FILE = PROCESSED_DATA_DIR / "stocks_features.csv"


# Pipeline

def run_cleaning_pipeline():
    print("\n=== RUNNING STOCKS CLEANING PIPELINE ===")

    if not STOCKS_RAW_FILE.exists():
        raise FileNotFoundError(f"Raw stocks file not found at {STOCKS_RAW_FILE}")

    print(f"[+] Loading raw data: {STOCKS_RAW_FILE}")
    df = pd.read_csv(STOCKS_RAW_FILE)

    # Apply cleaning functions from cleaning.py
    df = clean_percentage(df, column="chg_%")
    df = clean_volume(df, column="vol_")
    df = clean_timestamp(df, column="timestamp", sort=True)
    df = drop_unused_columns(df, ["time"])    # remove the useless 'time' column

    # Ensure the folder exists before saving
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save cleaned output
    df.to_csv(STOCKS_PROCESSED_FILE, index=False)
    print(f"Cleaned CSV saved to: {STOCKS_PROCESSED_FILE}")

    print("=== CLEANING PIPELINE FINISHED ===\n")


# Feature Engineering Pipeline
def run_feature_pipeline():
    print("\n=== RUNNING STOCKS FEATURE ENGINEERING PIPELINE ===")
    # Here we will check that the cleaned data exists
    if not STOCKS_PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Processed stock file not found at {STOCKS_PROCESSED_FILE}."
            " Please run the cleaning pipeline first."
        )
    # Assume the cleaned data exists, we load it
    print(f"[+] Loading cleaned data: {STOCKS_PROCESSED_FILE}")
    df = pd.read_csv(STOCKS_PROCESSED_FILE)
    # And let the show begin
    print("[+] Building feature dataset...")
    df = build_features_dataset(df)

    df = add_target_next_hour(df)

    df = df.dropna(subset=[
        "future_last_1h",
        "future_return_1h",
        "movement_1h"
    ])

    # let's ensure the folder exists before saving
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(STOCK_FEATURES_FILE, index=False)
    print(f" Feature engineered CSV saved to: {STOCK_FEATURES_FILE}")

    #and that's all folks
    print("=== FEATURE ENGINEERING PIPELINE FINISHED ===\n")

def run_model_pipeline():
    print("\n=== Running Model Training Pipeline ===")

    df = load_feature_data(STOCK_FEATURES_FILE)
    metrics = train_all_models(df)

    print("\n=== Model Performance ===")
    print(metrics)
# Main
if __name__ == "__main__":
    run_cleaning_pipeline() # this will be our step 1 : cleaning the raw data
    run_feature_pipeline()  # this will be our step 2 : feature engineering
    
    print("\n=== RUNNING MODEL TRAINING PIPELINE ===")
    df_features = load_feature_data(STOCK_FEATURES_FILE)
    results = train_all_models(df_features)

    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    print("Logistic (thr=0.5):", results["logistic_default"])
    print("RandomForest (thr=0.5):", results["rf_default"])
    print("XGBoost (thr=0.5):", results["xgb_default"])
    print("\nXGBoost Backtest (thr=0.7):", results["xgb_backtest"])