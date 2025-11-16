import os
from pathlib import Path
import pandas as pd
from src.cleaning import(
    clean_percentage,
    clean_timestamp,
    clean_volume,
    drop_unused_columns
)

# Path config

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
PROCESSED_DATA_DIR = DATA_DIR / "02_processed"
STOCKS_RAW_FILE = RAW_DATA_DIR / "stocks.csv"
STOCKS_PROCESSED_FILE = PROCESSED_DATA_DIR / "stocks_clean.csv"


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
    print(f"[✓] Cleaned CSV saved to: {STOCKS_PROCESSED_FILE}")

    print("=== CLEANING PIPELINE FINISHED ===\n")


# Main
if __name__ == "__main__":
    run_cleaning_pipeline()