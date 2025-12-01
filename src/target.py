import pandas as pd
import numpy as np
import logging

# Logging configuration 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# This Python file is for target engineering...
# Our main target will be the price movement direction in the next hour
# We will be generating 2 target columbns :
# 1. 'future_return_1h' : pure finance return. Useful for metrics, evaluation, and maybe regression later on
# 2. 'movement_1h' : Binary classification target (1: price up, 0: price down) -> This will be our main ML target

def add_target_next_hour(
        df: pd.DataFrame,
        last_col: str = "last",
        group_col: str = "name",
        timestamp_col: str = "timestamp",
        threshold: float = 0.0005
) -> pd.DataFrame:
    logger.info("Adding next-hour target (future_return_1h & movement_1h)")

    # Like always, let's not forget to copy the dataframe to avoid modifying the original one
    df = df.copy()

    # Basic Validation 101
    required_cols = [last_col, group_col, timestamp_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in DataFrame: {missing}")
    # let's ensure proper ordering
    df = df.sort_values([group_col, timestamp_col])

    # baby step 1: Next hour price (feature value)
    df["future_last_1h"] = (
        df.groupby(group_col)[last_col]
        .shift(-1) # Now the FUTURE value will become the current present value
    )

    # baby step 2: Future return calculation
    df["future_return_1h"] = (
        (df["future_last_1h"] - df[last_col]) / df[last_col]
    )

    # Baby step 3: let's classify the movement to 1 or 0
    df["movement_1h"] = (df["future_return_1h"] > threshold).astype(int)

    # final baby step : let's drop the rows where the future price doesn't exist ( end of each stock)
    df = df.dropna(subset=["future_last_1h", "future_return_1h"])

    logger.info("Target engineering completed successfully.")
    logger.info(f"Class Distribution: \n{df['movement_1h'].value_counts(normalize=True)}")

    return df