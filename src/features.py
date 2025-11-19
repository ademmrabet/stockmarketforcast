import pandas as pd
import numpy as np
import logging

# Logging configuration 
logger = logging.getLogger(__name__)

# This Python file is for features engineering
# Our main targets are going to be : 

def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame :
    logger.info(f"Adding time features from column: {timestamp_col}")

    df=df.copy()

    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found in dataframe")
    
    # let's ensure the column is datetime (in case it's still string for some reason)

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

    # we set basic time components :
    df["hour"] = df[timestamp_col].dt.hour
    df["weekday"] = df[timestamp_col].dt.weekday
    df["month"] = df[timestamp_col].dt.month

    # Let's not forget about weekends. Weekend flag : 1 if saturday (5) or sunday (6), else 0
    df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)

    return df

# So in this part we will be adding price-based features such as intrahour range, lagged prices, returns, and rolling statistics, computed per asset
def add_price_features(
        df: pd.DataFrame,
        last_col: str = "last",
        high_col: str = "high",
        low_col: str = "low",
        group_col: str ="name",
        timestamp_col: str ="timestamp",
        ma_windows: tuple = (3,6),
) -> pd.DataFrame:
    logger.info(
        f"Adding price features using last = '{last_col}', high='{high_col}',"
        f"low = '{low_col}', grouped by '{group_col}'"
    )

    df = df.copy() # don't make my mistake and forget this like i did in the earlier feature

    # So like always basic validation is required.
    required_cols = [last_col,high_col,low_col,group_col,timestamp_col]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns in DataFrame: {missing}")
    
    # Okay so now let's ensure proper time ordering within each asset
    df = df.sort_values(by=[group_col, timestamp_col])

    # first feature -> Intrahour range : high - low
    df["high_low_range"] = df[high_col] - df[low_col]

    # Second feature -> Lagged last price and 1-hour return per asset
    df["last_lag_1"] = (
        df.groupby(group_col)[last_col]
        .shift(1)
    )

    # NOTE : let's not forget to avoid division by 0:  we will let pandas handle inf/NaN, can clean later
    df["return_1h"] = (df[last_col] - df["last_lag_1"])/ df["last_lag_1"]

    # Third feature : Rolling moving averages and volatility per asset

    for w in ma_windows:
        # here we will try to find the moving average of 'last' for windows w
        ma_col = f"ma_last_{w}h"
        vol_col= f"volatility_{w}h"

        logger.info(f"Computing rolling features with windows={w} for column '{last_col}'.")

        rolled = (
            df.groupby(group_col)[last_col]
            .rolling(window=w, min_periods=1)
        )

        # Rolling mean
        df[ma_col] = rolled.mean().reset_index(level=0, drop=True)

        # Rolling std ( volatility)
        df[vol_col] = rolled.std(ddof=0).reset_index(level=0,drop=True)

    return df


# This function takes cleaned stock dataframe and adds volume-realated features per stock name, like : 
# previous_volume, volume_change and rolling_average_volume
def add_volume_features(
        df: pd.DataFrame,
        vol_col : str = "vol_",
        group_col : str = "name",
        timestamp_col : str = "timestamp",
        ma_windows : tuple = (3,6)
) -> pd.DataFrame:
    logger.info(
        f"Adding Volume features using volume = {vol_col},groupes = {group_col}"
    )

    df = df.copy()

    # Basic validation
    required_cols = [vol_col,group_col, timestamp_col]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns in dataframe: {missing}")
    
    # Let's also ensurre proper time ordering within each asset
    df = df.sort_values(by=[group_col,timestamp_col])

    # Volume feature 1 :  Lagged Volume and 1-Hour volume change (per stock)
    df["vol_lag_1"]=(
        df.groupby(group_col)[vol_col]
        .shift(1)
    )

    df["vol_change_1h"] = (df[vol_col] - df["vol_lag_1"]) / df["vol_lag_1"]

    for w in ma_windows:
        vol_ma_col = f"vol_ma_{w}h"

        logger.info(f"Computing rolling volume mean with window={w} for '{vol_col}'.")

        rolled = (
            df.groupby(group_col)[vol_col]
            .rolling(window=w, min_periods=1)
        )

        df[vol_ma_col] = rolled.mean().reset_index(level=0, drop=True)

    return df



def build_features_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_price_features(df)
    df = add_volume_features(df)

    df = df.dropna(subset=["vol_lag_1", "vol_change_1h", "vol_ma_3h", "vol_ma_6h", "last_lag_1", "return_1h", "high_low_range"])

    return df