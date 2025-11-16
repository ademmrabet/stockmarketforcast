import pandas as pd
import numpy as np
import logging

# module-level logger configuration
logger = logging.getLogger(__name__)

def clean_percentage(df: pd.DataFrame, column: str = "chg_%") -> pd.DataFrame:
    df = df.copy()

    #Remove the % symbol
    df[column] = df[column].str.replace('%', '', regex=False)
    # Remove Whitespace
    df[column] = df[column].str.strip()
    # Convert to float
    df[column] = df[column].astype(float)
    # convert from % to decimal
    df[column] = df[column]/100

    return df


def clean_volume(df: pd.DataFrame, column: str = "vol_") -> pd.DataFrame:
    df = df.copy()
    
    def convert_vol(value):
        if pd.isna(value):
            return np.nan
        
        value = str(value).replace('.', '').strip()

        # Millions
        if value.endswith('M'):
            return float(value[:-1]) * 1_000_000
        if value.endswith('K'):
            return float(value[:-1]) * 1_000
        
        return float(value)
    
    df[column] = df[column].apply(convert_vol)

    return df


def clean_timestamp(
        df: pd.DataFrame,
        column: str = "timestamp",
        sort: bool = True
) -> pd.DataFrame:
    
    df = df.copy()

    # 1. Convert to datetime
    df[column] = pd.to_datetime(df[column], errors="coerce")

    # 2. logging if some timestamps failed to parse
    n_missing = df[column].isna().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} rows have invalid '{column}' values after conversion")

    # 3. we will sort it chronologically
    if sort:
        df = df.sort_values(by=column).reset_index(drop=True)

    return df

def drop_unused_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:


    logger.info(f"Dropping columns (if present): {columns_to_drop}")

    df=df.copy()

    # we will keep only the columns that actually exist in the dataframe
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    if not existing_cols:
        logger.info("No matching columns to drop.")
        return df
    
    df = df.drop(columns=existing_cols)

    return df
