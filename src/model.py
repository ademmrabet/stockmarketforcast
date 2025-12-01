import os
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Let's start building our modeling pipeline
#  1. Loading the feature Dataset

def load_feature_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading feature dataset from : {path}")
    df = pd.read_csv(path)
    return df

# 2. Time based Split
def time_series_split(
        df: pd.DataFrame,
        time_col: str = "timestamp",
        test_size: float = 0.2,
):
    # We will be splitting based on time -> chronological split: (1-test_size) for train, last test_size for test
    # like always, let's copy the dataframe to avoid modifying the original one
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"time series split : train={len(train_df)} rows, test={len(test_df)} rows.")

    return train_df, test_df

# 3. Let's choose our features and target
def get_features_and_target(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
):
    # We will be automatically selecting all numeric columns, excluding targets / future info.

    target_col = "movement_1h"

    # Collect numerical columns
    numerica_train = train_df.select_dtypes(include=["number"])
    numeric_test = test_df.select_dtypes(include=["number"])

    # columns to execlude from features
    exclude_cols = [
        target_col,
        "future_return_1h",
        "future_last_1h",
    ]

    feature_cols = [
        c for c in numerica_train.columns
        if c not in exclude_cols
    ]

    x_train = numerica_train[feature_cols]
    y_train = train_df[target_col]

    x_test = numeric_test[feature_cols]
    y_test = test_df[target_col]

    logger.info(f"Using {len(feature_cols)} feature columns.")
    logger.info(f"Feature cols: {feature_cols}")

    return x_train, y_train,x_test, y_test, feature_cols


# 4. Evaluation Helpers
def evaluate_at_threshold(y_true, y_proba, threshold: float = 0.5):
    # We will conver probabilities into labels using a given threshold and calculate metrics
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def scan_thresholds(y_true, y_proba, thresholds=None):
    # Scan multiple threshold and return a small table of metrics
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    rows = []
    for thr in thresholds:
        rows.append(evaluate_at_threshold(y_true, y_proba, thr))
    
    return pd.DataFrame(rows)

# 5. simple Long-only backtest
def simple_long_only_backtest(
        df_test: pd.DataFrame,
        y_proba: np.ndarray,
        threshold: float = 0.7,
        return_col: str = "future_return_1h"
):
    # This is a very simple strategy : 
    # If P(up) >= THRESHOLD -> We go long (buy) for the next hour
    # Else : We stay out
    # Profit = sum of realized returns on long positions

    if return_col not in df_test.columns:
        raise KeyError(f"Backtest needs column '{return_col}' in test dataframe.")
    
    # Decision : 1 if we take a long position
    decisions = (y_proba >= threshold).astype(int)

    future_returns = df_test[return_col].to_numpy()

    # Strategy returns = future return only when we are long, else 0
    strategy_returns = decisions * future_returns

    n_trades = decisions.sum()
    if n_trades > 0:
        hit_rate = (strategy_returns > 0).sum() / n_trades
        avg_return_per_trade = strategy_returns[strategy_returns != 0].mean()
    else:
        hit_rate = 0.0
        avg_return_per_trade = 0.0
    
    cumulative_return = (strategy_returns +1).prod() -1

    summary = {
        "threshold": threshold,
        "n_trades":int(n_trades),
        "hit_rate": hit_rate,
        "avg_return_per_trade": avg_return_per_trade,
        "cumulative_return": cumulative_return,
    }

    return summary

# 6. Training our models
def train_logistic_regression(x_train, y_train):
    logger.info("Training Logistic Regression model")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )
    model.fit(x_train_scaled, y_train)
    return model, scaler

def tune_random_forest(x_train, y_train):
    logger.info("Training and Tuning Random Forest Model with GridsearchCV")

    # Defining the hyperparameter grid
    param_grid = {
        "n_estimators" : [100, 200, 300, 500],
        "max_depth": [10,12,15,20],
        "min_samples_split": [2,4,8],
        "min_samples_leaf": [1, 2, 4]
    }

    # Intialize Random Forest model
    rf = RandomForestClassifier(class_weight="balanced_subsample", random_state=42)

    # Randomized Search CV since it took me ages to run GridSearchCV last time since there is so many combinations
    random_search_rf = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=10, cv=3, n_jobs=-1, verbose=2)
    # fit model to the training data
    random_search_rf.fit(x_train, y_train)
    # let's get the best model from RandomizedSearch
    best_rf = random_search_rf.best_estimator_
    logger.info(f"Best Random Forest parameters: {random_search_rf.best_params_}")
    return best_rf

def tune_xgboost(x_train, y_train):

    # Defining the hyperparameter grid for XGBoost
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }

    # Initialize XGBoost model
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist", #using the histogram method
        device="cpu" # we need to specify the use of GPU for faster training
        )

    # Initialize GridSearchCV
    grid_search_xgb = GridSearchCV(xgb, param_grid, cv=3, scoring="f1", n_jobs=-1)

    # fit model to the training data
    grid_search_xgb.fit(x_train, y_train)

    # let's get the best model from GridSearch
    best_xgb = grid_search_xgb.best_estimator_

    logger.info(f"Best XGBoost parameters: {grid_search_xgb.best_params_}")

    return best_xgb

# 7. Let's save / Load Helpers
def save_model(model, scaler, path: str):
    os.makedirs(Path(path).parent, exist_ok=True)
    logger.info(f"Saving model bundle to {path}")
    joblib.dump({"model": model, "scaler": scaler}, path)

def save_model_no_scaler(model, path: str):
    os.makedirs(Path(path).parent, exist_ok=True)
    logger.info(f"Saving model to {path}")
    joblib.dump({"model": model}, path)

# 8. Full training pipeline
def train_all_models(df: pd.DataFrame, models_dir: str = "models"):

    #Full pipeline :
    # 1. time series split
    # 2. feature selection
    # 3. train Logestic, RF, XGB
    # 4. evaluate on test set
    # 5. Scan thresholds
    # 6. run a simple backtest on the best model

    train_df, test_df = time_series_split(df, time_col="timestamp", test_size=0.2)
    x_train, y_train, x_test, y_test, feature_cols = get_features_and_target(train_df, test_df)

    # 1. Logistic Regression
    log_reg, scaler = train_logistic_regression(x_train, y_train,)
    y_proba_logreg = log_reg.predict_proba(scaler.transform(x_test))[:,1]
    logreg_default_metrics = evaluate_at_threshold(y_test, y_proba_logreg, threshold=0.5)
    logreg_threshold_scan = scan_thresholds(y_test, y_proba_logreg)

    save_model(log_reg, scaler, os.path.join(models_dir, "logistic_regression.pkl"))

    # 2. Tune and Train Random Forest
    best_rf = tune_random_forest(x_train, y_train)
    y_proba_rf = best_rf.predict_proba(x_test)[:, 1]
    rf_default_metrics = evaluate_at_threshold(y_test, y_proba_rf, threshold=0.5)
    rf_threshold_scan = scan_thresholds(y_test, y_proba_rf)

    save_model_no_scaler(best_rf, os.path.join(models_dir, "random_forest.pkl"))

    # 3) Tune and Train XGBoost
    best_xgb = tune_xgboost(x_train, y_train)
    y_proba_xgb = best_xgb.predict_proba(x_test)[:, 1]
    xgb_default_metrics = evaluate_at_threshold(y_test, y_proba_xgb, threshold=0.5)
    xgb_threshold_scan = scan_thresholds(y_test, y_proba_xgb)

    save_model_no_scaler(best_xgb, os.path.join(models_dir, "xgboost.pkl"))

    # Backtest on XGBoost with a stricter threshold (my choice will be 0.7)
    backtest_summary = simple_long_only_backtest(
        test_df,
        y_proba_xgb,
        threshold=0.7,
        return_col="future_return_1h",
    )

    logger.info("=== Training Completed ===")
    logger.info(f"Logistic (thr=0.5): {logreg_default_metrics}")
    logger.info(f"RandomForest (thr=0.5): {rf_default_metrics}")
    logger.info(f"XGBoost (thr=0.5): {xgb_default_metrics}")
    logger.info(f"XGBoost backtest (thr=0.7): {backtest_summary}")

    results = {
        "logistic_default" : logreg_default_metrics,
        "rf_default" : rf_default_metrics,
        "xgb_default" : xgb_default_metrics,
        "logistic_threshold":logreg_threshold_scan,
        "rf_threshold": rf_threshold_scan,
        "xgb_threshold": xgb_threshold_scan,
        "xgb_backtest": backtest_summary,
    }

    return results

# Function to load a saved model
def load_model(model_path: str):

    # Loads the saved model from the provided path.

    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise

# Function to make predictions using the loaded model
def predict(model, X: pd.DataFrame):
    # we will be making prediction using the trained model and input data X
    try:
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        raise