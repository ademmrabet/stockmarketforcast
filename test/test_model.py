import pytest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from src.model import load_model, predict


# Testing the model loading function
def test_load_model():
    model_path = "C:/Users/ademm/OneDrive/Bureau/stockmarketforcast/models/random_forest.pkl"
    model = load_model(model_path)
    assert model is not None, f"Failed to load model from {model_path}"


# Test model Prediction
def test_model_prediction():
    # Example input data for prediction
    test_data = pd.DataFrame({
        'last': [150.41],
        'high': [150.79],
        'low': [147.06],
        'chg_': [4.31],
        'chg_%': [0.0295],
        'vol_': [409000000.0],
        'hour': [15],
        'weekday': [0],
        'month': [3],
        'is_weekend': [0],
        'high_low_range': [3.73],
        'last_lag_1': [150.41],
        'return_1h': [0.0],
        'ma_last_3h': [150.410000],
        'volatility_3h': [0.000000],
        'ma_last_6h': [150.410000],
        'volatility_6h': [0.000000],
        'vol_lag_1': [409000000.0],
        'vol_change_1h': [0.0],
        'vol_ma_3h': [409000000.0],
        'vol_ma_6h': [409000000.0]
    })
    
    model_path = "../models/random_forest.pkl"
    model = load_model(model_path)
    predictions = predict(model, test_data)

    # Assert that predictions are 1D array with same length as input
    assert predictions.shape[0] == test_data.shape[0], f"Prediction size mismatch: {predictions.shape[0]} != {test_data.shape[0]}"

# Test model accuracy on a real test dataset
def test_model_accuracy():
    # Load the feature dataset
    df = pd.read_csv("C:/Users/ademm/OneDrive/Bureau/stockmarketforcast/data/02_processed/stocks_features.csv")

    # Split test data
    x_test = df.drop(columns=['movement_1h'])
    y_test = df['movement_1h']

    # Load the model
    model = load_model("../models/random_forest.pkl")

    # Make predictions
    predictions = model.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.7, f"Model accuracy is too low: {accuracy}"
