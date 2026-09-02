import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from .sequence_utils import create_input_sequence

# Load model + scalers
model = load_model("models/final_lstm_model.h5")
feature_scaler = joblib.load("models/feature_scaler.pkl")
target_scaler = joblib.load("models/target_scaler.pkl")

SEQ_LEN = 15

def predict_next_price(last_15_rows):
    """
    last_15_rows → DataFrame with shape (15, 10 feature columns)
    """

    # Scale features
    scaled_features = feature_scaler.transform(last_15_rows)

    # Create proper LSTM input shape
    X = scaled_features.reshape(1, SEQ_LEN, last_15_rows.shape[1])

    # Predict (scaled)
    scaled_pred = model.predict(X)[0][0]

    # Reverse scale to original price
    final_pred = target_scaler.inverse_transform([[scaled_pred]])[0][0]

    return final_pred
