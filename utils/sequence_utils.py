import numpy as np
import pandas as pd

def create_input_sequence(recent_rows, seq_len=15, n_features=10):
    """
    Convert last seq_len rows into a single LSTM input sequence.
    recent_rows: pandas DataFrame of last 15 timesteps with 10 features each
    """
    if len(recent_rows) != seq_len:
        raise ValueError(f"Need exactly {seq_len} rows. Got {len(recent_rows)}")
        
    # Convert to numpy (2D → 3D)
    X = recent_rows.values.reshape(1, seq_len, n_features)
    return X
