import pandas as pd
import numpy as np

def downsample_rolling_window(df, window_size=200, step_size=100):
    """
    Downsample data using rolling window approach
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing time series data
    window_size : int, default=200
        Size of the rolling window
    step_size : int, default=100
        Step size for moving the window
        
    Returns:
    --------
    pandas.DataFrame
        Downsampled data using rolling window statistics
    """
    windows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start+window_size].copy()
        window['window_start'] = df['IMU_TimeSeries'].iloc[start]
        windows.append(window)
    return pd.concat(windows, ignore_index=True)
