from scipy.signal import filtfilt, butter
import numpy as np
import pandas as pd

def bandpass_filter_emg(series_signal, fs=1259, lowcut=20, highcut=450, order=4):
    arr = series_signal.values if isinstance(series_signal, pd.Series) else np.array(series_signal)    
    if np.isnan(arr).all() or len(arr) == 0: #edge case check
        return arr
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, series_signal)

# IMU Low-pass Filter (<20Hz)
def lowpass_filter_imu(series_signal, fs=148, cutoff=20, order=4):
    arr = series_signal.values if isinstance(series_signal, pd.Series) else np.array(series_signal)
    if np.isnan(arr).all() or len(arr) == 0: #edge case check
        return arr
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, series_signal)