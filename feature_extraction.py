import numpy as np
import pandas as pd
from scipy.fft import fft
from numpy import mean, median

def extract_features_sliding_window(df, emg_col, acc_cols, gyro_cols, label, gender,
                                    window_ms=200, step_ms=100, freq_hz=148, emg_fs=1259):
    window_size = int((window_ms / 1000) * freq_hz)
    step_size = int((step_ms / 1000) * freq_hz)

    features = []

    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        window = df.iloc[start:end]

        emg = window[emg_col].values
        acc = window[acc_cols]
        gyro = window[gyro_cols]

        emg_fft = np.abs(fft(emg))
        freqs = np.fft.fftfreq(len(emg_fft), d=1/emg_fs)
        mask = (freqs > 20) & (freqs < 450)
        valid_fft = emg_fft[mask]
        valid_freqs = freqs[mask]

        fft_mean = mean(valid_freqs * valid_fft)
        fft_median = median(valid_freqs * valid_fft)
        fft_power = np.sum(valid_fft**2)

        feature_row = {
            'emg_max': emg.max(),
            'emg_min': emg.min(),
            'emg_rms': np.sqrt(np.mean(emg**2)),
            'acc_peak': np.linalg.norm(acc, axis=1).max(),
            'acc_range': np.ptp(np.linalg.norm(acc, axis=1)),
            'gyro_peak': np.linalg.norm(gyro, axis=1).max(),
            'gyro_range': np.ptp(np.linalg.norm(gyro, axis=1)),
            'emg_fft_mean_freq': fft_mean,
            'emg_fft_median_freq': fft_median,
            'emg_fft_power': fft_power,
            'label': label,
            'gender': gender
        }

        features.append(feature_row)

    return pd.DataFrame(features)
