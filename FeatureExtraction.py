import pandas as pd
import numpy as np
from pprint import pprint

def read_clean_csv(filename):
    df = pd.read_csv(
        filename,
        header=5,               # Row 6 = actual column headers
        skiprows=[6, 7],        # Rows 7 & 8 = garbage
        usecols=range(56),      # Only use columns A to BD
        on_bad_lines='skip'
    )
    df.columns = df.columns.str.strip()           # Remove leading/trailing spaces
    df = df.apply(pd.to_numeric, errors='coerce') # Convert everything to numeric
    return df

def compute_emg_features(signal):
    return {
        'mean': np.nanmean(signal),
        'max': np.nanmax(signal),
        'min': np.nanmin(signal),
        'std': np.nanstd(signal),
        'rms': np.sqrt(np.nanmean(signal**2))
    }

def compute_accel_features(a_x, a_y, a_z):
    a_mag = np.sqrt(a_x**2 + a_y**2 + a_z**2)
    return {
        'peak_accel': np.nanmax(a_mag),
        'mean_accel': np.nanmean(a_mag),
        'total_accel': np.sqrt(np.nanmean(a_x**2) + np.nanmean(a_y**2) + np.nanmean(a_z**2)),
        'accel_range': np.nanmax(a_mag) - np.nanmin(a_mag)
    }

def compute_gyro_features(w_x, w_y, w_z):
    w_mag = np.sqrt(w_x**2 + w_y**2 + w_z**2)
    return {
        'peak_angular_vel': np.nanmax(w_mag),
        'mean_angular_vel': np.nanmean(w_mag),
        'total_angular_vel': np.sqrt(np.nanmean(w_x**2) + np.nanmean(w_y**2) + np.nanmean(w_z**2)),
        'angular_vel_range': np.nanmax(w_mag) - np.nanmin(w_mag)
    }

# === Run It ===
df = read_clean_csv("P3_Exo_1_0.csv")

# Select first sensor (no .1, .2, .3 suffix)
emg = df['EMG 1 (mV)']
acc_x = df['ACC X (G)']
acc_y = df['ACC Y (G)']
acc_z = df['ACC Z (G)']
gyro_x = df['GYRO X (deg/s)']
gyro_y = df['GYRO Y (deg/s)']
gyro_z = df['GYRO Z (deg/s)']

# Extract features
features = {
    'emg': compute_emg_features(emg),
    'accel': compute_accel_features(acc_x, acc_y, acc_z),
    'gyro': compute_gyro_features(gyro_x, gyro_y, gyro_z)
}

# Show output
pprint(features)
