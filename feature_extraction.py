import pdb
import pandas as pd
import numpy as np
# Calculations for Feature Extraction from Project_Guide

def extract_features(df):
    # Group by relevant columns
    group_cols = ['BodyPart', 'run_num', 'gender', 'exo']  # adapt as needed
    feature_rows = []
    for group_vals, group in df.groupby(group_cols):
        # Accelerometer features
        a_x, a_y, a_z = group['ACC X (G)_filtered'], group['ACC Y (G)_filtered'], group['ACC Z (G)_filtered']
        a_mag = np.sqrt(a_x**2 + a_y**2 + a_z**2)
        accel_peak = np.max(a_mag)
        accel_mean = np.mean(a_mag)
        accel_total = np.sqrt(np.mean(a_x**2) + np.mean(a_y**2) + np.mean(a_z**2))
        accel_range = np.max(a_mag) - np.min(a_mag)

        # Gyroscope features
        w_x, w_y, w_z = group['GYRO X (deg/s)_filtered'], group['GYRO Y (deg/s)_filtered'], group['GYRO Z (deg/s)_filtered']
        w_mag = np.sqrt(w_x**2 + w_y**2 + w_z**2)
        gyro_peak = np.max(w_mag)
        gyro_mean = np.mean(w_mag)
        gyro_total = np.sqrt(np.mean(w_x**2) + np.mean(w_y**2) + np.mean(w_z**2))
        gyro_range = np.max(w_mag) - np.min(w_mag)

        # EMG features (filtered)
        emg = group['EMG_MilliVolts_filtered']
        emg_mean = np.mean(emg)
        emg_max = np.max(emg)
        emg_min = np.min(emg)
        emg_std = np.std(emg)
        emg_rms = np.sqrt(np.mean(emg**2))

        # Build feature dict
        feature_dict = {
            'BodyPart': group_vals[0],
            'run_num': group_vals[1],
            'gender': group_vals[2],
            'exo': group_vals[3],
            'accel_peak': accel_peak,
            'accel_mean': accel_mean,
            'accel_total': accel_total,
            'accel_range': accel_range,
            'gyro_peak': gyro_peak,
            'gyro_mean': gyro_mean,
            'gyro_total': gyro_total,
            'gyro_range': gyro_range,
            'emg_mean': emg_mean,
            'emg_max': emg_max,
            'emg_min': emg_min,
            'emg_std': emg_std,
            'emg_rms': emg_rms,
        }
        feature_rows.append(feature_dict)
    #THIS IS LAME (only 17 rows) BRUH
    # Return as a new DataFrame
    return pd.DataFrame(feature_rows)

#old funcs 
def compute_emg_features(df):
    signal = df['EMG_MilliVolts']
    return {
        'mean': np.mean(signal),
        'max': np.max(signal),
        'min': np.min(signal),
        'std': np.std(signal),
        'rms': np.sqrt(np.mean(signal**2))
    }

def compute_accel_features(df):
    a_x = df['ACC X (G)'], a_y = df['ACC Y (G)'], a_z = df['ACC Z (G)']
    a_mag = np.sqrt(a_x**2 + a_y**2 + a_z**2)
    
    features = {
        'peak_accel': np.max(a_mag),
        'mean_accel': np.mean(a_mag),
        'total_accel': np.sqrt(np.mean(a_x**2) + np.mean(a_y**2) + np.mean(a_z**2)),
        'accel_range': np.max(a_mag) - np.min(a_mag)
    }
    return features

def compute_gyro_features(df):
    w_x = df['GYRO X (deg/s)'], w_y = df['GYRO Y (deg/s)'], w_z = df['GYRO Z (deg/s)']
    w_mag = np.sqrt(w_x**2 + w_y**2 + w_z**2)
    
    features = {
        'peak_angular_vel': np.max(w_mag),
        'mean_angular_vel': np.mean(w_mag),
        'total_angular_vel': np.sqrt(np.mean(w_x**2) + np.mean(w_y**2) + np.mean(w_z**2)),
        'angular_vel_range': np.max(w_mag) - np.min(w_mag)
    }
    return features 
    # fft_mean = mean(valid_freqs * valid_fft)
    # fft_median = median(valid_freqs * valid_fft)
    # fft_power = np.sum(valid_fft**2)

    # feature_row = {
    #     'emg_max': emg.max(),
    #     'emg_min': emg.min(),
    #     'emg_rms': np.sqrt(np.mean(emg**2)),
    #     'acc_peak': np.linalg.norm(acc, axis=1).max(),
    #     'acc_range': np.ptp(np.linalg.norm(acc, axis=1)),
    #     'gyro_peak': np.linalg.norm(gyro, axis=1).max(),
    #     'gyro_range': np.ptp(np.linalg.norm(gyro, axis=1)),
    #     'emg_fft_mean_freq': fft_mean,
    #     'emg_fft_median_freq': fft_median,
    #     'emg_fft_power': fft_power,
    #     'label': label,
    #     'gender': gender
    # }