# Calculations for Feature Extraction from Project_Guide
def compute_emg_features(signal):
    return {
        'mean': np.mean(signal),
        'max': np.max(signal),
        'min': np.min(signal),
        'std': np.std(signal),
        'rms': np.sqrt(np.mean(signal**2))
    }

def compute_accel_features(a_x, a_y, a_z):
    a_mag = np.sqrt(a_x**2 + a_y**2 + a_z**2)
    
    features = {
        'peak_accel': np.max(a_mag),
        'mean_accel': np.mean(a_mag),
        'total_accel': np.sqrt(np.mean(a_x**2) + np.mean(a_y**2) + np.mean(a_z**2)),
        'accel_range': np.max(a_mag) - np.min(a_mag)
    }
    return features

def compute_gyro_features(w_x, w_y, w_z):
    w_mag = np.sqrt(w_x**2 + w_y**2 + w_z**2)
    
    features = {
        'peak_angular_vel': np.max(w_mag),
        'mean_angular_vel': np.mean(w_mag),
        'total_angular_vel': np.sqrt(np.mean(w_x**2) + np.mean(w_y**2) + np.mean(w_z**2)),
        'angular_vel_range': np.max(w_mag) - np.min(w_mag)
    }
    return features 

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy

def calculate_rms(signal):
    """Calculate Root Mean Square"""
    return np.sqrt(np.mean(np.square(signal)))

def calculate_mav(signal):
    """Calculate Mean Absolute Value"""
    return np.mean(np.abs(signal))

def calculate_wl(signal):
    """Calculate Waveform Length"""
    return np.sum(np.abs(np.diff(signal)))

def calculate_ssc(signal, threshold=0):
    """Calculate Slope Sign Change"""
    diff = np.diff(signal)
    return np.sum(np.logical_and(diff[:-1] * diff[1:] < 0, 
                                np.abs(diff[:-1] - diff[1:]) > threshold))

def calculate_entropy(signal, bins=10):
    """Calculate Shannon Entropy"""
    hist, _ = np.histogram(signal, bins=bins, density=True)
    return entropy(hist)

def extract_emg_features(emg_signal, fs=1259):
    """Extract features from EMG signal"""
    # Remove any NaN values
    emg_signal = np.array(emg_signal)
    emg_signal = emg_signal[~np.isnan(emg_signal)]
    
    if len(emg_signal) == 0:
        return {
            'rms': np.nan,
            'mav': np.nan,
            'wl': np.nan,
            'ssc': np.nan,
            'entropy': np.nan,
            'mean_freq': np.nan,
            'median_freq': np.nan
        }
    
    # Time domain features
    features = {
        'rms': calculate_rms(emg_signal),
        'mav': calculate_mav(emg_signal),
        'wl': calculate_wl(emg_signal),
        'ssc': calculate_ssc(emg_signal),
        'entropy': calculate_entropy(emg_signal)
    }
    
    # Frequency domain features
    try:
        f, psd = signal.welch(emg_signal, fs=fs, nperseg=min(256, len(emg_signal)))
        features['mean_freq'] = np.sum(f * psd) / np.sum(psd)
        features['median_freq'] = f[np.where(np.cumsum(psd) >= np.sum(psd)/2)[0][0]]
    except:
        features['mean_freq'] = np.nan
        features['median_freq'] = np.nan
    
    return features

def extract_imu_features(accel_data, gyro_data):
    """Extract features from IMU data"""
    features = {}
    
    # Accelerometer features
    for axis in ['X', 'Y', 'Z']:
        acc = accel_data[f'ACC {axis}'].values
        acc = acc[~np.isnan(acc)]
        
        if len(acc) == 0:
            features.update({
                f'acc_{axis}_mean': np.nan,
                f'acc_{axis}_std': np.nan,
                f'acc_{axis}_max': np.nan,
                f'acc_{axis}_min': np.nan,
                f'acc_{axis}_rms': np.nan
            })
        else:
            features.update({
                f'acc_{axis}_mean': np.mean(acc),
                f'acc_{axis}_std': np.std(acc),
                f'acc_{axis}_max': np.max(acc),
                f'acc_{axis}_min': np.min(acc),
                f'acc_{axis}_rms': calculate_rms(acc)
            })
    
    # Gyroscope features
    for axis in ['X', 'Y', 'Z']:
        gyro = gyro_data[f'GYRO {axis}'].values
        gyro = gyro[~np.isnan(gyro)]
        
        if len(gyro) == 0:
            features.update({
                f'gyro_{axis}_mean': np.nan,
                f'gyro_{axis}_std': np.nan,
                f'gyro_{axis}_max': np.nan,
                f'gyro_{axis}_min': np.nan,
                f'gyro_{axis}_rms': np.nan
            })
        else:
            features.update({
                f'gyro_{axis}_mean': np.mean(gyro),
                f'gyro_{axis}_std': np.std(gyro),
                f'gyro_{axis}_max': np.max(gyro),
                f'gyro_{axis}_min': np.min(gyro),
                f'gyro_{axis}_rms': calculate_rms(gyro)
            })
    
    return features

def extract_all_features(df):
    """Extract all features from the dataset"""
    # Group by muscle
    features_list = []
    
    for muscle in df['Muscle'].unique():
        muscle_data = df[df['Muscle'] == muscle].copy()
        
        # Extract EMG features
        emg_features = extract_emg_features(muscle_data['EMG_MV'].values)
        
        # Extract IMU features
        accel_data = muscle_data[['ACC X', 'ACC Y', 'ACC Z']]
        gyro_data = muscle_data[['GYRO X', 'GYRO Y', 'GYRO Z']]
        imu_features = extract_imu_features(accel_data, gyro_data)
        
        # Combine features
        muscle_features = {
            'Muscle': muscle,
            **emg_features,
            **imu_features
        }
        features_list.append(muscle_features)
    
    # Create features dataframe
    features_df = pd.DataFrame(features_list)

    # ---- Feature Engineering ----
    # emg_max_x_gyro_peak
    if 'emg_max' in features_df.columns and 'gyro_peak' in features_df.columns:
        features_df['emg_max_x_gyro_peak'] = features_df['emg_max'] * features_df['gyro_peak']
    # gyro_range_div_acc_range
    if 'gyro_range' in features_df.columns and 'acc_range' in features_df.columns:
        features_df['gyro_range_div_acc_range'] = features_df['gyro_range'] / (features_df['acc_range'] + 1e-6)
    # emg_min_plus_acc_peak
    if 'emg_min' in features_df.columns and 'acc_peak' in features_df.columns:
        features_df['emg_min_plus_acc_peak'] = features_df['emg_min'] + features_df['acc_peak']
    # emg_range
    if 'emg_max' in features_df.columns and 'emg_min' in features_df.columns:
        features_df['emg_range'] = features_df['emg_max'] - features_df['emg_min']
    # emg_variance
    if 'emg_std' in features_df.columns:
        features_df['emg_variance'] = features_df['emg_std'] ** 2
    else:
        features_df['emg_variance'] = 0.01 ** 2
    # acc_std
    if 'acc_range' in features_df.columns:
        features_df['acc_std'] = features_df['acc_range'] / 2
    # gyro_std
    if 'gyro_range' in features_df.columns:
        features_df['gyro_std'] = features_df['gyro_range'] / 2
    # acc_energy
    if 'acc_peak' in features_df.columns:
        features_df['acc_energy'] = features_df['acc_peak'] ** 2
    # gyro_energy
    if 'gyro_peak' in features_df.columns:
        features_df['gyro_energy'] = features_df['gyro_peak'] ** 2

    return features_df

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