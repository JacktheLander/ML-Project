import pandas as pd
import numpy as np
skiprow_num = 5
df_p3_exo = pd.read_csv("P3_Exo_1_0.csv",skiprows=skiprow_num) #first run, male
df_p3_noexo = pd.read_csv("P3_NoExo_1_0.csv", skiprows=skiprow_num) #second run, male
df_p4_exo = pd.read_csv("P4_Exo_1_0.csv", skiprows=skiprow_num) #1st run female
df_p4_noexo = pd.read_csv("P4_NoExo_1_0.csv", skiprows=skiprow_num) #2nd run female

# # Show the head of the data
# df_p3_exo.describe()
df_p3_noexo.head()
# df_p4_exo.head()
# df_p4_noexo.head()

# # Choose inputs
# features = df_p3_exo[['EMG 1 (mV)', 'ACC X (G)', 'ACC Y (G)', 'ACC Z (G)', 'GYRO X (deg/s)', 'GYRO Y (deg/s)', 'GYRO Z (deg/s)']].dropna()
# features.head()


# Functions for Calculating Features from data

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

# Run functions for each feature type for p3_exo data
emg_features = compute_emg_features(df_p3_exo['EMG 1 (mV)'])
accel_features = compute_accel_features(df_p3_exo['ACC X (G)', 'ACC Y (G)', 'ACC Z (G)'])
gyro_features = compute_gyro_features(df_p3_exo['GYRO X (deg/s)', 'GYRO Y (deg/s)', 'GYRO Z (deg/s)'])
