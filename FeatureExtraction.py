import pandas as pd
import numpy as np
skiprow_num = 5
df_p3_exo = pd.read_csv("P3_Exo_1_0.csv",skiprows=skiprow_num) #first run, male
df_p3_noexo = pd.read_csv("P3_NoExo_1_0.csv", skiprows=skiprow_num) #second run, male
df_p4_exo = pd.read_csv("P4_Exo_1_0.csv", skiprows=skiprow_num) #1st run female
df_p4_noexo = pd.read_csv("P4_NoExo_1_0.csv", skiprows=skiprow_num) #2nd run female
dfs = [df_p3_exo, df_p3_noexo, df_p4_exo, df_p4_noexo]

# # Show the head of the data
# df_p3_exo.describe()
df_p3_noexo.head()
# df_p4_exo.head()
# df_p4_noexo.head()

# # Choose inputs
# features = df_p3_exo[['EMG 1 (mV)', 'ACC X (G)', 'ACC Y (G)', 'ACC Z (G)', 'GYRO X (deg/s)', 'GYRO Y (deg/s)', 'GYRO Z (deg/s)']].dropna()
# features.head()

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

feature_sets = []

# Run functions to extract features for each dataframe
for df in dfs:
    emg_features = compute_emg_features(df['EMG 1 (mV)'])
    accel_features = compute_accel_features(df['ACC X (G)'], df['ACC Y (G)'], df['ACC Z (G)'])
    gyro_features = compute_gyro_features(df['GYRO X (deg/s)'], df['GYRO Y (deg/s)'], df['GYRO Z (deg/s)'])
    features = {
        'emg': emg_features,
        'accel': accel_features,
        'gyro': gyro_features
    }
    feature_sets.append(features)

# feature_sets now contains extracted features for each df
p3exo_feats, p3noexo_feats, p4exo_feats, p4noexo_feats = feature_sets
