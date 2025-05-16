import pandas as pd
import numpy as np
import pdb

# Data Labels:

# Label for EMG Data shared:

#     Open CSV files to check what they look like. Use skiprows=5 and low_memory=False to load it properly (the top 5 rows are metadata)
#     4 EMG Sensors are used: Sensor 1 (right bicep), Sensor 2 (right delt), Sensor 3 (left bicep), Sensor 4 (left delt). Rename the sensor columns appropriately
#     Each EMG and IMU column has another time series column so feel free to drop the duplicate Time columns and this will help you reduce the overall column number of your dataset.
#     Also, the sampling frequency for EMG is 1259 Hz and IMU 148 Hz - you would have to handle the mismatch in sampling frequency if you plan to use both type of data for your analysis.

#top 5 columns are meta data, but first 2 lines of data after header are also garbage for our purposes, for the
#EMG/IMU readings they state each sensor's frequency and cycle time (which we don't care about). Therefore
#this is tricky bc it means that we can't immediately grab the columns since we have to remove the 2 lines after them and 
#get the rest of the data following them. 

def read_run(filename, skiprows=7, header_row=5):
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == header_row:
                header = [x.strip() for x in line.split(',')]
                break
    usecols = list(range(0, 56)) 
    df = pd.read_csv(filename, low_memory = False, 
                     header = 0,  
                     skiprows=skiprows,
                    #  names=header,
                     usecols = usecols,
                     on_bad_lines='skip') 
    df.columns = ['RDelt_TimeSeries', 'RDelt_MilliVolts', 'RDelt_Acc X Time Series(s)', 'RDelt_ACC X (G)', 'RDelt_Acc Y Time Series(s)',   'RDelt_ACC Y (G)',  'RDelt_Acc Z Time Series(s)', 'RDelt_ACC Z (G)','RDelt_GyroXTime Series(s)', 'RDelt_GYRO X (deg/s)','RDelt_GyroYTime Series(s)', 'RDelt_GYRO Y (deg/s)', 'RDelt_GyroZTime Series(s)', 'RDelt_GYRO Z (deg/s)',
                 'LDelt_TimeSeries', 'LDelt_MilliVolts', 'LDelt_Acc X Time Series(s)', 'LDelt_ACC X (G)', 'LDelt_Acc Y Time Series(s)',   'LDelt_ACC Y (G)',  'LDelt_Acc Z Time Series(s)', 'LDelt_ACC Z (G)','LDelt_GyroXTime Series(s)', 'LDelt_GYRO X (deg/s)','LDelt_GyroYTime Series(s)', 'LDelt_GYRO Y (deg/s)', 'LDelt_GyroZTime Series(s)', 'LDelt_GYRO Z (deg/s)',
                 'RBicep_TimeSeries', 'RBicep_MilliVolts', 'RBicep_Acc X Time Series(s)', 'RBicep_ACC X (G)', 'RBicep_Acc Y Time Series(s)',   'RBicep_ACC Y (G)',  'RBicep_Acc Z Time Series(s)', 'RBicep_ACC Z (G)','RBicep_GyroXTime Series(s)', 'RBicep_GYRO X (deg/s)','RBicep_GyroYTime Series(s)', 'RBicep_GYRO Y (deg/s)', 'RBicep_GyroZTime Series(s)', 'RBicep_GYRO Z (deg/s)',
                 'LBicep_TimeSeries', 'LBicep_MilliVolts', 'LBicep_Acc X Time Series(s)', 'LBicep_ACC X (G)', 'LBicep_Acc Y Time Series(s)',   'LBicep_ACC Y (G)',  'LBicep_Acc Z Time Series(s)', 'LBicep_ACC Z (G)','LBicep_GyroXTime Series(s)', 'LBicep_GYRO X (deg/s)','LBicep_GyroYTime Series(s)', 'LBicep_GYRO Y (deg/s)', 'LBicep_GyroZTime Series(s)', 'LBicep_GYRO Z (deg/s)'
                ]
    pdb.set_trace()
    return df

df_p3_exo = read_run("P3_Exo_1_0.csv") #first run, male
df_p3_noexo = read_run("P3_NoExo_1_0.csv") #second run, male
df_p4_exo = read_run("P4_Exo_1_0.csv") #1st run female
df_p4_noexo = read_run("P4_NoExo_1_0.csv") #2nd run female
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
#CP: does this make sure to remove the redundant time series columns?
#can keep  ACC X Time Series (s) in each sensor group, and remove any other column with 'Time Series (s)' in its name 
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
