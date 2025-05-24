import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from modif_cols import tidy_emg_imu_as_measured
from resampling import downsample_rolling_window
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

#also, data points end at the bottom when sensors are stopped, but time/mV measurements will keep going, so must trim those too for each run
#went ahead and fixed up time series rows, and added gender/run# column
#still need to break up redundant columns into a sensor # column, could maybe use df.melt
def read_run(file_path):
    """
    Read and preprocess a single run data file
    """
    df = pd.read_csv(file_path)
    return df

def column_clean(df):
    """
    Clean and standardize column names and data types
    """
    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Standardize missing values
    df = df.replace(['', ' ', 'NA', None], np.nan)
    
    return df

def create_sensor_col(df, run_num, gender, exo):
    """
    Add sensor information columns
    """
    df['run_num'] = run_num
    df['gender'] = gender
    df['exo'] = exo
    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    """
    # Forward fill for time series data
    time_cols = [col for col in df.columns if 'TimeSeries' in col]
    for col in time_cols:
        df[col] = df[col].fillna(method='ffill')
    
    # Interpolate for sensor data
    sensor_cols = [col for col in df.columns if 'EMG' in col or 'ACC' in col or 'GYRO' in col]
    for col in sensor_cols:
        df[col] = df[col].interpolate(method='linear')
    
    return df

def bandpass_filter_emg(signal, fs=1259, lowcut=20, highcut=450, order=4):
    """
    Apply bandpass filter to EMG signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def lowpass_filter_imu(signal, fs=148, cutoff=20, order=4):
    """
    Apply lowpass filter to IMU signal
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

def standardize_time_series(df):
    """
    Standardize time series data using rolling window
    """
    return downsample_rolling_window(df)

def preprocessing(full_df):
    """
    Main preprocessing function
    """
    # Handle missing values
    full_df = handle_missing_values(full_df)
    
    # Apply filters
    emg_cols = [col for col in full_df.columns if 'EMG' in col]
    for col in emg_cols:
        full_df[col] = bandpass_filter_emg(full_df[col])
    
    imu_cols = [col for col in full_df.columns if 'ACC' in col or 'GYRO' in col]
    for col in imu_cols:
        full_df[col] = lowpass_filter_imu(full_df[col])
    
    return full_df
