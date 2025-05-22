import pandas as pd

def upsample(df):
    IMU_cols = [
       'RDelt_ACC X (G)', 'RDelt_ACC Y (G)', 'RDelt_ACC Z (G)', 'RDelt_GYRO X (deg/s)',
       'RDelt_GYRO Y (deg/s)', 'RDelt_GYRO Z (deg/s)',  'LDelt_ACC X (G)', 'LDelt_ACC Y (G)',
       'LDelt_ACC Z (G)', 'LDelt_GYRO X (deg/s)', 'LDelt_GYRO Y (deg/s)',
       'LDelt_GYRO Z (deg/s)',  'RBicep_ACC X (G)',
       'RBicep_ACC Y (G)', 'RBicep_ACC Z (G)', 'RBicep_GYRO X (deg/s)',
       'RBicep_GYRO Y (deg/s)', 'RBicep_GYRO Z (deg/s)',
       'LBicep_ACC X (G)', 'LBicep_ACC Y (G)',
       'LBicep_ACC Z (G)', 'LBicep_GYRO X (deg/s)', 'LBicep_GYRO Y (deg/s)',
       'LBicep_GYRO Z (deg/s)'
       ]
  df['IMU_TimeSeries'] = pd.to_numeric(df['IMU_TimeSeries'])
  df['time'] = pd.to_timedelta(df['IMU_TimeSeries'], unit='s')
  df = df.dropna(subset=['IMU_TimeSeries']) # Drop rows where IMU timestamps are NaN
  df = df.set_index('time')
  freq_nanseconds = int(0.0007941176470588235 * 1e9)  # Convert to integer microseconds
  IMU_upsampled = df[IMU_cols].resample(f'{freq_nanseconds}ns').asfreq()
  IMU_upsampled = IMU_upsampled.fillna(method='ffill') #forward fill the values 
  IMU_upsampled = IMU_upsampled.apply(lambda x: pd.to_numeric(x))
  IMU_upsampled = IMU_upsampled.interpolate(method='linear')  # Interpolates the data using the linear method to match EMG data
  #sine interpolation is best 
  df_new = IMU_upsampled.join(df['RDelt_EMG_MilliVolts'])
  return df_new

def downsample(df):
  EMG_cols = ['RDelt_EMG_MilliVolts', 'LDelt_EMG_MilliVolts', 'RBicep_EMG_MilliVolts', 'LBicep_EMG_MilliVolts']
  
  df['EMG_TimeSeries'] = pd.to_numeric(df['EMG_TimeSeries'])
  df['time'] = pd.to_timedelta(df['EMG_TimeSeries'], unit='s')
  df = df.set_index('time')
  
  df[EMG_cols] = df[EMG_cols].resample('6.75ms').asfreq()  # Scales these columns to be the same length as IMU data
  return df

## If downsample using pandas.resample doesn't work, use this alternative function that uses the rows index
def alternative(df):
  high_rate = 1259
  low_rate = 148
  step = high_rate / low_rate
  
  EMG_cols = ['RDelt_EMG_MilliVolts', 'LDelt_EMG_MilliVolts', 'RBicep_EMG_MilliVolts', 'LBicep_EMG_MilliVolts']
  
  # Indexes to sample
  indices = np.round(np.arange(0, len(df), step)).astype(int)
  indices = indices[indices < len(df)]  # Ensure we stay within bounds

  # Downsample using nearest index
  df[EMG_cols] = df[EMG_cols].iloc[indices].reset_index(drop=True)
  df = df.drop(columns=['EMG_TimeSeries'])
  return df
