import pandas as pd

def upsample(df):
  IMU_cols = ['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']

  df['time'] = pd.to_timedelta(df['ACC X Time Series (s)'], unit='s')
  df = df.set_index('time')
  IMU_upsampled = df[IMU_cols].resample('794117ns').asfreq()  # Scales these columns to be the same length as EMG data
  IMU_upsampled = IMU_upsampled.interpolate(method='linear')  # Interpolates the data using the linear method to match EMG data
  df_new = IMU_upsampled.join(df['EMG 1 (mV)'])
  return df_new
