
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

#also, data points end at the bottom when sensors are stopped, but time/mV measurements will keep going, so must trim those too for each run
#went ahead and fixed up time series rows, and added gender/run# column
#still need to break up redundant columns into a sensor # column, could maybe use df.melt
def read_run(filename, skiprows=7): #skip the first 7 rows (freq/cycle time fields as well as metadata)
    usecols = list(range(0, 56)) 
    df = pd.read_csv(filename, low_memory = False, 
                     header = 0,  
                     skiprows=skiprows,
                    #  names=header,
                     usecols = usecols,
                     on_bad_lines='skip') 
    df.columns = ['RDelt_EMG_TimeSeries', 'RDelt_EMG_MilliVolts', 'RDelt_IMU_Acc X Time Series(s)', 'RDelt_ACC X (G)', 'RDelt_Acc Y Time Series(s)',   'RDelt_ACC Y (G)',  'RDelt_Acc Z Time Series(s)', 'RDelt_ACC Z (G)','RDelt_GyroXTime Series(s)', 'RDelt_GYRO X (deg/s)','RDelt_GyroYTime Series(s)', 'RDelt_GYRO Y (deg/s)', 'RDelt_GyroZTime Series(s)', 'RDelt_GYRO Z (deg/s)',
                 'LDelt_TimeSeries', 'LDelt_MilliVolts', 'LDelt_Acc X Time Series(s)', 'LDelt_ACC X (G)', 'LDelt_Acc Y Time Series(s)',   'LDelt_ACC Y (G)',  'LDelt_Acc Z Time Series(s)', 'LDelt_ACC Z (G)','LDelt_GyroXTime Series(s)', 'LDelt_GYRO X (deg/s)','LDelt_GyroYTime Series(s)', 'LDelt_GYRO Y (deg/s)', 'LDelt_GyroZTime Series(s)', 'LDelt_GYRO Z (deg/s)',
                 'RBicep_TimeSeries', 'RBicep_MilliVolts', 'RBicep_Acc X Time Series(s)', 'RBicep_ACC X (G)', 'RBicep_Acc Y Time Series(s)',   'RBicep_ACC Y (G)',  'RBicep_Acc Z Time Series(s)', 'RBicep_ACC Z (G)','RBicep_GyroXTime Series(s)', 'RBicep_GYRO X (deg/s)','RBicep_GyroYTime Series(s)', 'RBicep_GYRO Y (deg/s)', 'RBicep_GyroZTime Series(s)', 'RBicep_GYRO Z (deg/s)',
                 'LBicep_TimeSeries', 'LBicep_MilliVolts', 'LBicep_Acc X Time Series(s)', 'LBicep_ACC X (G)', 'LBicep_Acc Y Time Series(s)',   'LBicep_ACC Y (G)',  'LBicep_Acc Z Time Series(s)', 'LBicep_ACC Z (G)','LBicep_GyroXTime Series(s)', 'LBicep_GYRO X (deg/s)','LBicep_GyroYTime Series(s)', 'LBicep_GYRO Y (deg/s)', 'LBicep_GyroZTime Series(s)', 'LBicep_GYRO Z (deg/s)'
                ]
    return df #raw data 

def column_clean(df, run_num, gender):
    #remove all time series columns except RDelt_EMG_TimeSeries' and 'RDelt_IMU_Acc X Time Series(s)', so keep time scale for both EMG and IMU 
    extr_time_series = [ 'RDelt_Acc Y Time Series(s)', 'RDelt_Acc Z Time Series(s)', 'RDelt_GyroXTime Series(s)',
                         'RDelt_GyroYTime Series(s)', 'RDelt_GyroZTime Series(s)', 'LDelt_TimeSeries', 'LDelt_Acc X Time Series(s)', 
                         'LDelt_Acc Y Time Series(s)', 'LDelt_Acc Z Time Series(s)', 'LDelt_GyroXTime Series(s)',
                         'LDelt_GyroYTime Series(s)', 'LDelt_GyroZTime Series(s)', 'RBicep_TimeSeries', 'RBicep_Acc X Time Series(s)',
                         'RBicep_Acc Y Time Series(s)', 'RBicep_Acc Z Time Series(s)', 'RBicep_GyroXTime Series(s)',
                         'RBicep_GyroYTime Series(s)', 'RBicep_GyroZTime Series(s)', 'LBicep_TimeSeries', 'LBicep_Acc X Time Series(s)',
                         'LBicep_Acc Y Time Series(s)', 'LBicep_Acc Z Time Series(s)', 'LBicep_GyroXTime Series(s)', 
                         'LBicep_GyroYTime Series(s)', 'LBicep_GyroZTime Series(s)']
    
    df = df.drop(extr_time_series, axis = 1)
    # measurement_cols = [col for col in df.columns if (('ACC' in col or 'GYRO' in col) and 'Time Series' not in col)] #exclude mV and time cols
    # df.columns = df.columns.str.strip()           # Remove leading/trailing spaces (Yuxuan)
    # df = df.apply(pd.to_numeric, errors='coerce') # Conver  t everything to numeric (Yuxuan)
    df = df.replace(['', ' ', 'NA', None], np.nan) #stdize missing data
    df['gender'] = gender
    df['run_num'] = run_num
    return df 

def create_sensor_col(df): 
    columns_to_keep = ['RDelt_EMG_TimeSeries', 'RDelt_IMU_Acc X Time Series(s)', 'gender', 'run_num']
    pdb.set_trace()
    # Identify all measurement columns, including EMG millivolts
    measurement_columns = [col for col in df.columns if any(prefix in col for prefix in ['RDelt', 'LDelt', 'RBicep', 'LBicep']) and col not in columns_to_keep]
    df_melted = df.melt(id_vars=columns_to_keep, value_vars=measurement_columns, var_name="sensor_measurement", value_name="value")
    
    # Extract the Sensor Body Position
    df_melted["Sensor_Body_Position"] = df_melted["sensor_measurement"].str.extract(r'^(RDelt|LDelt|RBicep|LBicep)')
    # Extract measurement type, including EMG millivolts
    df_melted["measurement_type"] = df_melted["sensor_measurement"].str.extract(r'_(EMG_MilliVolts|MilliVolts|ACC X|ACC Y|ACC Z|GYRO X|GYRO Y|GYRO Z)')    # Drop the original sensor_measurement column
    df_melted = df_melted.drop(columns=["sensor_measurement"])    # Pivot the DataFrame so each measurement type becomes a separate column
    df_melted["value"] = df_melted["value"].astype(str).str.strip() #make sure that values are clean if they're strings (no extra space)
    df_melted["value"] = pd.to_numeric(df_melted["value"], errors="coerce") #make sure all values are cast to numeric
    df_melted.fillna(np.nan, inplace=True)
    # Pivot the DataFrame so each measurement type becomes a separate column
    df_pivoted = df_melted.pivot_table(index=['Sensor_Body_Position', 'RDelt_EMG_TimeSeries', 'RDelt_IMU_Acc X Time Series(s)', 'gender', 'run_num'], 
                                    columns='measurement_type', values='value')
    df_pivoted.columns = df_pivoted.columns.get_level_values(0)
    df_pivoted.columns = df_pivoted.columns.str.strip()
    df_pivoted = df_pivoted.reset_index()
    pdb.set_trace()
    df_pivoted.to_csv("pivoted_df.csv")
    return df_pivoted

def standardize_time_series():
    # interpolate()
    pass

def preprocessing(full_df):
    pass
