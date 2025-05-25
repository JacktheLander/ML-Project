import pandas as pd
import numpy as np
from dataCleaning import read_run, column_clean, preprocessing
from dataCleaning import create_sensor_col
from resampling import downsample
from feature_extraction import compute_emg_features, compute_accel_features, compute_gyro_features

import pdb

def overall_cleaning():
    df_p3_exo = read_run("P3_Exo_1_0.csv") # 2nd run, male
    df_p3_noexo = read_run("P3_NoExo_1_0.csv") # first run, male
    df_p4_exo = read_run("P4_Exo_1_0.csv") # 1st run female
    df_p4_noexo = read_run("P4_NoExo_1_0.csv") # 2nd female

    df_p3_exo = column_clean(df_p3_exo)
    df_p3_noexo = column_clean(df_p3_noexo)
    df_p4_exo = column_clean(df_p4_exo)
    df_p4_noexo = column_clean(df_p4_noexo)
    #downsample EMG to match IMU
    df_p3_exo = downsample(df_p3_exo)
    df_p3_noexo = downsample(df_p3_noexo)
    df_p4_exo = downsample(df_p4_exo)
    df_p4_noexo = downsample(df_p4_noexo)
    #melt sensor columns into a body part sensor
    df_p3_exo = create_sensor_col(df_p3_exo, run_num = 2, gender = 'male', exo=True)
    df_p3_noexo = create_sensor_col(df_p3_noexo, run_num = 1, gender = 'male', exo=False)
    df_p4_exo = create_sensor_col(df_p4_exo, run_num = 1, gender = 'female', exo=True)
    df_p4_noexo = create_sensor_col(df_p4_noexo, run_num = 2, gender = 'female', exo=False)
    dfs = [df_p3_exo, df_p3_noexo, df_p4_exo, df_p4_noexo] #jack's list for the data cleaning he does later.
    combined_df = pd.concat(dfs, ignore_index=True)
    # Run functions to extract features for each dataframe
    pdb.set_trace() 
    emg_features = compute_emg_features(combined_df)
    accel_features = compute_accel_features(combined_df)
    gyro_features = compute_gyro_features(combined_df)  # feature_sets = []
    #do this on EMG cols:
    # bandpass_filter_emg(df)
    #on IMU cols: 
    # lowpass_filter_imu(df)
    
    return combined_df

def main():
    final_df =  overall_cleaning()

if __name__ == '__main__':
    main()