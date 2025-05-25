import pandas as pd
import numpy as np
from dataCleaning import read_run, column_clean, preprocessing
from dataCleaning import create_sensor_col, standardize_time_series
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
    #upsample IMU to match EMG
    df_p3_exo = standardize_time_series(df_p3_exo)
    df_p3_noexo = standardize_time_series(df_p3_noexo)
    df_p4_exo = standardize_time_series(df_p4_exo)
    df_p4_noexo = standardize_time_series(df_p4_noexo)
    df_p3_exo = create_sensor_col(df_p3_exo, run_num = 2, gender = 'male', exo=True)
    df_p3_noexo = create_sensor_col(df_p3_noexo, run_num = 1, gender = 'male', exo=False)
    df_p4_exo = create_sensor_col(df_p4_exo, run_num = 1, gender = 'female', exo=True)
    df_p4_noexo = create_sensor_col(df_p4_noexo, run_num = 2, gender = 'female', exo=False)

    dfs = [df_p3_exo, df_p3_noexo, df_p4_exo, df_p4_noexo] #jack's list for the data cleaning he does later.
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = standardize_time_series(combined_df)
    # # Show the head of the data
    # df_p3_exo.describe()
    # df_p3_noexo.head()
    # df_p4_exo.head()
    # df_p4_noexo.head()
    # # Choose inputs
    # features = df_p3_exo[['EMG 1 (mV)', 'ACC X (G)', 'ACC Y (G)', 'ACC Z (G)', 'GYRO X (deg/s)', 'GYRO Y (deg/s)', 'GYRO Z (deg/s)']].dropna()
    # features.head()
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
    #TO-DO make Exo or No Exo variable?
    #imputation/preprocessing
    # feature_sets now contains extracted features for each df
    p3exo_feats, p3noexo_feats, p4exo_feats, p4noexo_feats = feature_sets
    return p3exo_feats, p3noexo_feats, p4exo_feats, p4noexo_feats


def main():
    p3exo_feats, p3noexo_feats, p4exo_feats, p4noexo_feats = overall_cleaning()


if __name__ == '__main__':
    main()