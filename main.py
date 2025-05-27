import pandas as pd
import numpy as np
from dataCleaning import read_run, column_clean, preprocessing_actions
from dataCleaning import create_sensor_col
from resampling import downsample
from feature_extraction import extract_features
from filtering import bandpass_filter_emg, lowpass_filter_imu
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
    #filter out IMU and EMG outliers using filters: 
    imu_cols = ['ACC X (G)', 'ACC Y (G)', 'ACC Z (G)', 'GYRO X (deg/s)', 'GYRO Y (deg/s)', 'GYRO Z (deg/s)']
    for col in imu_cols:
        combined_df[col + '_filtered'] = combined_df.groupby(['BodyPart', 'run_num', 'gender', 'exo'])[col].transform(lowpass_filter_imu)
    combined_df['EMG_MilliVolts_filtered'] = combined_df.groupby(['BodyPart', 'run_num', 'gender', 'exo'])['EMG_MilliVolts'].transform(bandpass_filter_emg)
    pdb.set_trace()
    features_df = extract_features(combined_df) #TO-DO FIX
    #machine learning on combined_df
    #change the next line to call on features_df instead of combined_df when extracting features is fixed to return more data
    X_train_prepared, X_test_prepared, y_train, y_test, preprocessing_pipeline = preprocessing_actions(combined_df)
    #Return preprocessing_pipeline bc want to preprocess (scale, encode, etc.) any new or test data the same way as your training data.
    return X_train_prepared, X_test_prepared, y_train, y_test, preprocessing_pipeline

def main():
    final_df =  overall_cleaning()

if __name__ == '__main__':
    main()