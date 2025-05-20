import pandas as pd
import numpy as np

def tidy_emg_imu_as_measured(df):
    muscles = ['RDelt', 'LDelt', 'RBicep', 'LBicep']
    all_muscle_tables = []
    for muscle in muscles:
        # EMG rows (as measured)
        emg_df = pd.DataFrame({
            'Muscle': muscle,
            'EMG_TimeSeries': pd.to_numeric(df['EMG_TimeSeries'], errors='coerce'),
            'IMU_TimeSeries': np.nan,
            'EMG_MV': pd.to_numeric(df[f'{muscle}_EMG_MilliVolts'], errors='coerce'),
            'ACC X': np.nan, 'ACC Y': np.nan, 'ACC Z': np.nan,
            'GYRO X': np.nan, 'GYRO Y': np.nan, 'GYRO Z': np.nan
        })
        # IMU rows (as measured)
        imu_df = pd.DataFrame({
            'Muscle': muscle,
            'EMG_TimeSeries': np.nan,
            'IMU_TimeSeries': pd.to_numeric(df['IMU_TimeSeries'], errors='coerce'),
            'EMG_MV': np.nan,
            'ACC X': pd.to_numeric(df[f'{muscle}_ACC X (G)'], errors='coerce'),
            'ACC Y': pd.to_numeric(df[f'{muscle}_ACC Y (G)'], errors='coerce'),
            'ACC Z': pd.to_numeric(df[f'{muscle}_ACC Z (G)'], errors='coerce'),
            'GYRO X': pd.to_numeric(df[f'{muscle}_GYRO X (deg/s)'], errors='coerce'),
            'GYRO Y': pd.to_numeric(df[f'{muscle}_GYRO Y (deg/s)'], errors='coerce'),
            'GYRO Z': pd.to_numeric(df[f'{muscle}_GYRO Z (deg/s)'], errors='coerce')
        })
        all_muscle_tables.append(pd.concat([emg_df, imu_df], ignore_index=True))
    tidy = pd.concat(all_muscle_tables, ignore_index=True)
    # Order and sort (optional)
    columns_order = ['Muscle', 'EMG_TimeSeries', 'IMU_TimeSeries', 'EMG_MV',
                     'ACC X', 'ACC Y', 'ACC Z', 'GYRO X', 'GYRO Y', 'GYRO Z']
    tidy = tidy[columns_order]
    tidy['SortTime'] = tidy['EMG_TimeSeries'].combine_first(tidy['IMU_TimeSeries'])
    tidy = tidy.sort_values(['Muscle', 'SortTime']).drop(columns=['SortTime'])
    df_pivoted_sorted = tidy.sort_values('EMG_TimeSeries', na_position='last')
    return df_pivoted_sorted