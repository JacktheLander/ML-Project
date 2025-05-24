import pandas as pd
from filters import bandpass_filter_emg, lowpass_filter_imu
from feature_extraction import extract_features_sliding_window

def read_run(file_path):
    return pd.read_csv(file_path, skiprows=5, low_memory=False)

def column_clean(df, run_num, gender):
    df = df.dropna(how='all')
    df = df.reset_index(drop=True)
    df = df.rename(columns={
        'Sensor 1': 'Right Bicep',
        'Sensor 2': 'Left Bicep',
        'RDelt_EMG_MilliVolts': 'EMG',
        'RDelt_ACC X (G)': 'ACC_X',
        'RDelt_ACC Y (G)': 'ACC_Y',
        'RDelt_ACC Z (G)': 'ACC_Z',
        'RDelt_GYRO X (deg/s)': 'GYRO_X',
        'RDelt_GYRO Y (deg/s)': 'GYRO_Y',
        'RDelt_GYRO Z (deg/s)': 'GYRO_Z'
    })
    df['run'] = run_num
    df['gender'] = gender
    return df

def process_all_files():
    files = [
        ("/content/P3_Exo_1_0.csv", 2, 'male', 'Exo'),
        ("/content/P3_NoExo_1_0.csv", 1, 'male', 'NoExo'),
        ("/content/P4_Exo_1_0.csv", 2, 'female', 'Exo'),
        ("/content/P4_NoExo_1_0.csv", 1, 'female', 'NoExo')
    ]

    results = []
    for path, run, gender, label in files:
        df = read_run(path)
        df = column_clean(df, run, gender)

        for col in ['EMG', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        df['Filtered_EMG'] = bandpass_filter_emg(df['EMG'])
        for axis in ['X', 'Y', 'Z']:
            df[f'Filtered_ACC_{axis}'] = lowpass_filter_imu(df[f'ACC_{axis}'])
            df[f'Filtered_GYRO_{axis}'] = lowpass_filter_imu(df[f'GYRO_{axis}'])

        feats = extract_features_sliding_window(
            df,
            emg_col='Filtered_EMG',
            acc_cols=[f'Filtered_ACC_{a}' for a in ['X', 'Y', 'Z']],
            gyro_cols=[f'Filtered_GYRO_{a}' for a in ['X', 'Y', 'Z']],
            label=label,
            gender=gender
        )
        results.append(feats)

    return pd.concat(results, ignore_index=True)
