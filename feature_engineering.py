def add_derived_features(df):
    df['emg_max_x_gyro_peak'] = df['emg_max'] * df['gyro_peak']
    df['gyro_range_div_acc_range'] = df['gyro_range'] / (df['acc_range'] + 1e-6)
    df['emg_min_plus_acc_peak'] = df['emg_min'] + df['acc_peak']
    df['emg_range'] = df['emg_max'] - df['emg_min']
    df['emg_variance'] = (df['emg_std'] if 'emg_std' in df else 0.01) ** 2
    df['acc_std'] = df['acc_range'] / 2
    df['gyro_std'] = df['gyro_range'] / 2
    df['acc_energy'] = df['acc_peak'] ** 2
    df['gyro_energy'] = df['gyro_peak'] ** 2
    return df
