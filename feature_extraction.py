# Calculations for Feature Extraction from Project_Guide
def compute_emg_features(signal):
    return {
        'mean': np.mean(signal),
        'max': np.max(signal),
        'min': np.min(signal),
        'std': np.std(signal),
        'rms': np.sqrt(np.mean(signal**2))
    }

def compute_accel_features(a_x, a_y, a_z):
    a_mag = np.sqrt(a_x**2 + a_y**2 + a_z**2)
    
    features = {
        'peak_accel': np.max(a_mag),
        'mean_accel': np.mean(a_mag),
        'total_accel': np.sqrt(np.mean(a_x**2) + np.mean(a_y**2) + np.mean(a_z**2)),
        'accel_range': np.max(a_mag) - np.min(a_mag)
    }
    return features

def compute_gyro_features(w_x, w_y, w_z):
    w_mag = np.sqrt(w_x**2 + w_y**2 + w_z**2)
    
    features = {
        'peak_angular_vel': np.max(w_mag),
        'mean_angular_vel': np.mean(w_mag),
        'total_angular_vel': np.sqrt(np.mean(w_x**2) + np.mean(w_y**2) + np.mean(w_z**2)),
        'angular_vel_range': np.max(w_mag) - np.min(w_mag)
    }
    return features 


        # fft_mean = mean(valid_freqs * valid_fft)
        # fft_median = median(valid_freqs * valid_fft)
        # fft_power = np.sum(valid_fft**2)

        # feature_row = {
        #     'emg_max': emg.max(),
        #     'emg_min': emg.min(),
        #     'emg_rms': np.sqrt(np.mean(emg**2)),
        #     'acc_peak': np.linalg.norm(acc, axis=1).max(),
        #     'acc_range': np.ptp(np.linalg.norm(acc, axis=1)),
        #     'gyro_peak': np.linalg.norm(gyro, axis=1).max(),
        #     'gyro_range': np.ptp(np.linalg.norm(gyro, axis=1)),
        #     'emg_fft_mean_freq': fft_mean,
        #     'emg_fft_median_freq': fft_median,
        #     'emg_fft_power': fft_power,
        #     'label': label,
        #     'gender': gender
        # }