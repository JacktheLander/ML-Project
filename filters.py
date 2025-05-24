import numpy as np
from scipy.signal import butter, filtfilt

# Bandpass filter for EMG signals (20–450 Hz)
def bandpass_filter_emg(signal, fs=1259, lowcut=20, highcut=450, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Low-pass filter for IMU signals (<20 Hz)
def lowpass_filter_imu(signal, fs=148, cutoff=20, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)
