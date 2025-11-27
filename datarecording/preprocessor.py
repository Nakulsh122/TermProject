import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

INPUT_DIR = "imu_dataset"
OUTPUT_DIR = "imu_dataset_preprocessed"
TARGET_LEN = 150
SAMPLE_RATE = 100
LOWPASS_CUTOFF = 15.0
FILTER_ORDER = 4
PRE_PEAK_FRAC = 0.40

os.makedirs(OUTPUT_DIR, exist_ok=True)
gestures = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR,d))]
for g in gestures:
    os.makedirs(os.path.join(OUTPUT_DIR, g), exist_ok=True)

def butter_lowpass_filter(data, cutoff=LOWPASS_CUTOFF, fs=SAMPLE_RATE, order=FILTER_ORDER):
    """Applies a Butterworth lowpass filter to remove noise from the data."""
    if data.shape[0] <= max(16, order*3):
        return data
    nyq = 0.5 * fs
    normal_cutoff = min(0.999, cutoff / nyq)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    try:
        return filtfilt(b, a, data, axis=0)
    except Exception:
        return data

def find_peak_index(acc_arr):
    """Identifies the index of the peak magnitude in the acceleration data."""
    mag = np.linalg.norm(acc_arr, axis=1)
    return int(np.argmax(mag))

def align_trim_pad(raw_arr, target_len=TARGET_LEN, pre_peak_frac=PRE_PEAK_FRAC):
    """Centers the data on the peak acceleration and fits it to the target length."""
    N, C = raw_arr.shape
    if N == 0:
        return np.zeros((target_len, C))
    if N < target_len:
        baseline = np.mean(raw_arr, axis=0, keepdims=True)
        pad_len = target_len - N
        raw_arr = np.vstack([raw_arr, np.tile(baseline, (pad_len,1))])
        N = raw_arr.shape[0]
    filtered = butter_lowpass_filter(raw_arr)
    peak_idx = find_peak_index(filtered[:, :3])
    peak_target = int(round(pre_peak_frac * target_len))
    start = peak_idx - peak_target
    end = start + target_len
    if start < 0:
        start, end = 0, target_len
    if end > N:
        end = N
        start = max(0, N - target_len)
    window = filtered[start:end, :]
    if window.shape[0] < target_len:
        baseline = np.mean(filtered[:min(10,N),:], axis=0)
        missing = target_len - window.shape[0]
        pad_arr = np.tile(baseline, (missing,1))
        window = np.vstack([window, pad_arr])
    elif window.shape[0] > target_len:
        window = window[:target_len,:]
    return window

def compute_derived_channels(window):
    """Calculates magnitude and derivative features for the IMU data window."""
    ax,ay,az = window[:,0], window[:,1], window[:,2]
    gx,gy,gz = window[:,3], window[:,4], window[:,5]
    acc_mag = np.sqrt(ax*ax + ay*ay + az*az)
    gyro_mag = np.sqrt(gx*gx + gy*gy + gz*gz)
    dax = np.concatenate([[0.0], np.diff(ax)])
    day = np.concatenate([[0.0], np.diff(ay)])
    daz = np.concatenate([[0.0], np.diff(az)])
    return np.stack([ax,ay,az,gx,gy,gz,acc_mag,gyro_mag,dax,day,daz], axis=1)

for gesture in gestures:
    folder_in = os.path.join(INPUT_DIR, gesture)
    folder_out = os.path.join(OUTPUT_DIR, gesture)
    files = [f for f in os.listdir(folder_in) if f.lower().endswith('.csv')]
    for f in files:
        path_in = os.path.join(folder_in, f)
        path_out = os.path.join(folder_out, f)
        df = pd.read_csv(path_in)
        cols = [c.lower() for c in df.columns]
        if set(['ax','ay','az','gx','gy','gz','acc_mag','gyro_mag','dax','day','daz']).issubset(set(cols)):
            arr = df[['ax','ay','az','gx','gy','gz','acc_mag','gyro_mag','dax','day','daz']].values
            if arr.shape[0] != TARGET_LEN:
                raw_first6 = df[['ax','ay','az','gx','gy','gz']].values
                window_raw = align_trim_pad(raw_first6, TARGET_LEN)
                features = compute_derived_channels(window_raw)
            else:
                features = arr
        else:
            if 'timestamp' in df.columns:
                arr = df.drop(columns=['timestamp']).values
            else:
                arr = df.values
            raw = arr[:, :6]
            window = align_trim_pad(raw, TARGET_LEN)
            features = compute_derived_channels(window)
        df_out = pd.DataFrame(features, columns=['ax','ay','az','gx','gy','gz','acc_mag','gyro_mag','dax','day','daz'])
        df_out.to_csv(path_out, index=False)

print("All files preprocessed with gesture centered in window and saved in", OUTPUT_DIR)