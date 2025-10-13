import os
import numpy as np
import pandas as pd
from scipy.signal import resample
import random

# ---------------- CONFIG ----------------
INPUT_DIR = "imu_dataset_preprocessed"
OUTPUT_DIR = "imu_dataset_augmented"
TARGET_LEN = 150
AUG_PER_SAMPLE = 2  # how many augmented copies per sample
TIME_WARP_MAX = 0.1  # ±10% time stretching
MAG_SCALE_MAX = 0.1  # ±10% scaling
ROT_ANGLE_MAX = 5     # degrees, small rotation
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
gestures = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR,d))]
for g in gestures:
    os.makedirs(os.path.join(OUTPUT_DIR, g), exist_ok=True)

# --------- AUGMENTATION FUNCTIONS ---------
def time_warp(X, max_frac=TIME_WARP_MAX):
    factor = 1 + random.uniform(-max_frac, max_frac)
    length = int(X.shape[0]*factor)
    return resample(X, length, axis=0)

def magnitude_scale(X, max_scale=MAG_SCALE_MAX):
    scale = 1 + random.uniform(-max_scale, max_scale)
    X[:, :6] *= scale  # only ax,ay,az,gx,gy,gz
    return X

def small_rotation(X, max_angle_deg=ROT_ANGLE_MAX):
    angle = np.deg2rad(random.uniform(-max_angle_deg, max_angle_deg))
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle),  np.cos(angle), 0],
                  [0,              0,             1]])
    X[:, :3] = X[:, :3] @ R.T  # rotate accelerometer only
    return X

def pad_or_trim(X, target_len=TARGET_LEN):
    N = X.shape[0]
    if N < target_len:
        pad_len = target_len - N
        baseline = np.mean(X[:min(10,N),:], axis=0)
        pad_arr = np.tile(baseline, (pad_len,1))
        X = np.vstack([X, pad_arr])
    elif N > target_len:
        X = X[:target_len,:]
    return X

# --------- PROCESS DATASET ---------
for gesture in gestures:
    folder_in = os.path.join(INPUT_DIR, gesture)
    folder_out = os.path.join(OUTPUT_DIR, gesture)
    files = [f for f in os.listdir(folder_in) if f.endswith('.csv')]
    for f in files:
        path_in = os.path.join(folder_in, f)
        df = pd.read_csv(path_in)
        X_orig = df.values.astype(np.float32)

        for i in range(AUG_PER_SAMPLE):
            X_aug = X_orig.copy()
            # apply random augmentations
            if random.random() < 0.5:
                X_aug = time_warp(X_aug)
            if random.random() < 0.5:
                X_aug = magnitude_scale(X_aug)
            if random.random() < 0.5:
                X_aug = small_rotation(X_aug)
            X_aug = pad_or_trim(X_aug)

            # save
            base_name = f"{os.path.splitext(f)[0]}_aug{i+1}.csv"
            df_out = pd.DataFrame(X_aug, columns=df.columns)
            df_out.to_csv(os.path.join(folder_out, base_name), index=False)

print("Augmented dataset saved to:", OUTPUT_DIR)
