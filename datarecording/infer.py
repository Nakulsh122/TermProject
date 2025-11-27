import os
import time
import json
import joblib
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import serial
from scipy.signal import butter, filtfilt, resample

SERIAL_PORT = "COM7"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 0.2
STREAM_TIMEOUT = 0.6
TARGET_LEN = 150
SAMPLE_RATE = 100
LOWPASS_CUTOFF = 15.0
FILTER_ORDER = 4
SAVED_MODEL_DIR = "model_checkpoints"
SCALER_PATH = os.path.join(SAVED_MODEL_DIR, "scaler.pkl")
LABELMAP_PATH = os.path.join(SAVED_MODEL_DIR, "label_map.json")
TORCHSCRIPT_PATH = os.path.join(SAVED_MODEL_DIR, "model_script.pt")
STATE_DICT_PATH = os.path.join(SAVED_MODEL_DIR, "best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Robust1DCNN(nn.Module):
    """Defines the 1D CNN architecture used for inference."""
    def __init__(self, in_channels=11, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.net(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

def butter_lowpass_filter(data, cutoff=LOWPASS_CUTOFF, fs=SAMPLE_RATE, order=FILTER_ORDER):
    """Applies a Butterworth lowpass filter to the input data."""
    if data.shape[0] <= 15:
        return data
    nyq = 0.5 * fs
    normal_cutoff = min(0.999, cutoff / nyq)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def find_peak_index(acc_arr):
    """Finds the index of the maximum magnitude in the acceleration data."""
    mag = np.linalg.norm(acc_arr[:, :3], axis=1)
    return int(np.argmax(mag))

def align_trim_pad(raw_arr, target_len=TARGET_LEN, pre_peak_frac=0.40):
    """Aligns the signal peak, trims to window size, and handles padding."""
    N, C = raw_arr.shape
    if N == 0:
        return np.zeros((target_len, C), dtype=np.float32)
    padded = raw_arr.copy()
    if N < 16:
        pad_len = 16 - N
        padded = np.vstack([raw_arr, np.tile(raw_arr[-1:], (pad_len,1))])
        N = padded.shape[0]

    filtered = butter_lowpass_filter(padded)

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
        baseline = np.mean(filtered[:min(10, N), :], axis=0)
        missing = target_len - window.shape[0]
        pad_arr = np.tile(baseline, (missing,1))
        window = np.vstack([window, pad_arr])
    elif window.shape[0] > target_len:
        window = window[:target_len, :]

    return window.astype(np.float32)

def compute_derived_channels(window):
    """Computes magnitude and derivative channels from the sensor window."""
    ax,ay,az = window[:,0], window[:,1], window[:,2]
    gx,gy,gz = window[:,3], window[:,4], window[:,5]
    acc_mag = np.sqrt(ax*ax + ay*ay + az*az)
    gyro_mag = np.sqrt(gx*gx + gy*gy + gz*gz)
    dax = np.concatenate([[0.0], np.diff(ax)])
    day = np.concatenate([[0.0], np.diff(ay)])
    daz = np.concatenate([[0.0], np.diff(az)])
    return np.stack([ax,ay,az,gx,gy,gz,acc_mag,gyro_mag,dax,day,daz], axis=1).astype(np.float32)

def load_artifacts(savedir=SAVED_MODEL_DIR):
    """Loads the scaler, label map, and trained model from disk."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found at: " + SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)

    if not os.path.exists(LABELMAP_PATH):
        raise FileNotFoundError("Label map not found at: " + LABELMAP_PATH)
    with open(LABELMAP_PATH, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {int(v):k for k,v in label_to_idx.items()}

    model = None
    if os.path.exists(TORCHSCRIPT_PATH):
        try:
            model = torch.jit.load(TORCHSCRIPT_PATH, map_location=DEVICE)
            model.to(DEVICE)
            model.eval()
            print("Loaded TorchScript model.")
            return scaler, idx_to_label, model
        except Exception as e:
            print("Warning: failed to load TorchScript model:", e)

    if os.path.exists(STATE_DICT_PATH):
        n_in = 11
        n_classes = len(idx_to_label)
        model_obj = Robust1DCNN(in_channels=n_in, n_classes=n_classes).to(DEVICE)
        state = torch.load(STATE_DICT_PATH, map_location=DEVICE)
        model_obj.load_state_dict(state)
        model_obj.eval()
        print("Loaded state_dict model.")
        return scaler, idx_to_label, model_obj

    raise FileNotFoundError("No model found in " + savedir)

def parse_line(line):
    """Parses a CSV string line into a list of floats."""
    parts = line.strip().split(",")
    if len(parts) != 7:
        return None
    try:
        vals = list(map(float, parts))
        return vals
    except ValueError:
        return None

def run_once(port=SERIAL_PORT, baud=BAUD_RATE):
    """Listens to the serial port, captures one gesture stream, and runs inference."""
    scaler, idx_to_label, model = load_artifacts()
    ser = None
    try:
        ser = serial.Serial(port, baud, timeout=SERIAL_TIMEOUT)
        time.sleep(2.0)
    except serial.SerialException as e:
        raise SystemExit(f"Could not open serial port {port}: {e}")

    print(f"Listening on {port} @ {baud}. Send one gesture (stream), stop the stream; prediction will be printed when the stream ends.\nPress Ctrl+C to quit.")

    raw_buffer = []
    last_received = None

    try:
        while True:
            raw = ser.readline()
            if raw:
                try:
                    line = raw.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue
                parsed = parse_line(line)
                if parsed is None:
                    continue
                ts, ax, ay, az, gx, gy, gz = parsed
                ax_m, ay_m, az_m = ax * 9.80665, ay * 9.80665, az * 9.80665
                gx_r, gy_r, gz_r = np.deg2rad(gx), np.deg2rad(gy), np.deg2rad(gz)
                raw_buffer.append([ax_m, ay_m, az_m, gx_r, gy_r, gz_r])
                last_received = time.time()
            else:
                if last_received is None:
                    continue
                if (time.time() - last_received) > STREAM_TIMEOUT and len(raw_buffer) > 0:
                    print(f"\nStream stopped; collected {len(raw_buffer)} samples. Preparing window...")

                    raw_arr = np.array(raw_buffer, dtype=np.float32)
                    window = align_trim_pad(raw_arr, target_len=TARGET_LEN)
                    feat = compute_derived_channels(window)
                    feat_scaled = scaler.transform(feat).astype(np.float32)
                    x = torch.from_numpy(feat_scaled).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        pred_idx = int(np.argmax(probs))
                        pred_label = idx_to_label[pred_idx]
                        confidence = float(probs[pred_idx])

                    print(f"Prediction: {pred_label}    confidence: {confidence:.3f}")
                    raw_buffer.clear()
                    last_received = None
                    print("\nReady for next stream... (send gesture and then stop streaming)")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
    finally:
        try:
            if ser:
                ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    run_once()