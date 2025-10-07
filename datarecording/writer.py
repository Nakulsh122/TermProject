# serial_record_with_features.py
import serial
import time
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path

# ----------------- CONFIG -----------------
SERIAL_PORT = 'COM7'       # ESP32 serial port
BAUD_RATE = 115200
WINDOW_DURATION = 1.25     # seconds per window (you said ~1.25s)
SAMPLE_RATE = 100          # ESP32 sampling rate (for reference; not strictly used here)
OUTPUT_DIR = "imu_dataset" # base directory to save CSVs
STREAM_TIMEOUT = 0.5       # seconds to wait before considering stream inactive
FILENAME_DIGITS = 3       # number of digits in filename counter (e.g. 001)
# -----------------------------------------

# Ensure base output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Ask user for label name (only once per script run)
label_name = input("Enter the label name for this recording session (e.g. left, right, circle): ").strip()
if not label_name:
    raise ValueError("Label name cannot be empty.")

# Create label-specific subdirectory
label_dir = Path(OUTPUT_DIR) / label_name
label_dir.mkdir(parents=True, exist_ok=True)

# Determine starting counter based on existing files in label_dir
pattern = re.compile(rf"^{re.escape(label_name)}_(\d{{{FILENAME_DIGITS},}})\.csv$")
existing_nums = []
for p in label_dir.iterdir():
    if p.is_file():
        m = pattern.match(p.name)
        if m:
            try:
                existing_nums.append(int(m.group(1)))
            except ValueError:
                pass
if existing_nums:
    counter = max(existing_nums) + 1
else:
    counter = 1

print(f"Saving files to: {label_dir.resolve()}")
print(f"Starting file counter at: {counter:0{FILENAME_DIGITS}d}")

# Open serial
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
except serial.SerialException as e:
    raise SystemExit(f"Could not open serial port {SERIAL_PORT}: {e}")

time.sleep(2)  # allow ESP32 to reset

print(f"\nRecording gestures for label '{label_name}'. Press Ctrl+C to stop.")
print("Press the button on ESP32 to start streaming...")

def parse_line_safe(line):
    """Try to parse CSV line; return None if fails.
       Expected 7 comma-separated floats: timestamp,ax,ay,az,gx,gy,gz
    """
    parts = line.strip().split(',')
    if len(parts) != 7:
        return None
    try:
        return list(map(float, parts))
    except ValueError:
        return None

def compute_features_and_save(window_data, out_path):
    """
    window_data: list of rows [ts, ax, ay, az, gx, gy, gz]
    out_path: pathlib.Path to save csv
    """
    arr = np.array(window_data, dtype=float)  # shape (N, 7)

    # columns: timestamp, ax, ay, az, gx, gy, gz
    timestamps = arr[:, 0]
    ax = arr[:, 1]
    ay = arr[:, 2]
    az = arr[:, 3]
    gx = arr[:, 4]
    gy = arr[:, 5]
    gz = arr[:, 6]

    # compute magnitudes
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # compute deltas (first difference). Prepend 0 for the first timestep.
    dax = np.concatenate([[0.0], np.diff(ax)])
    day = np.concatenate([[0.0], np.diff(ay)])
    daz = np.concatenate([[0.0], np.diff(az)])

    # assemble dataframe
    df = pd.DataFrame({
        'timestamp': timestamps,
        'ax': ax, 'ay': ay, 'az': az,
        'gx': gx, 'gy': gy, 'gz': gz,
        'acc_mag': acc_mag,
        'gyro_mag': gyro_mag,
        'dax': dax, 'day': day, 'daz': daz
    })

    df.to_csv(out_path, index=False)

try:
    while True:
        window_data = []
        stream_active = False
        start_time = None
        last_data_time = time.time()

        # Wait for data stream to become active
        while not stream_active:
            try:
                raw = ser.readline()
            except serial.SerialException:
                raw = b''
            if not raw:
                continue
            try:
                line = raw.decode('utf-8', errors='ignore').strip()
            except Exception:
                continue
            parsed = parse_line_safe(line)
            if parsed:
                ts, ax, ay, az, gx, gy, gz = parsed
                # convert units if your ESP sends g and deg/s
                ax, ay, az = ax * 9.80665, ay * 9.80665, az * 9.80665
                gx, gy, gz = np.deg2rad(gx), np.deg2rad(gy), np.deg2rad(gz)
                window_data.append([ts, ax, ay, az, gx, gy, gz])
                print(f"{ts}, {ax:.3f}, {ay:.3f}, {az:.3f}, {gx:.6f}, {gy:.6f}, {gz:.6f}")
                start_time = time.time()
                last_data_time = time.time()
                stream_active = True

        # Collect data while stream is active or until WINDOW_DURATION has elapsed since start
        while stream_active or (start_time and (time.time() - start_time < WINDOW_DURATION)):
            try:
                raw = ser.readline()
            except serial.SerialException:
                raw = b''
            if raw:
                try:
                    line = raw.decode('utf-8', errors='ignore').strip()
                except Exception:
                    line = ''
                parsed = parse_line_safe(line)
                if parsed:
                    ts, ax, ay, az, gx, gy, gz = parsed
                    ax, ay, az = ax * 9.80665, ay * 9.80665, az * 9.80665
                    gx, gy, gz = np.deg2rad(gx), np.deg2rad(gy), np.deg2rad(gz)
                    window_data.append([ts, ax, ay, az, gx, gy, gz])
                    print(f"{ts}, {ax:.3f}, {ay:.3f}, {az:.3f}, {gx:.6f}, {gy:.6f}, {gz:.6f}")
                    last_data_time = time.time()
            # check stream timeout
            if time.time() - last_data_time > STREAM_TIMEOUT:
                stream_active = False

        # Save window if we have enough data
        if len(window_data) > 0:
            filename = f"{label_name}_{counter:0{FILENAME_DIGITS}d}.csv"
            filepath = label_dir / filename

            # compute features and save
            try:
                compute_features_and_save(window_data, filepath)
                print(f"\nSaved window {counter} as {filepath}\n")
                counter += 1
            except Exception as e:
                print(f"Failed to save window: {e}")

except KeyboardInterrupt:
    print("\nRecording stopped by user.")

finally:
    try:
        ser.close()
    except Exception:
        pass
    print("Serial port closed. Goodbye.")
