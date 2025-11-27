import serial
import time
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path

SERIAL_PORT = 'COM7'
BAUD_RATE = 115200
WINDOW_DURATION = 1.25
SAMPLE_RATE = 100
OUTPUT_DIR = "imu_dataset"
STREAM_TIMEOUT = 0.5
FILENAME_DIGITS = 3
CONVERT_UNITS = True

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

label_name = input("Enter the label name for this recording session (e.g. left, right, circle): ").strip()
if not label_name:
    raise ValueError("Label name cannot be empty.")
label_dir = Path(OUTPUT_DIR) / label_name
label_dir.mkdir(parents=True, exist_ok=True)

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
counter = max(existing_nums)+1 if existing_nums else 1

print(f"Saving files to: {label_dir.resolve()}")
print(f"Starting file counter at: {counter:0{FILENAME_DIGITS}d}")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
except serial.SerialException as e:
    raise SystemExit(f"Could not open serial port {SERIAL_PORT}: {e}")

time.sleep(2)
print(f"\nRecording gestures for label '{label_name}'. Press Ctrl+C to stop.")
print("Press the button on ESP32 to start streaming...")

def parse_line_safe(line):
    """Safely parses a comma-separated string into a list of floats."""
    parts = line.strip().split(',')
    if len(parts) != 7:
        return None
    try:
        return list(map(float, parts))
    except ValueError:
        return None

try:
    while True:
        window_data = []
        stream_active = False
        start_time = None
        last_data_time = time.time()

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
                if CONVERT_UNITS:
                    ax, ay, az = ax * 9.80665, ay * 9.80665, az * 9.80665
                    gx, gy, gz = np.deg2rad(gx), np.deg2rad(gy), np.deg2rad(gz)
                window_data.append([ts, ax, ay, az, gx, gy, gz])
                print(f"{ts}, {ax:.3f}, {ay:.3f}, {az:.3f}, {gx:.6f}, {gy:.6f}, {gz:.6f}")
                start_time = time.time()
                last_data_time = time.time()
                stream_active = True

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
                    if CONVERT_UNITS:
                        ax, ay, az = ax * 9.80665, ay * 9.80665, az * 9.80665
                        gx, gy, gz = np.deg2rad(gx), np.deg2rad(gy), np.deg2rad(gz)
                    window_data.append([ts, ax, ay, az, gx, gy, gz])
                    print(f"{ts}, {ax:.3f}, {ay:.3f}, {az:.3f}, {gx:.6f}, {gy:.6f}, {gz:.6f}")
                    last_data_time = time.time()
            if time.time() - last_data_time > STREAM_TIMEOUT:
                stream_active = False

        if len(window_data) > 0:
            filename = f"{label_name}_{counter:0{FILENAME_DIGITS}d}.csv"
            filepath = label_dir / filename
            df = pd.DataFrame(window_data, columns=['timestamp','ax','ay','az','gx','gy','gz'])
            try:
                df.to_csv(filepath, index=False)
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