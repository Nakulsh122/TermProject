# IMU Gesture Recognition using ESP32 & PyTorch

A complete end-to-end pipeline for recognizing hand gestures using an **ESP32** and an **IMU sensor** (like MPU6050 / ADXL345 + Gyro).  
The project includes scripts for **data recording**, **preprocessing**, **training**, and **real-time inference** on live sensor data.

---

## 🧭 Project Overview

This system collects accelerometer and gyroscope data from an ESP32 over serial, preprocesses and aligns signals, trains a deep learning model to recognize gestures, and performs one-shot inference on live samples.

Supported gestures:
- **up**
- **down**
- **left**
- **right**
- **circle**
- **idle**

---

## ⚙️ Project Workflow

1. **Connect ESP32 + IMU**
   - Make proper I2C/SPI connections as required by your IMU (SDA/SCL or MISO/MOSI/SCK).
   - Flash the Arduino sketch (`sketch_sep28a.ino`) to your ESP32.

2. **Record IMU data**
   - Edit the `SERIAL_PORT` and `BAUD_RATE` in `writer.py` to match your setup.
   - Run:
     ```bash
     python writer.py
     ```
   - Perform gestures one by one. Each recording is saved as:
     ```
     imu_dataset/<gesture>/<gesture>_<n>.csv
     ```

3. **Preprocess Data**
   - Cleans, filters, and normalizes IMU readings.
   - Run:
     ```bash
     python preprocessor.py
     ```
   - Output is saved to:
     ```
     imu_dataset_preprocessed/
     ```

4. **Train Model**
   - Trains a PyTorch classifier using preprocessed data.
   - Run:
     ```bash
     python model.py
     ```
   - Artifacts are saved under `model_checkpoints/`:
     ```
     best_model.pth
     scaler.pkl
     label_map.json
     model_script.pt
     ```

5. **Run Inference on Live Data**
   - Connect ESP32 again and run:
     ```bash
     python infer.py
     ```
   - The system waits for one gesture window, classifies it, prints the **predicted gesture** and **confidence**, and resets for the next sample.

---

## 📁 Directory Structure

+-- imu_dataset/ # Raw recorded CSVs (by gesture)
+---- circle/
+---- down/
+---- idle/
+---- left/
+---- right/
+---- up/
+
+-- model_checkpoints/ # Model artifacts
+---- best_model.pth
+---- scaler.pkl
+---- label_map.json
+---- model_script.pt
+
+-- sketch_sep28a/
+---- sketch_sep28a.ino # ESP32 Arduino sketch
+
+-- writer.py # Records IMU data via serial
+--  preprocessor.py # Preprocesses raw CSVs
+-- model.py # Training pipeline
+-- infer.py # Real-time gesture inference
+-- sampler.py # (Optional) Data sampling helpers
+-- utils/ # (Optional) utility modules


---

## 🧩 Dependencies

Create a Python environment and install required packages:

```bash
python -m venv venv
source venv/bin/activate      # (Windows: venv\Scripts\activate)
pip install -U pip
pip install numpy pandas scikit-learn scipy matplotlib torch joblib pyserial


Each folder corresponds to one gesture label.  
Each CSV file inside represents **one gesture sample recording**, typically about **1–2 seconds** long.

---

## 🧾 CSV Format

Each `.csv` file contains readings sampled at **~100 Hz** from the IMU sensor connected to the ESP32.

| Column Name | Description | Units | Source |
|--------------|--------------|--------|---------|
| `timestamp` | Time since start of recording | ms | ESP32 (millis) |
| `ax` | Acceleration in X-axis | m/s² | Accelerometer |
| `ay` | Acceleration in Y-axis | m/s² | Accelerometer |
| `az` | Acceleration in Z-axis | m/s² | Accelerometer |
| `gx` | Angular velocity in X-axis | °/s | Gyroscope |
| `gy` | Angular velocity in Y-axis | °/s | Gyroscope |
| `gz` | Angular velocity in Z-axis | °/s | Gyroscope |
| `acc_mag` | Magnitude of acceleration vector: √(ax² + ay² + az²) | m/s² | Derived |
| `gyro_mag` | Magnitude of angular velocity vector: √(gx² + gy² + gz²) | °/s | Derived |
| `dax`, `day`, `daz` | Change (delta) in acceleration between consecutive samples | m/s³ | Derived |



---

## 🧮 Data Characteristics

- **Sampling Frequency:** ~100 Hz (i.e., 100 samples per second)  
- **Gesture Duration:** ~1.0 – 1.5 seconds per file (~100–150 samples)  
- **Subjects:** ~9 unique users (merged in training phase)
- **Total Gestures:** 6 (circle, up, down, left, right, idle)
- **Recording Environment:** Stationary handheld IMU; small orientation variations between users

---

## 🧰 How Data Was Collected

1. The ESP32 streams IMU data via Serial at the set baud rate.
2. The `writer.py` script captures the live serial feed and writes data to `.csv`.
3. Each recording session corresponds to one gesture (manually labeled via folder name).

---

## 🧼 Preprocessing Notes

During preprocessing (`preprocessor.py`):

- The timestamp is normalized and converted to consistent sampling intervals.  
- Columns like `acc_mag`, `gyro_mag`, `dax`, `day`, `daz` are added if missing.  
- A **StandardScaler** is fit to normalize feature values (stored as `scaler.pkl`).  
- Final data is reshaped into sliding windows for model training.

---

## 📈 Example Sample

| timestamp | ax | ay | az | gx | gy | gz | acc_mag | gyro_mag | dax | day | daz |
|------------|----|----|----|----|----|----|----------|-----------|-----|-----|-----|
| 0 | -0.02 | 0.01 | 9.81 | 0.5 | 0.2 | 0.1 | 9.81 | 0.55 | 0.0 | 0.0 | 0.0 |
| 10 | -0.03 | 0.02 | 9.78 | 0.6 | 0.3 | 0.2 | 9.78 | 0.67 | -0.01 | 0.01 | -0.03 |
| 20 | -0.05 | 0.04 | 9.75 | 0.8 | 0.4 | 0.3 | 9.75 | 0.92 | -0.02 | 0.02 | -0.03 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 🧠 Labels and Mapping

| Folder | Label Index |
|---------|--------------|
| `circle` | 0 |
| `down`   | 1 |
| `idle`   | 2 |
| `left`   | 3 |
| `right`  | 4 |
| `up`     | 5 |

Label mapping is stored in:  

