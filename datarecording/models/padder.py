"""
train_gestures.py

- Loads CSVs from imu_dataset/<label>_*.csv for labels idle,left,right,circlecw
- Creates overlapping windows (WINDOW_SIZE=120, STRIDE=60)
- Scales accel (ax,ay,az) by ACCEL_RANGE and gyro (gx,gy,gz) by GYRO_RANGE to [-1,1]
- Trains a small 1D-CNN on GPU (if available)
- Prints epoch-by-epoch metrics
- Saves Keras model and both float and int8-quantized TFLite models
"""

import os, glob, math, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ---------------- CONFIG ----------------
DATA_DIR = "imu_dataset"
WINDOW_SIZE = 120
STRIDE = WINDOW_SIZE // 2   # 50% overlap -> 60
CHANNELS = ['ax','ay','az','gx','gy','gz']
ACCEL_RANGE = 4.0
GYRO_RANGE = 500.0
LABEL_MAP = {'idle':0, 'left':1, 'right':2, 'circlecw':3}
BATCH_SIZE = 32
EPOCHS = 50
TEST_SIZE = 0.2
RANDOM_SEED = 42
AUGMENT_PROB = 0.25   # augment some windows with small noise
NOISE_STD = 0.01      # small gaussian noise (after normalization)
OUTPUT_DIR = "models_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ----------------------------------------

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------- GPU setup ----------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs found:", gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth on GPU(s).")
    except Exception as e:
        print("Could not set memory growth:", e)
else:
    print("No GPU found — running on CPU. To use GPU, install TF with CUDA support.")

# ---------- utility: sliding windows ----------
def sliding_windows_from_array(arr, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    arr: (N, channels)
    returns: (n_windows, window_size, channels)
    """
    n = arr.shape[0]
    if n < window_size:
        # pad with zeros
        pad = np.zeros((window_size - n, arr.shape[1]), dtype=arr.dtype)
        arr = np.vstack([arr, pad])
        n = arr.shape[0]
    windows = []
    for start in range(0, n - window_size + 1, stride):
        windows.append(arr[start:start+window_size])
    return np.array(windows)

# ---------- load and window all CSVs ----------
X_list = []
y_list = []

print("Loading and windowing CSV files...")
for label, idx in LABEL_MAP.items():
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{label}_*.csv")))
    if len(files) == 0:
        print(f"Warning: no files found for label '{label}' in {DATA_DIR}")
    for fpath in files:
        df = pd.read_csv(fpath)
        # Ensure required channels exist
        if not set(CHANNELS).issubset(df.columns):
            raise RuntimeError(f"File {fpath} missing expected channels {CHANNELS}")
        arr = df[CHANNELS].values.astype(np.float32)  # columns: ax,ay,az,gx,gy,gz
        # scale before windowing (or after — here we scale now)
        arr[:,0:3] = arr[:,0:3] / ACCEL_RANGE
        arr[:,3:6] = arr[:,3:6] / GYRO_RANGE
        # sliding windows
        wins = sliding_windows_from_array(arr, WINDOW_SIZE, STRIDE)
        # optional augmentation: add small noise to a fraction of windows
        for w in wins:
            if random.random() < AUGMENT_PROB:
                noise = np.random.normal(0.0, NOISE_STD, size=w.shape).astype(np.float32)
                w_aug = w + noise
                X_list.append(w_aug)
                y_list.append(idx)
            # add the original window
            X_list.append(w)
            y_list.append(idx)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
print(f"Total windows: {X.shape[0]}, shape per window: {X.shape[1:]}")

# Shuffle dataset
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# Train/test split (stratify to preserve class ratios)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
print("Train:", X_train.shape, y_train.shape, "Test:", X_test.shape, y_test.shape)

# ---------- build model ----------
n_timesteps = WINDOW_SIZE
n_channels = len(CHANNELS)
n_classes = len(LABEL_MAP)

def build_model():
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=(n_timesteps, n_channels)),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    return model

model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------- callbacks ----------
class EpochLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', 0.0)
        acc = logs.get('accuracy', 0.0)
        val_loss = logs.get('val_loss', 0.0)
        val_acc = logs.get('val_accuracy', 0.0)
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {loss:.4f}, acc: {acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

ckpt_path = os.path.join(OUTPUT_DIR, "best_model.h5")
callbacks = [
    ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', mode='max', patience=8, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    EpochLogger()
]

# ---------- train ----------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=callbacks
)

# ---------- evaluate ----------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal test accuracy: {acc*100:.2f}% (loss {loss:.4f})")

# ---------- save keras model ----------
keras_path = os.path.join(OUTPUT_DIR, "gesture_model_keras.h5")
model.save(keras_path)
print("Saved Keras model to:", keras_path)

# ---------- convert to TFLite (float) ----------
tflite_float_path = os.path.join(OUTPUT_DIR, "gesture_model_float.tflite")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(tflite_float_path, "wb") as f:
    f.write(tflite_model)
print("Saved float TFLite model to:", tflite_float_path)

# ---------- convert to TFLite (int8 quantized) ----------
# Use representative dataset from training set
def representative_dataset_generator():
    for i in range(min(100, X_train.shape[0])):  # use up to 100 samples
        sample = X_train[i]
        # model expects float32 input; representative generator yields list of [sample]
        yield [np.expand_dims(sample.astype(np.float32), axis=0)]

converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
converter_q.representative_dataset = representative_dataset_generator
# allow full integer quantization
converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_q.inference_input_type = tf.int8
converter_q.inference_output_type = tf.int8

try:
    tflite_quant_path = os.path.join(OUTPUT_DIR, "gesture_model_int8.tflite")
    tflite_q = converter_q.convert()
    with open(tflite_quant_path, "wb") as f:
        f.write(tflite_q)
    print("Saved int8 quantized TFLite model to:", tflite_quant_path)
except Exception as e:
    print("Quantized conversion failed:", e)
    print("You can still use the float TFLite model on-device if int8 fails.")

print("\nAll done. Models are in:", OUTPUT_DIR)
