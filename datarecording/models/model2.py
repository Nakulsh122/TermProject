"""
train_gestures_updated.py

Updated training script for IMU gesture recognition:
 - Each window exactly 120 points (pad/trim)
 - Per-channel normalization
 - Reduced model capacity + stronger dropout/L2
 - Augmentations: jitter, scaling, time-shift, random-crop-resize
 - tf.data pipeline for efficient training
 - file-level GroupShuffleSplit to avoid leakage
 - Evaluation: confusion matrix + classification report
 - Saves Keras model + float / int8 TFLite

Usage:
 - Place CSV files as imu_dataset/<label>_*.csv with channels: timestamp,ax,ay,az,gx,gy,gz
 - Run: python train_gestures_updated.py
"""

import os
import glob
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, GlobalAveragePooling1D,
                                     Dense, Dropout, SpatialDropout1D)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)

# ---------------- CONFIG ----------------
DATA_DIR = "imu_dataset"
WINDOW_SIZE = 120
CHANNELS = ['ax','ay','az','gx','gy','gz']
ACCEL_RANGE = 4.0
GYRO_RANGE = 500.0
LABEL_MAP = {'idle':0, 'left':1, 'right':2, 'circlecw':3}
BATCH_SIZE = 32
EPOCHS = 50
TEST_SIZE = 0.2
RANDOM_SEED = 42
OUTPUT_DIR = "models_out"
AUGMENT_PROB = 0.6  # probability to attempt augmentation for a training window
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ----------------------------------------

# ---------- utils: pad/trim ----------

def pad_or_trim_center(arr, target=WINDOW_SIZE):
    L = arr.shape[0]
    if L == target:
        return arr
    if L > target:
        start = (L - target) // 2
        return arr[start:start+target]
    pad_before = (target - L) // 2
    pad_after = target - L - pad_before
    pad_b = np.zeros((pad_before, arr.shape[1]), dtype=arr.dtype)
    pad_a = np.zeros((pad_after, arr.shape[1]), dtype=arr.dtype)
    return np.vstack([pad_b, arr, pad_a])

# ---------- augmentations (numpy) ----------

def jitter(x, sigma=0.03):
    return x + np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)

def scaling(x, sigma=0.08):
    factor = np.random.normal(1.0, sigma, size=(x.shape[1],)).astype(np.float32)
    return (x * factor).astype(np.float32)

def time_shift(x, max_shift=8):
    shift = np.random.randint(-max_shift, max_shift+1)
    return np.roll(x, shift, axis=0)

def random_crop_resize(x, crop_frac_range=(0.85, 1.0)):
    L = x.shape[0]
    frac = np.random.uniform(crop_frac_range[0], crop_frac_range[1])
    crop_L = max(2, int(L * frac))
    start = np.random.randint(0, max(1, L - crop_L + 1))
    cropped = x[start:start+crop_L]
    # resize back
    xp = np.arange(cropped.shape[0])
    xnew = np.linspace(0, cropped.shape[0]-1, L)
    out = np.zeros((L, cropped.shape[1]), dtype=cropped.dtype)
    for c in range(cropped.shape[1]):
        out[:, c] = np.interp(xnew, xp, cropped[:, c])
    return out

def augment_window(x):
    if np.random.rand() < 0.5:
        x = jitter(x, sigma=0.03)
    if np.random.rand() < 0.4:
        x = scaling(x, sigma=0.08)
    if np.random.rand() < 0.5:
        x = time_shift(x, max_shift=8)
    if np.random.rand() < 0.2:
        x = random_crop_resize(x, crop_frac_range=(0.85,1.0))
    return x.astype(np.float32)

# ---------- load CSVs ----------

def load_windows_from_csvs(data_dir=DATA_DIR, window_size=WINDOW_SIZE,
                           channels=CHANNELS, label_map=LABEL_MAP):
    X_list, y_list, groups = [], [], []
    files_per_label = defaultdict(list)
    print("Loading and windowing CSV files...")
    for label in label_map:
        pat = os.path.join(data_dir, f"{label}_*.csv")
        files = sorted(glob.glob(pat))
        if len(files) == 0:
            print(f"Warning: no files found for label '{label}'")
        for fpath in files:
            files_per_label[label].append(fpath)
            df = pd.read_csv(fpath)
            arr = df[channels].values.astype(np.float32)
            arr[:,0:3] /= ACCEL_RANGE
            arr[:,3:6] /= GYRO_RANGE
            # ensure exactly WINDOW_SIZE points
            arr2 = pad_or_trim_center(arr, target=window_size)
            X_list.append(arr2)
            y_list.append(label_map[label])
            groups.append(fpath)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups)
    print(f"Total windows: {X.shape[0]}, shape per window: {X.shape[1:]}")
    for k,v in files_per_label.items():
        print(f"  {k}: {len(v)} files")
    # global per-channel normalization
    mean = X.mean(axis=(0,1), keepdims=True)
    std = X.std(axis=(0,1), keepdims=True) + 1e-6
    X = (X - mean) / std
    return X, y, groups

# ---------- tf.data pipeline ----------

def make_datasets(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, training=True):
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    def _augment_np(x, y):
        x = tf.numpy_function(lambda z: augment_window(z), [x], tf.float32)
        x.set_shape([WINDOW_SIZE, len(CHANNELS)])
        return x, y

    if training:
        train_ds = train_ds.shuffle(1024, seed=RANDOM_SEED)
        train_ds = train_ds.map(_augment_np, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

# ---------- model ----------

def build_model_small(window_size=WINDOW_SIZE, n_channels=len(CHANNELS), n_classes=len(LABEL_MAP)):
    inp = Input(shape=(window_size, n_channels))
    x = Conv1D(8, kernel_size=7, padding='same', activation='relu', kernel_regularizer=l2(1e-3))(inp)
    x = BatchNormalization()(x)
    x = Conv1D(16, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(0.4)(x)
    x = Conv1D(16, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.5)(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    return model

# ---------- representative generator ----------

def representative_dataset_generator(X_train, num_samples=100):
    n = min(num_samples, X_train.shape[0])
    for i in range(n):
        sample = X_train[i:i+1].astype(np.float32)
        yield [sample]

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=DATA_DIR)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--test_size', type=float, default=TEST_SIZE)
    parser.add_argument('--output_dir', default=OUTPUT_DIR)
    args = parser.parse_args()

    X, y, groups = load_windows_from_csvs(data_dir=args.data_dir)

    # file-level split to avoid leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Train:", X_train.shape, y_train.shape, "Test:", X_test.shape, y_test.shape)

    train_ds, val_ds = make_datasets(X_train, y_train, X_test, y_test, batch_size=args.batch_size, training=True)

    model = build_model_small()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'best_model.h5')

    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', mode='min', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # evaluation
    print('\nEvaluating on holdout test windows...')
    preds = model.predict(X_test, batch_size=args.batch_size)
    y_pred = np.argmax(preds, axis=1)
    print('\nClassification report:')
    labels_names = list(LABEL_MAP.keys())
    print(classification_report(y_test, y_pred, target_names=labels_names))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

    # save keras model
    keras_path = os.path.join(args.output_dir, 'gesture_model_keras.h5')
    model.save(keras_path)
    print('Saved Keras model to:', keras_path)

    # save float TFLite
    tflite_float_path = os.path.join(args.output_dir, 'gesture_model_float.tflite')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    try:
        tflite_model = converter.convert()
        with open(tflite_float_path, 'wb') as f:
            f.write(tflite_model)
        print('Saved float TFLite model to:', tflite_float_path)
    except Exception as e:
        print('Float TFLite conversion failed:', e)

    # save int8 quantized model
    try:
        converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_q.representative_dataset = lambda: representative_dataset_generator(X_train, num_samples=100)
        converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_q.inference_input_type = tf.int8
        converter_q.inference_output_type = tf.int8
        tflite_q = converter_q.convert()
        tflite_quant_path = os.path.join(args.output_dir, 'gesture_model_int8.tflite')
        with open(tflite_quant_path, 'wb') as f:
            f.write(tflite_q)
        print('Saved int8 quantized TFLite model to:', tflite_quant_path)
    except Exception as e:
        print('Quantized conversion failed:', e)
        print('Use float TFLite if int8 fails.')

    print('\nAll done. Models are in:', args.output_dir)

if __name__ == '__main__':
    main()
