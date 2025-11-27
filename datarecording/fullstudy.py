#!/usr/bin/env python3
import os
import time
import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.signal import resample
import joblib

DATA_DIR = "imu_dataset_preprocessed"
OUT_ROOT = "report_results"
BASE_DIR = os.path.join(OUT_ROOT, "base_training")
FEATURE_DIR = os.path.join(OUT_ROOT, "feature_study")
SEQ_DIR = os.path.join(OUT_ROOT, "seq_len_study")
SENSOR_DIR = os.path.join(OUT_ROOT, "sensor_study")
AUG_DIR = os.path.join(OUT_ROOT, "augmentation_study")
SUMMARY_LOG = os.path.join(OUT_ROOT, "summary_log.txt")

for d in [OUT_ROOT, BASE_DIR, FEATURE_DIR, SEQ_DIR, SENSOR_DIR, AUG_DIR]:
    os.makedirs(d, exist_ok=True)

NUM_EPOCHS = 40
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
TARGET_LEN = 150
SEED = 42
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIME_WARP_MAX = 0.1
MAG_SCALE_MAX = 0.1
ROT_ANGLE_MAX = 5

ACC_IND = [0,1,2]
GYR_IND = [3,4,5]
DER_IND = [6,7,8,9,10]
ALL_FEATURES = list(range(11))
MAX_FEATURES = 11

SEQ_LEN_LIST = [50, 100, 150, 200]
AUG_MODES = ["none","timewarp","magscale","rotate","combined"]

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def log(msg, to_console=True):
    """Logs a message with a timestamp to the console and a summary file."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if to_console:
        print(line)
    with open(SUMMARY_LOG, "a") as f:
        f.write(line + "\n")

class Robust1DCNN(nn.Module):
    """Defines a robust 1D CNN architecture for time-series classification."""
    def __init__(self, in_channels=11, n_classes=6, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)

def pad_or_trim(X, target_len=TARGET_LEN):
    """Adjusts the input array to the target length by padding or trimming."""
    N = X.shape[0]
    if N < target_len:
        pad_len = target_len - N
        baseline = np.mean(X[:min(10, N), :], axis=0)
        pad = np.tile(baseline, (pad_len, 1))
        X = np.vstack([X, pad])
    elif N > target_len:
        X = X[:target_len, :]
    return X

def compute_derived_channels(X):
    """Calculates magnitude and derivative features from raw acceleration and gyro data."""
    ax, ay, az, gx, gy, gz = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4], X[:,5]
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    dax = np.diff(ax, prepend=ax[0])
    day = np.diff(ay, prepend=ay[0])
    daz = np.diff(az, prepend=az[0])
    derived = np.stack([acc_mag, gyro_mag, dax, day, daz], axis=1)
    return np.concatenate([X[:,:6], derived], axis=1)

def scale_array_select(X, scaler, cols):
    """Scales specific columns of the input array using a pre-fitted global scaler."""
    mean = scaler.mean_[cols]
    scale = scaler.scale_[cols]
    return (X[:, cols] - mean) / (scale + 1e-12)

def time_warp(X, max_frac=TIME_WARP_MAX):
    """Resamples the input signal to simulate time warping."""
    factor = 1 + random.uniform(-max_frac, max_frac)
    new_len = max(1, int(X.shape[0] * factor))
    return resample(X, new_len, axis=0)

def magnitude_scale(X, max_scale=MAG_SCALE_MAX):
    """Scales the magnitude of the raw sensor channels."""
    scale = 1 + random.uniform(-max_scale, max_scale)
    X[:, :6] = X[:, :6] * scale
    return X

def small_rotation(X, max_angle_deg=ROT_ANGLE_MAX):
    """Rotates the acceleration vectors by a small random angle."""
    angle = np.deg2rad(random.uniform(-max_angle_deg, max_angle_deg))
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle),  np.cos(angle), 0],
                  [0, 0, 1]])
    X[:, :3] = X[:, :3] @ R.T
    return X

def apply_aug_mode_raw(X_raw, mode):
    """Applies the specified augmentation mode to raw sensor data."""
    X = X_raw.copy()
    if mode == "none":
        return X
    if mode == "timewarp":
        return time_warp(X)
    if mode == "magscale":
        return magnitude_scale(X)
    if mode == "rotate":
        return small_rotation(X)
    if mode == "combined":
        if random.random() < 0.6:
            X = time_warp(X)
        if random.random() < 0.6:
            X = magnitude_scale(X)
        if random.random() < 0.6:
            X = small_rotation(X)
        return X
    return X

class StudyDataset(Dataset):
    """Dataset class handling data loading, dynamic augmentation, and scaling."""
    def __init__(self, samples, label_to_idx, cols, scaler, target_len=TARGET_LEN, aug_mode=None, is_train=False):
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.cols = cols
        self.scaler = scaler
        self.target_len = target_len
        self.aug_mode = aug_mode
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        df = pd.read_csv(s['path']).values
        if self.aug_mode and self.is_train:
            raw6 = df[:, :6]
            raw6 = apply_aug_mode_raw(raw6, self.aug_mode)
            raw6 = pad_or_trim(raw6, self.target_len)
            full = compute_derived_channels(raw6)
            X = full[:, self.cols]
        else:
            X = df[:, self.cols]
            X = pad_or_trim(X, self.target_len)
        if self.scaler is not None:
            mean = self.scaler.mean_[self.cols]
            scale = self.scaler.scale_[self.cols]
            X = (X - mean) / (scale + 1e-12)
        y = self.label_to_idx[s['label']]
        return torch.tensor(X).float(), torch.tensor(y).long()

def collate_batch(batch):
    """Collates a list of samples into a batch of tensors."""
    X, y = zip(*batch)
    return torch.stack(X), torch.tensor(y)

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Trains the model for one epoch and returns the average loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += X.size(0)
    return total_loss/total, correct/total

def eval_model_return_all(model, loader, criterion, device):
    """Evaluates the model and returns loss, accuracy, and all predictions."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_y, all_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            pred = out.argmax(dim=1)
            total_loss += loss.item() * X.size(0)
            correct += (pred == y).sum().item()
            total += X.size(0)
            all_y.extend(y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
    return total_loss/total, correct/total, np.array(all_y), np.array(all_pred)

def fit_and_train(model, train_loader, val_loader, n_epochs=NUM_EPOCHS, patience=PATIENCE, save_path=None):
    """Manages the full training loop including early stopping and checkpointing."""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val = 0.0
    patience_ctr = 0
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_state = None

    for epoch in range(1, n_epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_acc, _, _ = eval_model_return_all(model, val_loader, criterion, DEVICE)
        scheduler.step(v_loss)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        dt = time.time() - t0
        log(f"Epoch {epoch}/{n_epochs} - train_loss {tr_loss:.4f} acc {tr_acc:.4f} | val_loss {v_loss:.4f} acc {v_acc:.4f} (epoch time {dt:.1f}s)")

        if v_acc > best_val:
            best_val = v_acc
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            patience_ctr = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val

def prepare_splits(seed=SEED):
    """Splits the dataset into training, validation, and test sets by gesture."""
    gestures = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    files_by_gesture = {
        g: sorted([os.path.join(DATA_DIR, g, f) for f in os.listdir(os.path.join(DATA_DIR,g)) if f.endswith('.csv')])
        for g in gestures
    }
    train_samples, val_samples, test_samples = [], [], []
    from sklearn.model_selection import train_test_split
    for g in gestures:
        all_files = files_by_gesture[g]
        test_files = all_files[-50:]
        trainval = all_files[:-50]
        train_files, val_files = train_test_split(trainval, test_size=0.1, random_state=seed)
        train_samples += [{'label': g, 'path': p} for p in train_files]
        val_samples   += [{'label': g, 'path': p} for p in val_files]
        test_samples  += [{'label': g, 'path': p} for p in test_files]
    label_to_idx = {g:i for i,g in enumerate(gestures)}
    return train_samples, val_samples, test_samples, label_to_idx

def run_base_training(train_samples, val_samples, test_samples, label_to_idx):
    """Executes the base training procedure using all features and saves results."""
    log("Starting BASE training", True)
    t0 = time.time()
    feat_list = []
    for s in train_samples:
        arr = pd.read_csv(s['path']).values
        feat_list.append(arr)
    all_train = np.vstack(feat_list)
    scaler = StandardScaler().fit(all_train)
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.joblib"))
    log("Scaler fitted and saved.", True)

    train_ds = StudyDataset(train_samples, label_to_idx, ALL_FEATURES, scaler, target_len=TARGET_LEN, aug_mode=None, is_train=True)
    val_ds   = StudyDataset(val_samples, label_to_idx, ALL_FEATURES, scaler, target_len=TARGET_LEN, aug_mode=None, is_train=False)
    test_ds  = StudyDataset(test_samples, label_to_idx, ALL_FEATURES, scaler, target_len=TARGET_LEN, aug_mode=None, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch, num_workers=NUM_WORKERS)

    model = Robust1DCNN(in_channels=MAX_FEATURES, n_classes=len(label_to_idx))
    save_path = os.path.join(BASE_DIR, "best_model.pth")
    model, history, best_val = fit_and_train(model, train_loader, val_loader, n_epochs=NUM_EPOCHS, patience=PATIENCE, save_path=save_path)
    duration = time.time() - t0
    log(f"Base training finished in {duration/60:.2f} minutes. Best val acc: {best_val:.4f}")

    criterion = nn.CrossEntropyLoss()
    t_loss, t_acc, t_y, t_pred = eval_model_return_all(model, test_loader, criterion, DEVICE)
    log(f"Base Test acc: {t_acc:.4f}", True)

    cls_txt = os.path.join(BASE_DIR, "classification_report.txt")
    with open(cls_txt, "w") as f:
        f.write(classification_report(t_y, t_pred, target_names=list(label_to_idx.keys())))
    cm = confusion_matrix(t_y, t_pred)
    pd.DataFrame(cm, index=list(label_to_idx.keys()), columns=list(label_to_idx.keys())).to_csv(os.path.join(BASE_DIR, "confusion_matrix.csv"))

    plt.figure(figsize=(8,5))
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Base Training Accuracy")
    plt.savefig(os.path.join(BASE_DIR, "training_accuracy.png"))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Base Training Loss")
    plt.savefig(os.path.join(BASE_DIR, "training_loss.png"))
    plt.close()

    pd.DataFrame({"true": t_y, "pred": t_pred}).to_csv(os.path.join(BASE_DIR, "test_predictions.csv"), index=False)

    return scaler, label_to_idx, (t_y, t_pred), duration

def study_feature_incremental(train_samples, val_samples, test_samples, label_to_idx, scaler):
    """Evaluates model performance by incrementally adding features."""
    log("Starting Feature Incremental Study", True)
    results = []
    t0_all = time.time()
    for nf in range(1, MAX_FEATURES+1):
        t0 = time.time()
        log(f"Feature Study: training with first {nf} features", True)
        cols = list(range(nf))
        train_ds = StudyDataset(train_samples, label_to_idx, cols, scaler, target_len=TARGET_LEN, aug_mode=None, is_train=True)
        val_ds   = StudyDataset(val_samples, label_to_idx, cols, scaler, target_len=TARGET_LEN, aug_mode=None, is_train=False)
        test_ds  = StudyDataset(test_samples, label_to_idx, cols, scaler, target_len=TARGET_LEN, aug_mode=None, is_train=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
        test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

        model = Robust1DCNN(in_channels=nf, n_classes=len(label_to_idx))
        model, history, best_val = fit_and_train(model, train_loader, val_loader, n_epochs=NUM_EPOCHS, patience=PATIENCE)
        criterion = nn.CrossEntropyLoss()
        _, test_acc, ty, tp = eval_model_return_all(model, test_loader, criterion, DEVICE)
        duration = time.time() - t0
        log(f" -> nf={nf}: best val {best_val:.4f}, test {test_acc:.4f}, time {duration:.1f}s")
        results.append({"num_features": nf, "val_acc": best_val, "test_acc": test_acc, "time_s": duration})

    df = pd.DataFrame(results)
    csvp = os.path.join(FEATURE_DIR, "feature_incremental_results.csv")
    df.to_csv(csvp, index=False)
    plt.figure(figsize=(8,5))
    plt.plot(df["num_features"], df["val_acc"], marker="o", label="val")
    plt.plot(df["num_features"], df["test_acc"], marker="o", label="test")
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.title("Feature Incremental Study")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FEATURE_DIR, "feature_incremental_plot.png"))
    plt.close()
    total_time = time.time() - t0_all
    log(f"Feature incremental study done in {total_time/60:.2f} minutes")
    return df

def study_seq_len(train_samples, val_samples, test_samples, label_to_idx, scaler):
    """Evaluates model performance across different input sequence lengths."""
    log("Starting Sequence Length Study", True)
    rows = []
    t0_all = time.time()
    for L in SEQ_LEN_LIST:
        t0 = time.time()
        log(f"SeqLen Study: TRAIN with target_len={L}", True)
        cols = ALL_FEATURES
        train_ds = StudyDataset(train_samples, label_to_idx, cols, scaler, target_len=L, aug_mode=None, is_train=True)
        val_ds   = StudyDataset(val_samples, label_to_idx, cols, scaler, target_len=L, aug_mode=None, is_train=False)
        test_ds  = StudyDataset(test_samples, label_to_idx, cols, scaler, target_len=L, aug_mode=None, is_train=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
        test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

        model = Robust1DCNN(in_channels=MAX_FEATURES, n_classes=len(label_to_idx))
        model, history, best_val = fit_and_train(model, train_loader, val_loader, n_epochs=NUM_EPOCHS, patience=PATIENCE)
        criterion = nn.CrossEntropyLoss()
        _, test_acc, ty, tp = eval_model_return_all(model, test_loader, criterion, DEVICE)
        duration = time.time() - t0
        log(f" -> L={L}: best val {best_val:.4f}, test {test_acc:.4f}, time {duration:.1f}s")
        rows.append({"target_len": L, "val_acc": best_val, "test_acc": test_acc, "time_s": duration})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SEQ_DIR, "seq_len_results.csv"), index=False)
    plt.figure(figsize=(8,5))
    plt.plot(df["target_len"], df["val_acc"], marker="o", label="val")
    plt.plot(df["target_len"], df["test_acc"], marker="o", label="test")
    plt.xlabel("Sequence length (samples)")
    plt.ylabel("Accuracy")
    plt.title("Sequence Length vs Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SEQ_DIR, "seq_len_plot.png"))
    plt.close()
    total_time = time.time() - t0_all
    log(f"Sequence length study done in {total_time/60:.2f} minutes")
    return df

def study_sensor_ablation(train_samples, val_samples, test_samples, label_to_idx, scaler):
    """Evaluates model performance by ablating different sensor groups."""
    log("Starting Sensor Ablation Study", True)
    groups = {
        "accel": ACC_IND,
        "gyro": GYR_IND,
        "derived": DER_IND
    }
    combos = []
    for r in range(1, len(groups)+1):
        for comb in itertools.combinations(groups.keys(), r):
            cols = [c for g in comb for c in groups[g]]
            combos.append(("+".join(comb), cols))
    rows = []
    t0_all = time.time()
    for name, cols in combos:
        t0 = time.time()
        log(f"Sensor Ablation: {name} -> {len(cols)} features", True)
        train_ds = StudyDataset(train_samples, label_to_idx, cols, scaler, target_len=TARGET_LEN, aug_mode=None, is_train=True)
        val_ds   = StudyDataset(val_samples, label_to_idx, cols, scaler, target_len=TARGET_LEN, aug_mode=None, is_train=False)
        test_ds  = StudyDataset(test_samples, label_to_idx, cols, scaler, target_len=TARGET_LEN, aug_mode=None, is_train=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
        test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

        model = Robust1DCNN(in_channels=len(cols), n_classes=len(label_to_idx))
        model, history, best_val = fit_and_train(model, train_loader, val_loader, n_epochs=NUM_EPOCHS, patience=PATIENCE)
        criterion = nn.CrossEntropyLoss()
        _, test_acc, ty, tp = eval_model_return_all(model, test_loader, criterion, DEVICE)
        duration = time.time() - t0
        log(f" -> {name}: best val {best_val:.4f}, test {test_acc:.4f}, time {duration:.1f}s")
        rows.append({"combination": name, "num_features": len(cols), "val_acc": best_val, "test_acc": test_acc, "time_s": duration})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SENSOR_DIR, "sensor_ablation.csv"), index=False)
    plt.figure(figsize=(10,6))
    df_plot = df.copy()
    df_plot = df_plot.sort_values("num_features")
    x = np.arange(len(df_plot))
    plt.bar(x - 0.15, df_plot["val_acc"], width=0.3, label="val")
    plt.bar(x + 0.15, df_plot["test_acc"], width=0.3, label="test")
    plt.xticks(x, df_plot["combination"], rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Sensor Ablation Study")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SENSOR_DIR, "sensor_ablation_plot.png"))
    plt.close()
    total_time = time.time() - t0_all
    log(f"Sensor ablation study done in {total_time/60:.2f} minutes")
    return df

def study_augmentations(train_samples, val_samples, test_samples, label_to_idx, scaler):
    """Evaluates model performance under different data augmentation strategies."""
    log("Starting Augmentation Study", True)
    rows = []
    t0_all = time.time()
    cols = ALL_FEATURES
    for mode in AUG_MODES:
        t0 = time.time()
        log(f"Augmentation mode: {mode}", True)
        train_ds = StudyDataset(train_samples, label_to_idx, cols, scaler, target_len=TARGET_LEN, aug_mode=mode if mode!="none" else None, is_train=True)
        val_ds   = StudyDataset(val_samples, label_to_idx, cols, scaler, target_len=TARGET_LEN, aug_mode=None, is_train=False)
        test_ds  = StudyDataset(test_samples, label_to_idx, cols, scaler, target_len=TARGET_LEN, aug_mode=None, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
        test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

        model = Robust1DCNN(in_channels=MAX_FEATURES, n_classes=len(label_to_idx))
        model, history, best_val = fit_and_train(model, train_loader, val_loader, n_epochs=NUM_EPOCHS, patience=PATIENCE)
        criterion = nn.CrossEntropyLoss()
        _, test_acc, ty, tp = eval_model_return_all(model, test_loader, criterion, DEVICE)
        duration = time.time() - t0
        log(f" -> aug={mode}: best val {best_val:.4f}, test {test_acc:.4f}, time {duration:.1f}s")
        rows.append({"augmentation": mode, "val_acc": best_val, "test_acc": test_acc, "time_s": duration})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AUG_DIR, "augmentation_results.csv"), index=False)
    plt.figure(figsize=(8,5))
    x = np.arange(len(df))
    plt.bar(x - 0.15, df["val_acc"], width=0.3, label="val")
    plt.bar(x + 0.15, df["test_acc"], width=0.3, label="test")
    plt.xticks(x, df["augmentation"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Augmentation Study")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(AUG_DIR, "augmentation_plot.png"))
    plt.close()
    total_time = time.time() - t0_all
    log(f"Augmentation study done in {total_time/60:.2f} minutes")
    return df

def run_all():
    """Orchestrates the entire study: preparation, base training, and all sub-studies."""
    if os.path.exists(SUMMARY_LOG):
        os.remove(SUMMARY_LOG)
    log("=== Comprehensive Study Report Started ===", True)
    global_start = time.time()

    train_samples, val_samples, test_samples, label_to_idx = prepare_splits(SEED)
    log(f"Data splits: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} classes={len(label_to_idx)}", True)

    scaler, label_map, base_preds, base_time = None, None, None, None
    scaler, label_map, base_preds, base_time = run_base_wrapper(train_samples, val_samples, test_samples, label_to_idx)

    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))

    df_feat = study_feature_incremental(train_samples, val_samples, test_samples, label_to_idx, scaler)
    df_seq  = study_seq_len(train_samples, val_samples, test_samples, label_to_idx, scaler)
    df_sens = study_sensor_ablation(train_samples, val_samples, test_samples, label_to_idx, scaler)
    df_aug  = study_augmentations(train_samples, val_samples, test_samples, label_to_idx, scaler)

    global_duration = time.time() - global_start
    log(f"=== All studies completed in {global_duration/60:.2f} minutes ===", True)
    df_feat.to_csv(os.path.join(OUT_ROOT, "feature_summary.csv"), index=False)
    df_seq.to_csv(os.path.join(OUT_ROOT, "seq_len_summary.csv"), index=False)
    df_sens.to_csv(os.path.join(OUT_ROOT, "sensor_summary.csv"), index=False)
    df_aug.to_csv(os.path.join(OUT_ROOT, "augmentation_summary.csv"), index=False)
    log("Summary CSVs saved.", True)

def run_base_wrapper(train_samples, val_samples, test_samples, label_to_idx):
    """Wraps the base training run to return metadata."""
    t0 = time.time()
    scaler, label_map, preds, duration = None, None, None, None
    scaler, label_map, preds, duration = _run_base(train_samples, val_samples, test_samples, label_to_idx)
    return scaler, label_map, preds, duration

def _run_base(train_samples, val_samples, test_samples, label_to_idx):
    """Internal helper to execute base training."""
    start = time.time()
    scaler, label_map, (t_y, t_pred), duration = run_base_training(train_samples, val_samples, test_samples, label_to_idx)
    return scaler, label_map, (t_y, t_pred), duration

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"DATA_DIR '{DATA_DIR}' not found. Place preprocessed CSVs under this directory.")
    run_all()