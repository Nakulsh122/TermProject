import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random

# ---------------- CONFIG ----------------
DATA_DIR = "imu_dataset_preprocessed"  # original dataset
SAVE_DIR = "model_checkpoints"
NUM_EPOCHS = 40
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
TARGET_LEN = 150
SEED = 42
PATIENCE = 7
AUGMENT = True  # toggle augmentation
# ----------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- DATASET ----------------
class IMUDataset(Dataset):
    def __init__(self, samples, label_to_idx, scaler=None):
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.scaler = scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        df = pd.read_csv(s['path'])
        X = df.values.astype(np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        y = self.label_to_idx[s['label']]
        return torch.tensor(X).float(), torch.tensor(y).long()

def collate_batch(batch):
    X, y = zip(*batch)
    X = torch.stack(X)
    y = torch.tensor(y)
    return X, y

# ---------------- MODEL ----------------
class Robust1DCNN(nn.Module):
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
        x = x.permute(0,2,1)  # batch, channels, time
        x = self.net(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

# ---------------- AUGMENTATION ----------------
from scipy.signal import resample

TIME_WARP_MAX = 0.1
MAG_SCALE_MAX = 0.1
ROT_ANGLE_MAX = 5

def time_warp(X, max_frac=TIME_WARP_MAX):
    factor = 1 + random.uniform(-max_frac, max_frac)
    length = int(X.shape[0]*factor)
    return resample(X, length, axis=0)

def magnitude_scale(X, max_scale=MAG_SCALE_MAX):
    scale = 1 + random.uniform(-max_scale, max_scale)
    X[:, :6] *= scale
    return X

def small_rotation(X, max_angle_deg=ROT_ANGLE_MAX):
    angle = np.deg2rad(random.uniform(-max_angle_deg, max_angle_deg))
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle),  np.cos(angle), 0],
                  [0,              0,             1]])
    X[:, :3] = X[:, :3] @ R.T
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

def apply_augmentation(X):
    if random.random() < 0.5:
        X = time_warp(X)
    if random.random() < 0.5:
        X = magnitude_scale(X)
    if random.random() < 0.5:
        X = small_rotation(X)
    X = pad_or_trim(X)
    return X

# ---------------- TRAIN/EVAL ----------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0,0,0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*X.size(0)
        pred = out.argmax(dim=1)
        correct += (pred==y).sum().item()
        total += X.size(0)
    return total_loss/total, correct/total

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0,0,0
    all_y, all_pred = [],[]
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()*X.size(0)
            pred = out.argmax(dim=1)
            correct += (pred==y).sum().item()
            total += X.size(0)
            all_y.extend(y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
    return total_loss/total, correct/total, all_y, all_pred

# ---------------- MAIN ----------------
def main():
    gestures = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))]
    files_by_gesture = {g: sorted([os.path.join(DATA_DIR,g,f) for f in os.listdir(os.path.join(DATA_DIR,g)) if f.endswith('.csv')]) for g in gestures}

    # Split: all users except last (per gesture) -> train/val, last user -> test
    train_samples, val_samples, test_samples = [], [], []
    for g in gestures:
        all_files = files_by_gesture[g]
        test_files = all_files[-50:]  # last user
        trainval_files = all_files[:-50]
        train_files, val_files = train_test_split(trainval_files, test_size=0.1, random_state=SEED)
        
        for f in train_files:
            train_samples.append({'label': g, 'path': f})
        for f in val_files:
            val_samples.append({'label': g, 'path': f})
        for f in test_files:
            test_samples.append({'label': g, 'path': f})

    label_to_idx = {g:i for i,g in enumerate(gestures)}

    # Fit scaler on training data
    flat_feats = []
    for s in train_samples:
        df = pd.read_csv(s['path'])
        arr = df.values
        flat_feats.append(arr)
    flat_feats = np.vstack(flat_feats)
    scaler = StandardScaler().fit(flat_feats)
    print("Scaler ready.")

    # Datasets
    train_dataset = IMUDataset(train_samples, label_to_idx, scaler=scaler)
    val_dataset = IMUDataset(val_samples, label_to_idx, scaler=scaler)
    test_dataset = IMUDataset(test_samples, label_to_idx, scaler=scaler)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Robust1DCNN(in_channels=11, n_classes=len(gestures)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_acc = 0.0
    patience_ctr = 0

    for epoch in range(1, NUM_EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_y, val_pred = eval_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print("Saved best model.")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("Early stopping.")
                break

    # Final test
    print("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=device))
    test_loss, test_acc, test_y, test_pred = eval_model(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")
    
    # Ensure classification_report doesn't fail
    labels = list(range(len(gestures)))
    print("Classification report (test):")
    print(classification_report(test_y, test_pred, labels=labels, target_names=gestures))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(test_y, test_pred, labels=labels))

if __name__ == "__main__":
    main()
