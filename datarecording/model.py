import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import random
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_DIR = "imu_dataset_preprocessed"
SAVE_DIR = "model_checkpoints"
NUM_EPOCHS = 40
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
TARGET_LEN = 150
SEED = 42
PATIENCE = 7
# ----------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- DATASET ----------------
class IMUDataset(Dataset):
    def __init__(self, samples, label_to_idx, mode='train', augment=False, scaler=None):
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.mode = mode
        self.augment = augment
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

# ---------------- TRAIN/VAL ----------------
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

# ---------------- VISUALIZER ----------------
def visualize_samples(data_dir, n_samples=3):
    gestures = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))]
    for g in gestures:
        folder = os.path.join(data_dir, g)
        files = os.listdir(folder)
        sampled = random.sample(files, min(n_samples,len(files)))
        for f in sampled:
            df = pd.read_csv(os.path.join(folder,f))
            plt.figure(figsize=(10,3))
            for col in ['ax','ay','az','acc_mag']:
                if col in df.columns:
                    plt.plot(df[col], label=col)
            plt.title(f"{g} - {f}")
            plt.legend()
            plt.show()

# ---------------- MAIN ----------------
def main():
    visualize_samples(DATA_DIR, n_samples=2)

    # collect files
    gestures = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))]
    files, labels = [],[]
    for g in gestures:
        folder = os.path.join(DATA_DIR, g)
        fs = [f for f in os.listdir(folder) if f.endswith('.csv')]
        for f in fs:
            files.append(os.path.join(folder,f))
            labels.append(g)
    label_to_idx = {g:i for i,g in enumerate(gestures)}

    # stratified split: 80/10/10
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    idx_train, idx_testval = next(sss1.split(np.zeros(len(labels)), labels))
    testval_labels = [labels[i] for i in idx_testval]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    idx_val, idx_test = next(sss2.split(np.zeros(len(idx_testval)), testval_labels))

    train_samples = [{'label': labels[i], 'path': files[i]} for i in idx_train]
    val_samples = [{'label': labels[idx_testval[i]], 'path': files[idx_testval[i]]} for i in idx_val]
    test_samples = [{'label': labels[idx_testval[i]], 'path': files[idx_testval[i]]} for i in idx_test]

    # fit scaler on training features
    flat_feats = []
    for s in train_samples:
        df = pd.read_csv(s['path'])
        arr = df.values
        flat_feats.append(arr)
    flat_feats = np.vstack(flat_feats).reshape(-1,11)
    scaler = StandardScaler().fit(flat_feats)
    print("Scaler ready.")

    train_dataset = IMUDataset(train_samples, label_to_idx, mode='train', augment=False, scaler=scaler)
    val_dataset = IMUDataset(val_samples, label_to_idx, mode='val', augment=False, scaler=scaler)
    test_dataset = IMUDataset(test_samples, label_to_idx, mode='test', augment=False, scaler=scaler)

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

    # final test
    print("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=device))
    test_loss, test_acc, test_y, test_pred = eval_model(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")
    print("Classification report (test):")
    print(classification_report(test_y, test_pred, target_names=gestures))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(test_y, test_pred))

if __name__ == "__main__":
    main()
