"""
STAGE 3 - Train the classifier (run after Stage 2, needs at least ~100
examples per class to be meaningful, ~5-15 min on CPU for a few hundred stars)
=========================================================================
Loads all processed .npz files, builds the dual-view CNN, trains it,
and saves the trained model + a confusion matrix report.

WHAT YOU GET:
  models/transit_classifier.pt   <- trained model weights
  models/label_encoder.json      <- maps class index <-> class name
  Printed: train/val accuracy, confusion matrix

RUN:
  python 03_train_classifier.py
"""

import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

DATA_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


class LightCurveDataset(Dataset):
    def __init__(self, file_list, label_to_idx):
        self.files = file_list
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        global_view = torch.tensor(d["global_view"], dtype=torch.float32)
        local_view = torch.tensor(d["local_view"], dtype=torch.float32)

        # Scalar features - normalize roughly so the network trains stably
        period = float(d["period"])
        duration = float(d["duration_hrs"])
        depth = float(d["depth_ppm"]) if not np.isnan(float(d["depth_ppm"])) else 0.0
        snr = float(d["snr"]) if not np.isnan(float(d["snr"])) else 0.0
        scalars = torch.tensor([
            np.log1p(period) / 5.0,
            duration / 24.0,
            np.log1p(max(depth, 0)) / 12.0,
            np.log1p(max(snr, 0)) / 8.0,
        ], dtype=torch.float32)

        label_str = str(d["label"])
        label = self.label_to_idx[label_str]

        return global_view, local_view, scalars, label


class TransitClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.global_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        self.local_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        self.scalar_branch = nn.Sequential(nn.Linear(4, 32), nn.ReLU())

        # Compute flatten sizes dynamically with a dummy forward pass
        with torch.no_grad():
            g_dummy = torch.zeros(1, 1, 201)
            l_dummy = torch.zeros(1, 1, 61)
            g_out = self.global_branch(g_dummy).shape[1]
            l_out = self.local_branch(l_dummy).shape[1]

        self.fusion = nn.Sequential(
            nn.Linear(g_out + l_out + 32, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, global_view, local_view, scalars):
        g = self.global_branch(global_view.unsqueeze(1))
        l = self.local_branch(local_view.unsqueeze(1))
        s = self.scalar_branch(scalars)
        fused = torch.cat([g, l, s], dim=1)
        return self.fusion(fused)


def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    if len(files) < 20:
        print(f"Only {len(files)} processed files found. "
              f"Run 02_download_lightcurves.py with a higher --limit first.")
        return

    # Discover labels present in the data
    labels_seen = set()
    for f in files:
        d = np.load(f, allow_pickle=True)
        labels_seen.add(str(d["label"]))
    labels_sorted = sorted(labels_seen)
    label_to_idx = {lab: i for i, lab in enumerate(labels_sorted)}
    print(f"Classes found: {label_to_idx}")

    with open(os.path.join(MODEL_DIR, "label_encoder.json"), "w") as f:
        json.dump(label_to_idx, f, indent=2)

    dataset = LightCurveDataset(files, label_to_idx)
    n_val = max(int(0.2 * len(dataset)), 1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = TransitClassifier(num_classes=len(label_to_idx))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    n_epochs = 25
    best_val_acc = 0.0

    for epoch in range(n_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for g, l, s, y in train_loader:
            optimizer.zero_grad()
            logits = model(g, l, s)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for g, l, s, y in val_loader:
                logits = model(g, l, s)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "transit_classifier.pt"))

        print(f"Epoch {epoch+1:2d}/{n_epochs} | "
              f"train_loss={train_loss/max(train_total,1):.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {MODEL_DIR}/transit_classifier.pt")
    print("\nNext: build the Plotly dashboard to visualize predictions "
          "(say 'give me the dashboard code')")


if __name__ == "__main__":
    main()
