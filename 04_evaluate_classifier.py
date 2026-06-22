import os
import glob
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

import matplotlib.pyplot as plt

DATA_DIR = "data/processed"
MODEL_DIR = "models"
ASSETS_DIR = "assets"

os.makedirs(ASSETS_DIR, exist_ok=True)


class LightCurveDataset(Dataset):
    def __init__(self, file_list, label_to_idx):
        self.files = file_list
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)

        global_view = torch.tensor(
            d["global_view"],
            dtype=torch.float32
        )

        local_view = torch.tensor(
            d["local_view"],
            dtype=torch.float32
        )

        period = float(d["period"])
        duration = float(d["duration_hrs"])

        depth = float(d["depth_ppm"])
        if np.isnan(depth):
            depth = 0.0

        snr = float(d["snr"])
        if np.isnan(snr):
            snr = 0.0

        scalars = torch.tensor([
            np.log1p(period) / 5.0,
            duration / 24.0,
            np.log1p(max(depth, 0)) / 12.0,
            np.log1p(max(snr, 0)) / 8.0,
        ], dtype=torch.float32)

        label = self.label_to_idx[str(d["label"])]

        return global_view, local_view, scalars, label


class TransitClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.global_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten()
        )

        self.local_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten()
        )

        self.scalar_branch = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU()
        )

        with torch.no_grad():
            g_dummy = torch.zeros(1, 1, 201)
            l_dummy = torch.zeros(1, 1, 61)

            g_out = self.global_branch(g_dummy).shape[1]
            l_out = self.local_branch(l_dummy).shape[1]

        self.fusion = nn.Sequential(
            nn.Linear(g_out + l_out + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, global_view, local_view, scalars):
        g = self.global_branch(global_view.unsqueeze(1))
        l = self.local_branch(local_view.unsqueeze(1))
        s = self.scalar_branch(scalars)

        fused = torch.cat([g, l, s], dim=1)

        return self.fusion(fused)


def main():

    files = glob.glob(os.path.join(DATA_DIR, "*.npz"))

    with open(os.path.join(MODEL_DIR, "label_encoder.json")) as f:
        label_to_idx = json.load(f)

    dataset = LightCurveDataset(files, label_to_idx)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    model = TransitClassifier(
        num_classes=len(label_to_idx)
    )

    model.load_state_dict(
        torch.load(
            os.path.join(
                MODEL_DIR,
                "transit_classifier.pt"
            ),
            map_location="cpu"
        )
    )

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for g, l, s, y in loader:

            logits = model(g, l, s)

            probs = torch.softmax(
                logits,
                dim=1
            )

            preds = logits.argmax(1)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

            all_probs.extend(
                probs[:, 1].cpu().numpy()
            )

    print("\nCLASSIFICATION REPORT\n")

    print(
        classification_report(
            all_labels,
            all_preds
        )
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    print("\nCONFUSION MATRIX\n")
    print(cm)

    auc = roc_auc_score(
        all_labels,
        all_probs
    )

    print(f"\nROC AUC = {auc:.4f}")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(
        os.path.join(
            ASSETS_DIR,
            "confusion_matrix.png"
        )
    )
    plt.close()

    fpr, tpr, _ = roc_curve(
        all_labels,
        all_probs
    )

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.title(f"ROC Curve (AUC={auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(
        os.path.join(
            ASSETS_DIR,
            "roc_curve.png"
        )
    )
    plt.close()

    precision, recall, _ = precision_recall_curve(
        all_labels,
        all_probs
    )

    ap = average_precision_score(
        all_labels,
        all_probs
    )

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.title(f"PR Curve (AP={ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(
        os.path.join(
            ASSETS_DIR,
            "pr_curve.png"
        )
    )
    plt.close()

    print("\nSaved:")
    print("assets/confusion_matrix.png")
    print("assets/roc_curve.png")
    print("assets/pr_curve.png")


if __name__ == "__main__":
    main()