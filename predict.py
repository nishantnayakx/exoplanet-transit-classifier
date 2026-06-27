import os
import json
import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = "models"


class TransitClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.global_branch = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(),
        )

        self.local_branch = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(),
        )

        self.scalar_branch = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU()
        )

        with torch.no_grad():
            g = self.global_branch(torch.zeros(1, 1, 201))
            l = self.local_branch(torch.zeros(1, 1, 61))

        g_out = g.shape[1]
        l_out = l.shape[1]

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


with open(os.path.join(MODEL_DIR, "label_encoder.json")) as f:
    label_to_idx = json.load(f)

idx_to_label = {v: k for k, v in label_to_idx.items()}

model = TransitClassifier(num_classes=len(label_to_idx))

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "transit_classifier.pt"),
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully")


def predict_npz(path):

    print("A")
    d = np.load(path, allow_pickle=True)

    print("B")

    global_view = torch.tensor(
        d["global_view"],
        dtype=torch.float32
    ).unsqueeze(0)

    print("C")

    local_view = torch.tensor(
        d["local_view"],
        dtype=torch.float32
    ).unsqueeze(0)

    print("D")

    scalars = torch.tensor([[

        np.log1p(float(d["period"])) / 5.0,

        float(d["duration_hrs"]) / 24.0,

        np.log1p(max(float(d["depth_ppm"]), 0)) / 12.0,

        np.log1p(max(float(d["snr"]), 0)) / 8.0

    ]], dtype=torch.float32)

    print("E")

    with torch.no_grad():

        print("F")

        logits = model(
            global_view,
            local_view,
            scalars
        )

        print("G")

        probs = torch.softmax(logits, dim=1)[0]

        print("H")

    pred_idx = probs.argmax().item()

    print("I")

    return {
        "prediction": idx_to_label[pred_idx],
        "confidence": float(probs[pred_idx]),
        "scientific_score": 0,
        "global_view": d["global_view"].tolist(),
        "local_view": d["local_view"].tolist()
    }