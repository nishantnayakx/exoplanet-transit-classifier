import os
import json

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


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

        g_out = 1600
        l_out = 480

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


_model = None
_label_to_idx = None
_idx_to_label = None


def get_model():
    global _model, _label_to_idx, _idx_to_label
    if _model is None:
        torch.set_num_threads(1)
        with open(os.path.join(MODEL_DIR, "label_encoder.json")) as f:
            _label_to_idx = json.load(f)

        _idx_to_label = {v: k for k, v in _label_to_idx.items()}

        m = TransitClassifier(num_classes=len(_label_to_idx))
        m.load_state_dict(
            torch.load(
                os.path.join(MODEL_DIR, "transit_classifier.pt"),
                map_location="cpu"
            )
        )
        m.eval()
        _model = m
        print("TransitClassifier model loaded successfully in worker process")

    return _model, _idx_to_label


def predict_npz(path):
    model, idx_to_label = get_model()

    with np.load(path, allow_pickle=True) as d:
        g_arr = d["global_view"]
        l_arr = d["local_view"]
        period = float(d["period"])
        duration = float(d["duration_hrs"])
        depth = float(d["depth_ppm"])
        snr = float(d["snr"])

        global_view = torch.tensor(
            g_arr,
            dtype=torch.float32
        ).unsqueeze(0)

        local_view = torch.tensor(
            l_arr,
            dtype=torch.float32
        ).unsqueeze(0)

        scalars = torch.tensor([[
            np.log1p(period) / 5.0,
            duration / 24.0,
            np.log1p(max(depth, 0.0)) / 12.0,
            np.log1p(max(snr, 0.0)) / 8.0
        ]], dtype=torch.float32)

        with torch.no_grad():
            logits = model(
                global_view,
                local_view,
                scalars
            )
            probs = torch.softmax(logits, dim=1)[0]

        pred_idx = probs.argmax().item()

        scientific_score = (
            min(snr / 20.0, 1.0) * 0.4 +
            min(depth / 1000.0, 1.0) * 0.3 +
            min(duration / 10.0, 1.0) * 0.3
        )

        return {
            "prediction": idx_to_label[pred_idx],
            "confidence": float(probs[pred_idx]),
            "scientific_score": scientific_score,

            "period_days": period,
            "duration_hours": duration,
            "depth_ppm": depth,
            "snr": snr,

            "global_view": g_arr.tolist(),
            "local_view": l_arr.tolist()
        }
