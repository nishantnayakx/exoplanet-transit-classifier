import os
import glob
import pandas as pd

from predict import predict_npz

DATA_DIR = "data/processed"

results = []

files = glob.glob(
    os.path.join(DATA_DIR, "*.npz")
)

print(f"Found {len(files)} candidates")

for i, f in enumerate(files):

    try:

        r = predict_npz(f)

        results.append({
            "file": os.path.basename(f),
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "scientific_score": r["scientific_score"],
            "period_days": r["period_days"],
            "duration_hours": r["duration_hours"],
            "depth_ppm": r["depth_ppm"],
            "snr": r["snr"]
        })

        if i % 100 == 0:
            print(f"Processed {i}")

    except Exception as e:
        print(f"Failed: {f}")
        print(e)

df = pd.DataFrame(results)

# Normalize SNR
df["snr_norm"] = (
    (df["snr"] - df["snr"].min()) /
    (df["snr"].max() - df["snr"].min())
)

# Normalize depth
df["depth_norm"] = (
    (df["depth_ppm"] - df["depth_ppm"].min()) /
    (df["depth_ppm"].max() - df["depth_ppm"].min())
)

# Scientific ranking score
df["scientific_score"] = (
    0.90 * df["confidence"] +
    0.07 * df["snr_norm"] +
    0.03 * df["depth_norm"]
)

df = df.sort_values(
    "scientific_score",
    ascending=False
)

df["rank"] = range(
    1,
    len(df) + 1
)

df = df[
    [
        "rank",
        "file",
        "prediction",
        "confidence",
        "scientific_score",
        "period_days",
        "duration_hours",
        "depth_ppm",
        "snr"
    ]
]

df.to_csv(
    "candidate_ranking.csv",
    index=False
)

print("\nTop 10 Candidates\n")

print(
    df.head(10)
)

print(
    "\nSaved candidate_ranking.csv"
)