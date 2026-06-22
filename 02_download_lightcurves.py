"""
STAGE 2 - Bulk download light curves (run after Stage 1, slow - budget 1-3 hrs
for a few hundred stars depending on your connection, needs internet)
=========================================================================
For each labeled star in the catalog, downloads its Kepler light curve,
cleans it (Paper 1 methodology: remove flagged cadences, normalize),
phase-folds it at the known period, and saves the global + local views
as numpy arrays ready for the CNN.

This script is RESUMABLE - if it crashes or you stop it, just rerun it.
It skips stars that are already processed.

WHAT YOU GET:
  data/processed/<kepid>_<kepoi_name>.npz
    Each file contains: global_view (201,), local_view (61,),
    label (str), period, duration, depth, snr

RUN:
  python 02_download_lightcurves.py --limit 300
  (start with --limit 50 to test quickly, then increase)
"""

import os
import argparse
import numpy as np
import pandas as pd
import lightkurve as lk
import warnings
warnings.filterwarnings("ignore")  # lightkurve is chatty about minor FITS warnings

GLOBAL_BINS = 201   # full-orbit phase-folded resolution
LOCAL_BINS = 61      # zoomed-in transit window resolution
LOCAL_WINDOW_FRAC = 0.08  # local view spans +-8% of the phase around transit


def process_one_star(row, out_dir):
    kepid = int(row["kepid"])
    kepoi_name = row["kepoi_name"]
    period = float(row["koi_period"])
    duration_hrs = float(row["koi_duration"])
    label = row["label"]

    out_path = os.path.join(out_dir, f"{kepid}_{kepoi_name}.npz")
    if os.path.exists(out_path):
        return "skipped"  # already processed - resumability

    try:
        search = lk.search_lightcurve(
            f"KIC {kepid}", mission="Kepler", cadence="long"
        )
        if len(search) == 0:
            return "no_data"

        # Use the first available quarter - good enough for a training example.
        # For production you'd stitch all quarters together.
        lc = search[0].download()
        if lc is None:
            return "download_failed"

        # --- Paper 1 methodology: quality masking + normalization ---
        lc = lc.remove_nans()
        lc = lc[lc.quality == 0]          # drop flagged cadences (Paper 1 Sec 4.1)
        lc = lc.normalize()                # normalize by mean flux

        if len(lc) < 100:
            return "too_short"

        # --- Phase fold at the known period ---
        t0 = lc.time.value[0]  # approximate epoch - good enough for training data
        folded = lc.fold(period=period, epoch_time=t0)

        # --- Global view: bin the full folded curve into GLOBAL_BINS ---
        global_lc = folded.bin(bins=GLOBAL_BINS)
        global_view = np.nan_to_num(global_lc.flux.value, nan=1.0)
        if len(global_view) != GLOBAL_BINS:
            global_view = np.interp(
                np.linspace(0, 1, GLOBAL_BINS),
                np.linspace(0, 1, len(global_view)),
                global_view,
            )

        # --- Local view: zoom into the transit window around phase 0 ---
        phase = folded.phase.value
        mask = np.abs(phase) < LOCAL_WINDOW_FRAC
        if mask.sum() < 10:
            return "transit_window_too_sparse"

        local_flux = np.nan_to_num(folded.flux.value[mask], nan=1.0)
        local_view = np.interp(
            np.linspace(0, 1, LOCAL_BINS),
            np.linspace(0, 1, len(local_flux)),
            local_flux,
        )

        np.savez(
            out_path,
            global_view=global_view.astype(np.float32),
            local_view=local_view.astype(np.float32),
            label=label,
            period=period,
            duration_hrs=duration_hrs,
            depth_ppm=float(row.get("koi_depth", np.nan)),
            snr=float(row.get("koi_model_snr", np.nan)),
        )
        return "success"

    except Exception as e:
        return f"error: {str(e)[:80]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50,
                         help="Max number of NEW stars to process this run")
    parser.add_argument("--catalog", default="data/koi_catalog_clean.csv")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.catalog)

    # Balance classes a bit - don't let one class dominate the queue
    planets = df[df["label"] == "planet_transit"].sample(frac=1, random_state=42)
    false_pos = df[df["label"] == "false_positive"].sample(frac=1, random_state=42)
    interleaved = pd.concat(
        [planets, false_pos]
    ).sort_index(kind="stable")  # roughly interleaves the shuffled groups

    results = {"success": 0, "skipped": 0, "failed": 0}
    processed_this_run = 0

    for idx, row in interleaved.iterrows():
        if processed_this_run >= args.limit:
            break

        status = process_one_star(row, args.out_dir)

        if status == "success":
            results["success"] += 1
            processed_this_run += 1
            print(f"[{processed_this_run}/{args.limit}] OK  {row['kepoi_name']} ({row['label']})")
        elif status == "skipped":
            results["skipped"] += 1
            continue  # don't count toward limit - free resume
        else:
            results["failed"] += 1
            processed_this_run += 1
            print(f"[{processed_this_run}/{args.limit}] FAIL {row['kepoi_name']}: {status}")

    print("\n--- Run summary ---")
    print(f"Newly downloaded: {results['success']}")
    print(f"Already done (skipped): {results['skipped']}")
    print(f"Failed: {results['failed']}")
    print(f"\nTotal processed files now in {args.out_dir}: "
          f"{len([f for f in os.listdir(args.out_dir) if f.endswith('.npz')])}")
    print("\nRerun this script with a higher --limit to get more data.")
    print("Once you have 300+ examples per class, move to 03_train_classifier.py")


if __name__ == "__main__":
    main()
