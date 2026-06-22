"""
STAGE 1 - Download labels (run this first, ~30 seconds, needs internet)
=========================================================================
Downloads the official NASA Kepler Cumulative KOI catalog using astroquery
(the maintained NASA-supported library - more reliable than raw URL queries).

WHAT YOU GET:
  data/koi_catalog_clean.csv
  Columns: kepid, kepoi_name, label, koi_period, koi_duration, koi_depth,
           koi_prad, koi_model_snr, koi_steff, koi_slogg, koi_srad

RUN:
  python 01_download_catalog.py
"""

import os
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

os.makedirs("data", exist_ok=True)

print("Querying NASA Exoplanet Archive (cumulative KOI table)...")
table = NasaExoplanetArchive.query_criteria(
    table="cumulative",
    select=(
        "kepid,kepoi_name,koi_disposition,koi_period,koi_duration,"
        "koi_depth,koi_prad,koi_model_snr,koi_steff,koi_slogg,koi_srad"
    ),
)
df = table.to_pandas()
print(f"Downloaded {len(df)} KOI entries.")
print(df["koi_disposition"].value_counts())

# Map to training labels.
# NOTE: FALSE POSITIVE is a catch-all in this table - it bundles eclipsing
# binaries, background blends, and instrumental artifacts together. Stage 3
# splits these into proper sub-classes using diagnostic features once you
# have light curve data. For now this gives you a clean binary-ish base.
label_map = {
    "CONFIRMED": "planet_transit",
    "FALSE POSITIVE": "false_positive",
    "CANDIDATE": None,  # ambiguous disposition - exclude from training
}
df["label"] = df["koi_disposition"].map(label_map)
df_clean = df.dropna(subset=["label"]).copy()

required_cols = ["kepid", "kepoi_name", "koi_period", "koi_duration", "koi_depth"]
df_clean = df_clean.dropna(subset=required_cols)

out_path = "data/koi_catalog_clean.csv"
df_clean.to_csv(out_path, index=False)

print(f"\nSaved clean labeled dataset -> {out_path}")
print(f"Total usable rows: {len(df_clean)}")
print(df_clean["label"].value_counts())
print("\nNext step: run 02_download_lightcurves.py")
