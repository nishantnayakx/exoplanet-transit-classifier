# PS7 Classify pipeline - setup and run order

This is YOUR part of the team pipeline (the Classify stage). It is fully
automated: run the three scripts in order, on your own machine, with
internet access. Nothing here needs manual clicking on websites.

## One-time setup

```bash
pip install -r requirements.txt
```

If you're on Windows and pip complains about build tools for any package,
install via Anaconda/Miniconda instead - `conda install -c conda-forge
lightkurve astroquery pytorch` is the most reliable path on Windows.

## Run order

### Step 1 - Download labels (~30 seconds)
```bash
python 01_download_catalog.py
```
Downloads the NASA Kepler KOI catalog directly via the official astroquery
library - no manual CSV download needed. Produces `data/koi_catalog_clean.csv`.

### Step 2 - Download light curves (slow - run this overnight or in chunks)
```bash
python 02_download_lightcurves.py --limit 50
```
Start with `--limit 50` to make sure it works, then scale up:
```bash
python 02_download_lightcurves.py --limit 500
```
This script is **resumable** - if your internet drops or you stop it
(Ctrl+C), just run the same command again. It automatically skips stars
it already processed and only downloads new ones up to the limit you set.

Realistic target: 300-500 stars per class (600-1000 total) gives you a
reasonable training set for the hackathon timeline. Each star takes
roughly 5-15 seconds to download and process, so budget 1-3 hours of
mostly-unattended runtime split across a few sessions.

### Step 3 - Train the classifier (~5-15 minutes on a normal laptop CPU)
```bash
python 03_train_classifier.py
```
Trains the dual-view CNN on whatever you've downloaded so far. You can
run this at any point after Step 2 has produced at least ~50-100 files
per class - re-run it again later as you accumulate more data from Step 2.

Produces:
- `models/transit_classifier.pt` - trained weights
- `models/label_encoder.json` - class name <-> index mapping
- Printed train/val accuracy per epoch

## Notes

- All scripts create their own `data/` and `models/` folders automatically.
- If Step 2 reports many `no_data` or `error` results, that's normal -
  not every KIC ID has a usable light curve. The script just moves on.
- Once you have a trained model, the next step is the SHAP explainability
  panel and the Plotly dashboard - ask Claude for "the dashboard code"
  when you get there.
