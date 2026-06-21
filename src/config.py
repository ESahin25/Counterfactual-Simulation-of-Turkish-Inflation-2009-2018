"""Central paths and constants for the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Intermediate artefacts (parquet / json / npz) that pipeline stages exchange.
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Final user-facing outputs: PNGs, CSVs, summary JSONs.
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CSV_DIR = RESULTS_DIR / "csv"
FIG_DIR = RESULTS_DIR / "figures"
CSV_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# EVDS API key. Register your own at https://evds2.tcmb.gov.tr/
EVDS_API_KEY = "L7ruf6fi03"

# Sample window
START_DATE = "01-01-2003"
END_DATE = "01-03-2026"
