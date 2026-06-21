"""
Export raw and processed parquet files to CSV for non-Python users.

Reads everything in data/ that has a corresponding fetch step and writes a CSV
mirror into results/csv/.
"""
import polars as pl

from .config import DATA_DIR, CSV_DIR


EXPORTS = [
    ("cpi_main", "cpi_main.parquet"),
    ("cpi_sub", "cpi_sub.parquet"),
    ("cpi_special", "cpi_special.parquet"),
    ("fx", "fx.parquet"),
    ("import_uvi", "import_uvi.parquet"),
    ("rates", "rates.parquet"),
    ("policy_rate", "policy_rate.parquet"),
    ("ipi", "ipi.parquet"),
    ("commodity_index", "PALLFNFINDEXM.parquet"),
    ("master", "master.parquet"),
    ("regime_probabilities", "regime_probabilities.parquet"),
]


def run():
    for name, fname in EXPORTS:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  skip {name}: {path} not found")
            continue
        df = pl.read_parquet(path)
        df.write_csv(CSV_DIR / f"{name}.csv")
        print(f"  wrote {name}.csv  shape={df.shape}")
    print("Stage 7 complete: CSVs exported.")


if __name__ == "__main__":
    run()
