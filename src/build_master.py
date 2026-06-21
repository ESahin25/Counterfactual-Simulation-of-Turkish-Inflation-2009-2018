"""
Stage 1 — Master dataframe construction and transformations.

Merges the parquet artefacts produced in Stage 0 into a single monthly dataframe
keyed on first-of-month date, then computes:

  pi_{i,t}     = (ln P_{i,t} - ln P_{i,t-1}) x 100      monthly log inflation
  d_usdtry_t   = (ln e_t - ln e_{t-1}) x 100
  d_comm_t     = (ln C_t - ln C_{t-1}) x 100
  y_t          = 100 x cycle component of HP(log IPI), lambda = 14400

Writes data/master.parquet.
"""
import polars as pl
import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter

from .config import DATA_DIR


def _to_monthly_date(df, date_col):
    """Standardise whatever date column to a Date column named 'date' = first of month."""
    col = df[date_col]
    if col.dtype in (pl.Date, pl.Datetime) or str(col.dtype).startswith("Datetime"):
        return df.with_columns(
            pl.col(date_col).cast(pl.Date).dt.truncate("1mo").alias("date")
        ).drop(date_col)
    if col.dtype == pl.Utf8:
        return df.with_columns(
            pl.col(date_col).map_elements(
                lambda s: pd.Timestamp(s + "-01" if len(s) <= 7 else s).strftime("%Y-%m-%d")
                if s is not None else None,
                return_dtype=pl.Utf8,
            ).str.to_date("%Y-%m-%d").alias("date")
        ).drop(date_col)
    raise ValueError(f"Unexpected dtype for {date_col}: {col.dtype}")


def build_master():
    """Merge all stage-0 outputs into a single monthly master dataframe."""
    df_cpi_main = _to_monthly_date(pl.read_parquet(DATA_DIR / "cpi_main.parquet"), "Tarih")

    # J103 (post-secondary non-tertiary education) is discontinued in 2015,
    # J042 (imputed rentals) is sparse — drop both before estimation
    df_cpi_sub = pl.read_parquet(DATA_DIR / "cpi_sub.parquet")
    drop_cols = [c for c in df_cpi_sub.columns if "J103" in c or "J042" in c]
    if drop_cols:
        df_cpi_sub = df_cpi_sub.drop(drop_cols)
    df_cpi_sub = _to_monthly_date(df_cpi_sub, "Tarih")

    df_cpi_special = _to_monthly_date(pl.read_parquet(DATA_DIR / "cpi_special.parquet"), "Tarih")
    df_policy = pl.read_parquet(DATA_DIR / "policy_rate.parquet").rename({"month": "date"})
    df_ipi = _to_monthly_date(pl.read_parquet(DATA_DIR / "ipi.parquet"), "Tarih")
    df_fx = pl.read_parquet(DATA_DIR / "fx.parquet").rename({"TP_DK_USD_A_YTL": "usdtry"})
    df_fx = _to_monthly_date(df_fx, "Tarih")
    df_comm = _to_monthly_date(pl.read_parquet(DATA_DIR / "PALLFNFINDEXM.parquet"), "DATE")
    df_comm = df_comm.rename({"PALLFNFINDEXM": "commodity_index"})

    master = df_cpi_main
    for name, df_right in [
        ("cpi_sub", df_cpi_sub), ("cpi_special", df_cpi_special),
        ("policy_rate", df_policy), ("ipi", df_ipi),
        ("fx", df_fx), ("commodity", df_comm),
    ]:
        dupes = set(master.columns) & set(df_right.columns) - {"date"}
        if dupes:
            df_right = df_right.drop(list(dupes))
        master = master.join(df_right, on="date", how="left")
        print(f"  joined {name}: master now {master.shape}")
    return master


def add_transformations(master):
    """Compute log-difference inflation, FX change, and HP-filtered output gap."""
    # Sub-index columns (3-digit COICOP), excluding headline J0 and 2-digit groups
    cpi_sub_cols = [c for c in master.columns if c.startswith("TP_FG_J") and c != "TP_FG_J0"]
    cpi_headline = "TP_FG_J0"

    # MoM log-difference inflation for headline and all sub-indices
    inflation_exprs = [
        (pl.col(col).log().diff() * 100).alias(f"inf_{col}")
        for col in [cpi_headline] + cpi_sub_cols
    ]
    master = master.with_columns(inflation_exprs)

    # Exchange rate and commodity log-differences
    master = master.with_columns([
        (pl.col("usdtry").log().diff() * 100).alias("d_usdtry"),
        (pl.col("commodity_index").log().diff() * 100).alias("d_commodity"),
    ])

    # HP-filtered output gap on log(IPI), only over the non-null window
    ipi_series = master["ipi"].to_numpy()
    valid_mask = ~np.isnan(ipi_series)
    log_ipi_full = np.full(len(ipi_series), np.nan)
    gap_full = np.full(len(ipi_series), np.nan)

    log_ipi_valid = np.log(ipi_series[valid_mask])
    cycle, _trend = hpfilter(log_ipi_valid, lamb=14400)
    log_ipi_full[valid_mask] = log_ipi_valid
    gap_full[valid_mask] = cycle * 100  # express as percent of trend

    master = master.with_columns([
        pl.Series("log_ipi", log_ipi_full),
        pl.Series("output_gap", gap_full),
    ])

    # Convert NaN -> null (NaN slips through pl.Series construction from numpy)
    for col in ["output_gap", "log_ipi"]:
        master = master.with_columns(
            pl.when(pl.col(col).is_nan()).then(None).otherwise(pl.col(col)).alias(col)
        )
    return master


def run():
    master = build_master()
    master = add_transformations(master)
    out = DATA_DIR / "master.parquet"
    master.write_parquet(out)
    print(f"Stage 1 complete: master.parquet written, shape={master.shape}")
    print(f"  date range: {master['date'].min()} to {master['date'].max()}")


if __name__ == "__main__":
    run()
