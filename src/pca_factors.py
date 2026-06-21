"""
Stage 3 — PCA factors.

Classify the 43 sub-indices into high- vs. low-inertia groups based on the
post-2008 average alpha from Stage 2 (cross-sectional mean as the cutoff).
Extract the first principal component from each group, on standardised
inflation rates, to get:

  F_iner  = PC1 of high-inertia sub-indices  (backward-looking pricing)
  F_imp   = PC1 of low-inertia sub-indices   (market-determined pricing)

Sign-normalised so both are positively correlated with headline inflation.
Adds both as columns to data/master.parquet.

Also produces the factor-trajectory and scatter plots.
"""
import json
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, FIG_DIR


# Sectors whose Stage-1 fits blew up (sigma2_eta explosion or persistent
# negative alpha) — exclude before grouping.
EXCLUDE = ["TP_FG_J126"]


def run():
    master = pl.read_parquet(DATA_DIR / "master.parquet")
    with open(DATA_DIR / "kalman_results.json") as f:
        results = json.load(f)

    avg_alphas = {
        lab: results[lab]["avg_alpha_post2008"]
        for lab in results if not results[lab].get("failed")
    }
    mean_alpha = float(np.mean(list(avg_alphas.values())))

    high_inertia = sorted(lab for lab, a in avg_alphas.items()
                          if a >= mean_alpha and lab not in EXCLUDE)
    low_inertia = sorted(lab for lab, a in avg_alphas.items()
                         if a < mean_alpha and lab not in EXCLUDE)
    print(f"High-inertia: {len(high_inertia)} sectors  |  Low-inertia: {len(low_inertia)} sectors")
    print(f"Cross-sectional mean alpha (post-2008): {mean_alpha:.4f}")

    high_cols = [f"inf_{c}" for c in high_inertia]
    low_cols = [f"inf_{c}" for c in low_inertia]
    df_pca = master.select(["date"] + high_cols + low_cols).drop_nulls()
    dates = df_pca["date"].to_list()
    X_high = df_pca.select(high_cols).to_numpy()
    X_low = df_pca.select(low_cols).to_numpy()
    print(f"PCA sample: {df_pca.shape[0]} months ({dates[0]} to {dates[-1]})")

    # Standardise before PCA — sub-indices have wildly different volatilities
    X_high_std = StandardScaler().fit_transform(X_high)
    X_low_std = StandardScaler().fit_transform(X_low)
    pca_high = PCA(n_components=5).fit(X_high_std)
    pca_low = PCA(n_components=5).fit(X_low_std)
    F_iner = pca_high.transform(X_high_std)[:, 0]
    F_imp = pca_low.transform(X_low_std)[:, 0]
    loadings_high = pca_high.components_[0]
    loadings_low = pca_low.components_[0]

    # Sign normalisation: enforce positive correlation with headline inflation
    headline = master.filter(pl.col("date").is_in(dates))["inf_TP_FG_J0"].to_numpy()
    if np.corrcoef(F_iner, headline)[0, 1] < 0:
        F_iner = -F_iner; loadings_high = -loadings_high
    if np.corrcoef(F_imp, headline)[0, 1] < 0:
        F_imp = -F_imp; loadings_low = -loadings_low

    # Persist factors back into master
    df_factors = pl.DataFrame({"date": dates, "F_iner": F_iner.tolist(), "F_imp": F_imp.tolist()})
    master = pl.read_parquet(DATA_DIR / "master.parquet").join(df_factors, on="date", how="left")
    master.write_parquet(DATA_DIR / "master.parquet")

    # Save metadata
    pca_meta = {
        "high_inertia_group": high_inertia,
        "low_inertia_group": low_inertia,
        "mean_alpha": mean_alpha,
        "excluded": EXCLUDE,
        "pca_sample_start": str(dates[0]),
        "pca_sample_end": str(dates[-1]),
        "pca_sample_nobs": len(dates),
        "high_variance_explained": pca_high.explained_variance_ratio_.tolist(),
        "low_variance_explained": pca_low.explained_variance_ratio_.tolist(),
        "high_pc1_loadings": {c: float(l) for c, l in zip(high_inertia, loadings_high)},
        "low_pc1_loadings": {c: float(l) for c, l in zip(low_inertia, loadings_low)},
        "corr_iner_headline": float(np.corrcoef(F_iner, headline)[0, 1]),
        "corr_imp_headline": float(np.corrcoef(F_imp, headline)[0, 1]),
        "corr_iner_imp": float(np.corrcoef(F_iner, F_imp)[0, 1]),
    }
    with open(DATA_DIR / "pca_results.json", "w") as f:
        json.dump(pca_meta, f, indent=2)

    _make_plots(dates, F_iner, F_imp, headline,
                pca_high.explained_variance_ratio_[0],
                pca_low.explained_variance_ratio_[0])
    _make_alpha_plots(high_inertia, low_inertia, mean_alpha, avg_alphas)
    _make_weighted_plots(high_inertia, low_inertia, mean_alpha, avg_alphas)
    print(f"Stage 3 complete: PCA factors merged into master.parquet")


def _make_plots(dates, F_iner, F_imp, headline, var_high, var_low):
    dates_dt = [datetime(d.year, d.month, d.day) for d in dates]
    window = 12
    sm_dates = dates_dt[window - 1:]
    sm_iner = np.convolve(F_iner, np.ones(window) / window, mode="valid")
    sm_imp = np.convolve(F_imp, np.ones(window) / window, mode="valid")
    passive_start = datetime(2009, 2, 1)
    passive_end = datetime(2018, 5, 1)

    # ----- 3-panel time-series figure -----
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    for ax, series, sm, color, label, title in [
        (axes[0], F_iner, sm_iner, "#A32D2D", "F_iner (12m MA)",
         f"Inertial factor (PC1 high-inertia, {var_high*100:.1f}% variance)"),
        (axes[1], F_imp, sm_imp, "#185FA5", "F_imp (12m MA)",
         f"Imported-shock factor (PC1 low-inertia, {var_low*100:.1f}% variance)"),
    ]:
        ax.plot(dates_dt, series, color=color, linewidth=1, alpha=0.5)
        ax.plot(sm_dates, sm, color=color, linewidth=2.5, label=label)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvspan(passive_start, passive_end, color="green", alpha=0.05)
        ax.set_ylabel(label.split()[0]); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.2)
    ax = axes[2]
    headline_std = (headline - headline.mean()) / headline.std() * F_iner.std()
    ax.plot(dates_dt, headline_std, color="#888780", linewidth=0.8, alpha=0.4, label="Headline (rescaled)")
    ax.plot(dates_dt, F_iner, color="#A32D2D", linewidth=1.2, alpha=0.6, label="F_iner")
    ax.plot(dates_dt, F_imp, color="#185FA5", linewidth=1.2, alpha=0.6, label="F_imp")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvspan(passive_start, passive_end, color="green", alpha=0.05)
    ax.set_title("Both factors vs. headline inflation"); ax.legend(ncol=3); ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_factors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- scatter -----
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(F_imp, F_iner, c=range(len(F_iner)), cmap="coolwarm",
               s=15, alpha=0.6, edgecolors="none")
    m, b = np.polyfit(F_imp, F_iner, 1)
    xs = np.linspace(F_imp.min(), F_imp.max(), 100)
    r = np.corrcoef(F_iner, F_imp)[0, 1]
    ax.plot(xs, m * xs + b, "k--", linewidth=1.5, label=f"slope={m:.3f}, r={r:.3f}")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("F_imp (imported-shock factor)")
    ax.set_ylabel("F_iner (inertial factor)")
    ax.set_title("F_iner vs F_imp  (blue=2003, red=2025)")
    ax.legend(); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- excess-inertia plot -----
    fig, ax = plt.subplots(figsize=(18, 6))
    diff = F_iner - F_imp
    sm_diff = np.convolve(diff, np.ones(window) / window, mode="valid")
    ax.fill_between(dates_dt, 0, diff, where=diff > 0, color="#E24B4A", alpha=0.15)
    ax.fill_between(dates_dt, 0, diff, where=diff <= 0, color="#85B7EB", alpha=0.15)
    ax.plot(sm_dates, sm_diff, color="#2C2C2A", linewidth=2.5, label="F_iner - F_imp (12m MA)")
    ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
    ax.axvspan(passive_start, passive_end, color="green", alpha=0.05)
    ax.set_ylabel("F_iner - F_imp")
    ax.set_title("Excess inertial pressure: when red > 0, backward-indexed sectors\n"
                 "inflate faster than market-determined sectors")
    ax.legend(); ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_excess_inertia.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()


# COICOP short labels for the highlight figure
_COICOP_NAMES = {
    "TP_FG_J011": "Food", "TP_FG_J012": "Non-Alc Bev",
    "TP_FG_J021": "Alc Bev", "TP_FG_J022": "Tobacco",
    "TP_FG_J031": "Clothing", "TP_FG_J032": "Footwear",
    "TP_FG_J041": "Rent", "TP_FG_J043": "Maint & Repair",
    "TP_FG_J044": "Water/Misc", "TP_FG_J045": "Elec/Gas/Fuel",
    "TP_FG_J051": "Furniture", "TP_FG_J052": "HH Textiles",
    "TP_FG_J053": "HH Appliances", "TP_FG_J054": "Glassware",
    "TP_FG_J055": "Tools/Equip", "TP_FG_J056": "Routine HH Maint",
    "TP_FG_J061": "Med Products", "TP_FG_J062": "Outpatient",
    "TP_FG_J063": "Hospital", "TP_FG_J071": "Vehicles",
    "TP_FG_J072": "Transport Op", "TP_FG_J073": "Transport Svc",
    "TP_FG_J081": "Postal", "TP_FG_J082": "Phone Equip",
    "TP_FG_J083": "Phone Svc", "TP_FG_J091": "AV Equip",
    "TP_FG_J092": "Rec Durables", "TP_FG_J093": "Rec Items",
    "TP_FG_J094": "Rec Services", "TP_FG_J095": "Books/Stationery",
    "TP_FG_J096": "Package Hol", "TP_FG_J101": "Primary Edu",
    "TP_FG_J102": "Secondary Edu", "TP_FG_J104": "Tertiary Edu",
    "TP_FG_J105": "Other Edu", "TP_FG_J111": "Catering",
    "TP_FG_J112": "Accommodation", "TP_FG_J121": "Personal Care",
    "TP_FG_J123": "Personal Effects", "TP_FG_J124": "Social Prot",
    "TP_FG_J125": "Insurance", "TP_FG_J127": "Other Svc",
}


def _parse_date(d):
    s = str(d)
    return datetime.strptime(s[:10] if len(s) >= 10 else s[:7] + "-01", "%Y-%m-%d")


def _make_alpha_plots(high, low, mean_alpha, avg_alphas):
    """Sectoral alpha-trajectory plots from notebook cell 22."""
    with open(DATA_DIR / "alpha_paths.json") as f:
        alpha_paths = json.load(f)

    passive_start = datetime(2009, 2, 1)
    passive_end = datetime(2018, 5, 1)

    # Align all series to the shortest common length for group averages
    min_len = min(len(alpha_paths[c]["dates"]) for c in high + low)
    ref_dates = [_parse_date(d) for d in alpha_paths[high[0]]["dates"][-min_len:]]
    high_arrays = [np.array(alpha_paths[c]["alpha"][-min_len:]) for c in high]
    low_arrays = [np.array(alpha_paths[c]["alpha"][-min_len:]) for c in low]
    high_avg = np.mean(high_arrays, axis=0)
    low_avg = np.mean(low_arrays, axis=0)

    # ----- FIG: all 42 series, coloured by group -----
    fig, ax = plt.subplots(figsize=(18, 10))
    for code in low:
        dates_i = [_parse_date(d) for d in alpha_paths[code]["dates"]]
        ax.plot(dates_i, alpha_paths[code]["alpha"], color="#85B7EB", alpha=0.35, linewidth=0.8)
    for code in high:
        dates_i = [_parse_date(d) for d in alpha_paths[code]["dates"]]
        ax.plot(dates_i, alpha_paths[code]["alpha"], color="#E24B4A", alpha=0.35, linewidth=0.8)
    ax.plot(ref_dates, high_avg, color="#A32D2D", linewidth=2.5,
            label=f"High-inertia avg (n={len(high)})")
    ax.plot(ref_dates, low_avg, color="#185FA5", linewidth=2.5,
            label=f"Low-inertia avg (n={len(low)})")
    ax.axhline(mean_alpha, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Mean alpha = {mean_alpha:.3f}")
    ax.axvspan(passive_start, passive_end, color="green", alpha=0.05)
    ax.axvline(passive_start, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(passive_end, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Date"); ax.set_ylabel("alpha_{i,t} (backward-indexation weight)")
    ax.set_title("Time-varying backward-indexation weights: all 42 CPI sub-indices\n"
                 "Red = high-inertia group | Blue = low-inertia group")
    ax.legend(loc="upper left"); ax.set_ylim(-1, 3); ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alpha_all_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- FIG: highlight key high vs low sectors -----
    highlight_high = ["TP_FG_J041", "TP_FG_J043", "TP_FG_J111", "TP_FG_J012"]
    highlight_low = ["TP_FG_J071", "TP_FG_J072", "TP_FG_J031", "TP_FG_J123"]
    colors_high = ["#E24B4A", "#D85A30", "#D4537E", "#993556"]
    colors_low = ["#185FA5", "#0F6E56", "#534AB7", "#5F5E5A"]

    fig, ax = plt.subplots(figsize=(18, 10))
    for code in high:
        dates_i = [_parse_date(d) for d in alpha_paths[code]["dates"]]
        ax.plot(dates_i, alpha_paths[code]["alpha"], color="#E24B4A", alpha=0.08, linewidth=0.5)
    for code in low:
        dates_i = [_parse_date(d) for d in alpha_paths[code]["dates"]]
        ax.plot(dates_i, alpha_paths[code]["alpha"], color="#85B7EB", alpha=0.08, linewidth=0.5)
    for code, color in zip(highlight_high, colors_high):
        if code not in alpha_paths:
            continue
        dates_i = [_parse_date(d) for d in alpha_paths[code]["dates"]]
        name = _COICOP_NAMES.get(code, code)
        ax.plot(dates_i, alpha_paths[code]["alpha"], color=color, linewidth=2,
                label=f"{name} (alpha={avg_alphas[code]:.3f})")
    for code, color in zip(highlight_low, colors_low):
        if code not in alpha_paths:
            continue
        dates_i = [_parse_date(d) for d in alpha_paths[code]["dates"]]
        name = _COICOP_NAMES.get(code, code)
        ax.plot(dates_i, alpha_paths[code]["alpha"], color=color, linewidth=2, linestyle="--",
                label=f"{name} (alpha={avg_alphas[code]:.3f})")
    ax.axhline(1.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.4)
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.4)
    ax.axhline(mean_alpha, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvspan(passive_start, passive_end, color="green", alpha=0.05)
    ax.axvline(passive_start, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(passive_end, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Date"); ax.set_ylabel("alpha_{i,t}")
    ax.set_title("Key sectors: backward-indexation dynamics\n"
                 "Solid = high-inertia | Dashed = low-inertia | Gray background = all series")
    ax.legend(loc="upper left", fontsize=9, ncol=2); ax.set_ylim(-1, 3); ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alpha_key_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- FIG: group averages with +/- 1 s.d. bands -----
    high_std = np.std(high_arrays, axis=0)
    low_std = np.std(low_arrays, axis=0)
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.fill_between(ref_dates, high_avg - high_std, high_avg + high_std, color="#E24B4A", alpha=0.12)
    ax.fill_between(ref_dates, low_avg - low_std, low_avg + low_std, color="#85B7EB", alpha=0.12)
    ax.plot(ref_dates, high_avg, color="#A32D2D", linewidth=2.5,
            label=f"High-inertia mean +/- 1 s.d. (n={len(high)})")
    ax.plot(ref_dates, low_avg, color="#185FA5", linewidth=2.5,
            label=f"Low-inertia mean +/- 1 s.d. (n={len(low)})")
    ax.fill_between(ref_dates, low_avg, high_avg, color="#888780", alpha=0.08)
    ax.axvspan(passive_start, passive_end, color="green", alpha=0.05)
    ax.axvline(passive_start, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(passive_end, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Date"); ax.set_ylabel("Group average alpha_{i,t}")
    ax.set_title("High-inertia vs low-inertia group averages with +/- 1 s.d. bands")
    ax.legend(loc="upper left"); ax.set_ylim(-0.1, 0.9); ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alpha_group_averages.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# 3-digit COICOP weights from 2026 TÜİK item-level data, summed into parent
# sub-indices and cross-checked against the 2-digit totals.
# Aggregate target (excl. group 13 insurance/financial 0.62): ~99.4 of basket.
COICOP_WEIGHTS_2026 = {
    # 01 Food & Non-Alc Bev (25.78)
    "TP_FG_J011": 23.50, "TP_FG_J012": 2.28,
    # 02 Alc Bev & Tobacco (2.87)
    "TP_FG_J021": 0.62,  "TP_FG_J022": 2.25,
    # 03 Clothing & Footwear (8.06)
    "TP_FG_J031": 6.91,  "TP_FG_J032": 1.15,
    # 04 Housing (10.59)
    "TP_FG_J041": 7.04,  "TP_FG_J043": 0.36, "TP_FG_J044": 0.72, "TP_FG_J045": 2.47,
    # 05 Furnishings & HH (8.21)
    "TP_FG_J051": 3.16,  "TP_FG_J052": 0.54, "TP_FG_J053": 1.45, "TP_FG_J054": 0.75,
    "TP_FG_J055": 0.23,  "TP_FG_J056": 2.08,
    # 06 Health (3.16)
    "TP_FG_J061": 1.91,  "TP_FG_J062": 1.17, "TP_FG_J063": 0.07,
    # 07 Transport (16.49)
    "TP_FG_J071": 5.37,  "TP_FG_J072": 3.13, "TP_FG_J073": 7.99,
    # 08 Communications (2.43)
    "TP_FG_J081": 0.04,  "TP_FG_J082": 0.74, "TP_FG_J083": 1.32,
    # 09 Recreation (4.28)
    "TP_FG_J091": 0.42,  "TP_FG_J092": 0.25, "TP_FG_J093": 0.87, "TP_FG_J094": 1.29,
    "TP_FG_J095": 0.67,  "TP_FG_J096": 0.78,
    # 10 Education (1.86) - J105 residual
    "TP_FG_J101": 0.48,  "TP_FG_J102": 0.57, "TP_FG_J104": 0.81, "TP_FG_J105": 0.00,
    # 11 Hotels/Restaurants (11.05)
    "TP_FG_J111": 8.24,  "TP_FG_J112": 2.81,
    # 12 Misc (4.60)
    "TP_FG_J121": 2.62,  "TP_FG_J123": 0.86, "TP_FG_J124": 0.47,
    "TP_FG_J125": 0.37,  "TP_FG_J126": 0.17, "TP_FG_J127": 0.93,
}


def _make_weighted_plots(high, low, mean_alpha, avg_alphas):
    """CPI-weighted inertia plots from notebook cell 23."""
    with open(DATA_DIR / "alpha_paths.json") as f:
        alpha_paths = json.load(f)

    # Normalise weights to sum to 100
    total = sum(COICOP_WEIGHTS_2026.values())
    weights_norm = {k: v * 100 / total for k, v in COICOP_WEIGHTS_2026.items()}

    # Sectors with both a weight and an alpha path, excluding the exploding J126
    valid_codes = [c for c in alpha_paths
                   if c in weights_norm and c not in EXCLUDE
                   and weights_norm[c] > 0]
    min_len = min(len(alpha_paths[c]["dates"]) for c in valid_codes)
    ref_dates = [_parse_date(d) for d in alpha_paths[valid_codes[0]]["dates"][-min_len:]]

    # Weighted aggregate alpha
    total_w = sum(weights_norm[c] for c in valid_codes)
    weighted_agg = np.zeros(min_len)
    for c in valid_codes:
        weighted_agg += weights_norm[c] * np.array(alpha_paths[c]["alpha"][-min_len:])
    weighted_agg /= total_w

    # Group-conditional weighted averages
    high_in = [c for c in valid_codes if c in high]
    low_in = [c for c in valid_codes if c in low]
    w_high_total = sum(weights_norm[c] for c in high_in)
    w_low_total = sum(weights_norm[c] for c in low_in)

    weighted_high = np.zeros(min_len); contrib_high = np.zeros(min_len)
    for c in high_in:
        a = np.array(alpha_paths[c]["alpha"][-min_len:])
        weighted_high += weights_norm[c] * a
        contrib_high += weights_norm[c] * a
    weighted_high /= w_high_total

    weighted_low = np.zeros(min_len); contrib_low = np.zeros(min_len)
    for c in low_in:
        a = np.array(alpha_paths[c]["alpha"][-min_len:])
        weighted_low += weights_norm[c] * a
        contrib_low += weights_norm[c] * a
    weighted_low /= w_low_total

    print(f"  CPI basket: high-inertia {w_high_total:.1f}%, low-inertia {w_low_total:.1f}%")

    passive_start = datetime(2009, 2, 1)
    passive_end = datetime(2018, 5, 1)
    window = 12
    sm_dates = ref_dates[window - 1:]

    # ----- FIG 1: weighted aggregate -----
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(ref_dates, weighted_agg, color="#2C2C2A", linewidth=2.5,
            label="CPI-weighted aggregate inertia")
    if len(weighted_agg) > window:
        smoothed = np.convolve(weighted_agg, np.ones(window) / window, mode="valid")
        ax.plot(sm_dates, smoothed, color="#E24B4A", linewidth=2.5, label="12-month MA")
    ax.axvspan(passive_start, passive_end, color="green", alpha=0.05)
    ax.axvline(passive_start, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(passive_end, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Weighted avg alpha")
    ax.set_title("CPI-weighted aggregate backward-indexation intensity")
    ax.legend(); ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weighted_aggregate_inertia.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- FIG 2: weighted high vs low -----
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(ref_dates, weighted_high, color="#E24B4A", alpha=0.3, linewidth=0.8)
    ax.plot(ref_dates, weighted_low, color="#185FA5", alpha=0.3, linewidth=0.8)
    if len(weighted_high) > window:
        sm_h = np.convolve(weighted_high, np.ones(window) / window, mode="valid")
        sm_l = np.convolve(weighted_low, np.ones(window) / window, mode="valid")
        ax.plot(sm_dates, sm_h, color="#A32D2D", linewidth=2.5,
                label=f"High-inertia weighted avg ({w_high_total:.1f}% basket)")
        ax.plot(sm_dates, sm_l, color="#185FA5", linewidth=2.5,
                label=f"Low-inertia weighted avg ({w_low_total:.1f}% basket)")
        ax.fill_between(sm_dates, sm_l, sm_h, color="#888780", alpha=0.08)
    ax.axvspan(passive_start, passive_end, color="green", alpha=0.05)
    ax.axvline(passive_start, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(passive_end, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("CPI-weighted group avg alpha")
    ax.set_title("High-inertia vs low-inertia (CPI-weighted, 12-month MA)")
    ax.legend(); ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weighted_high_low.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- FIG 3: stacked contributions -----
    fig, ax = plt.subplots(figsize=(18, 7))
    if len(contrib_high) > window:
        sm_ch = np.convolve(contrib_high, np.ones(window) / window, mode="valid")
        sm_cl = np.convolve(contrib_low, np.ones(window) / window, mode="valid")
        ax.fill_between(sm_dates, 0, sm_ch, color="#E24B4A", alpha=0.4,
                        label=f"High-inertia contribution ({w_high_total:.1f}% basket)")
        ax.fill_between(sm_dates, sm_ch, sm_ch + sm_cl, color="#85B7EB", alpha=0.4,
                        label=f"Low-inertia contribution ({w_low_total:.1f}% basket)")
        ax.plot(sm_dates, sm_ch + sm_cl, color="#2C2C2A", linewidth=1.5,
                label="Total weighted inertia")
    ax.axvspan(passive_start, passive_end, color="green", alpha=0.05)
    ax.axvline(passive_start, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(passive_end, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("sum(w_i * alpha_{i,t})")
    ax.set_title("Decomposition of CPI-weighted backward-indexation by group (12-month MA)")
    ax.legend(); ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weighted_contribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- FIG 4: bubble scatter weight x avg alpha -----
    fig, ax = plt.subplots(figsize=(14, 10))
    for c in valid_codes:
        w = weights_norm[c]
        a = avg_alphas.get(c, 0)
        is_high = c in high
        color = "#E24B4A" if is_high else "#185FA5"
        ax.scatter(w, a, s=w * 30, c=color, alpha=0.6,
                   edgecolors="white", linewidth=0.5)
        label = _COICOP_NAMES.get(c, c)
        if w > 2.0 or abs(a) > 0.7 or (w > 1.0 and a > 0.5):
            ax.annotate(label, (w, a), fontsize=7, ha="left",
                        xytext=(5, 3), textcoords="offset points", alpha=0.8)
    ax.axhline(mean_alpha, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("CPI basket weight (%)")
    ax.set_ylabel("Average alpha (post-2008)")
    ax.set_title("CPI weight x backward-indexation strength\n"
                 "Red = high-inertia | Blue = low-inertia | Dot size = weight")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weight_vs_alpha_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
