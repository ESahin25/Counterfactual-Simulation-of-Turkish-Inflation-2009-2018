"""
Stage 4 — Markov-Switching Taylor Rule.

Estimate a 2-regime Taylor rule on the CBRT overnight borrowing rate:

  i_t = alpha(s_t) + phi_pi(s_t) * pi_{t-1} + phi_y(s_t) * y_t + e_t,
        e_t ~ N(0, sigma2(s_t)),  s_t in {0,1} with transition matrix P.

Regimes are labelled active vs. passive by which has phi_pi > 1.
Computes ergodic probabilities, expected durations, the time-varying
Davig-Leeper determinacy index phi_bar_t, and passive-regime episodes.

A robustness specification replaces realised lagged inflation with an AR(1)
rolling forecast.

Outputs:
  data/taylor_results.json
  data/regime_probabilities.parquet
  results/csv/regime_probabilities.csv
  results/figures/ms_taylor_rule.png, ms_taylor_robustness.png
"""
import json
import warnings
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from .config import DATA_DIR, CSV_DIR, FIG_DIR

warnings.filterwarnings("ignore")


def _fit_ms(endog, exog, n_attempts=20):
    """Fit a 2-regime MarkovRegression with multiple restarts; return best by LLF."""
    model = MarkovRegression(
        endog=endog, k_regimes=2, exog=exog,
        switching_variance=True, switching_exog=True, switching_trend=True,
    )
    best = None; best_llf = -np.inf
    for _ in range(n_attempts):
        try:
            fit = model.fit(disp=False, maxiter=500, search_reps=10)
            if fit.llf > best_llf:
                best_llf = fit.llf; best = fit
        except Exception:
            continue
    return best


def _smoothed_probs(fit, passive_idx):
    """Return (p_passive_t, p_active_t) as numpy arrays."""
    sp = fit.smoothed_marginal_probabilities
    if hasattr(sp, "iloc"):
        p_pass = sp.iloc[:, passive_idx].values
        p_act = sp.iloc[:, 1 - passive_idx].values
    elif isinstance(sp, np.ndarray) and sp.ndim == 2:
        p_pass = sp[:, passive_idx]; p_act = sp[:, 1 - passive_idx]
    else:
        p_pass = np.array(sp[passive_idx]); p_act = np.array(sp[1 - passive_idx])
    return p_pass, p_act


def run():
    master = pl.read_parquet(DATA_DIR / "master.parquet")
    master = master.with_columns(pl.col("output_gap").fill_null(0.0))

    df = (
        master.select(["date", "overnight_borrowing", "inf_TP_FG_J0", "output_gap"])
        .with_columns(pl.col("inf_TP_FG_J0").shift(1).alias("lagged_inf"))
        .drop_nulls()
    )
    dates = df["date"].to_list()
    i_t = df["overnight_borrowing"].to_numpy()
    pi_lag = df["lagged_inf"].to_numpy()
    y_t = df["output_gap"].to_numpy()
    exog = np.column_stack([pi_lag, y_t])
    print(f"MS Taylor rule sample: {len(i_t)} months ({dates[0]} to {dates[-1]})")

    fit = _fit_ms(i_t, exog)
    p = fit.params
    # statsmodels interleaves params by type, not by regime:
    # [p00, p10, const0, const1, x1[0], x1[1], x2[0], x2[1], sig0, sig1]
    regimes = {
        0: {"alpha": p[2], "phi_pi": p[4], "phi_y": p[6], "sigma2": p[8]},
        1: {"alpha": p[3], "phi_pi": p[5], "phi_y": p[7], "sigma2": p[9]},
    }
    active = 0 if regimes[0]["phi_pi"] > regimes[1]["phi_pi"] else 1
    passive = 1 - active
    p_stay_active = p[0] if active == 0 else (1 - p[1])
    p_stay_passive = (1 - p[1]) if active == 0 else p[0]
    erg_active = (1 - p_stay_passive) / (2 - p_stay_active - p_stay_passive)
    erg_passive = 1 - erg_active
    phi_bar = erg_active * regimes[active]["phi_pi"] + erg_passive * regimes[passive]["phi_pi"]

    print(f"Active phi_pi = {regimes[active]['phi_pi']:.4f},  passive phi_pi = {regimes[passive]['phi_pi']:.4f}")
    print(f"Ergodic P(active)={erg_active:.3f}, P(passive)={erg_passive:.3f}")
    print(f"Davig-Leeper phi_bar = {phi_bar:.4f}  ({'determinate' if phi_bar > 1 else 'INDETERMINATE'})")

    p_passive_t, p_active_t = _smoothed_probs(fit, passive)
    phi_bar_t = p_active_t * regimes[active]["phi_pi"] + p_passive_t * regimes[passive]["phi_pi"]

    # Passive episodes by p > 0.5
    episodes = []
    in_pass = False; start = None
    for i, (d, pp) in enumerate(zip(dates, p_passive_t)):
        if pp > 0.5 and not in_pass:
            start = d; in_pass = True
        elif pp <= 0.5 and in_pass:
            episodes.append((start, dates[i - 1])); in_pass = False
    if in_pass:
        episodes.append((start, dates[-1]))

    # First date phi_bar drops <1 for 6 consecutive months
    count = 0; breach = None
    for i, pb in enumerate(phi_bar_t):
        if pb < 1:
            count += 1
            if count >= 6 and breach is None:
                breach = dates[i - 5]
        else:
            count = 0

    # Persist results
    out = {
        "active_regime": int(active), "passive_regime": int(passive),
        "phi_pi_active": float(regimes[active]["phi_pi"]),
        "phi_pi_passive": float(regimes[passive]["phi_pi"]),
        "phi_y_active": float(regimes[active]["phi_y"]),
        "phi_y_passive": float(regimes[passive]["phi_y"]),
        "alpha_active": float(regimes[active]["alpha"]),
        "alpha_passive": float(regimes[passive]["alpha"]),
        "sigma2_active": float(regimes[active]["sigma2"]),
        "sigma2_passive": float(regimes[passive]["sigma2"]),
        "p_stay_active": float(p_stay_active),
        "p_stay_passive": float(p_stay_passive),
        "ergodic_active": float(erg_active),
        "ergodic_passive": float(erg_passive),
        "phi_bar": float(phi_bar),
        "determinacy": bool(phi_bar > 1),
        "loglik": float(fit.llf), "aic": float(fit.aic), "bic": float(fit.bic),
        "nobs": int(len(i_t)),
        "sample_start": str(dates[0]), "sample_end": str(dates[-1]),
        "breach_date": str(breach) if breach else None,
        "passive_episodes": [(str(s), str(e)) for s, e in episodes],
    }
    with open(DATA_DIR / "taylor_results.json", "w") as f:
        json.dump(out, f, indent=2)

    df_reg = pl.DataFrame({
        "date": dates,
        "p_active": p_active_t.tolist(),
        "p_passive": p_passive_t.tolist(),
        "phi_bar_t": phi_bar_t.tolist(),
    })
    df_reg.write_parquet(DATA_DIR / "regime_probabilities.parquet")
    df_reg.write_csv(CSV_DIR / "regime_probabilities.csv")

    _make_baseline_plot(dates, i_t, p_active_t, p_passive_t, phi_bar_t)

    # ------------- robustness: AR(1) forecast inflation -------------
    _robustness_ar1(master, dates, p_passive_t)
    print("Stage 4 complete: MS Taylor rule estimated.")


def _make_baseline_plot(dates, i_t, p_active_t, p_passive_t, phi_bar_t):
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    axes[0].plot(dates, i_t, color="#2C2C2A", linewidth=1.2, label="Overnight borrowing rate (%)")
    axes[0].set_title("CBRT Overnight Borrowing Rate"); axes[0].legend(); axes[0].grid(True, alpha=0.2)

    axes[1].fill_between(dates, 0, p_passive_t, color="#E24B4A", alpha=0.4, label="P(passive)")
    axes[1].fill_between(dates, 0, p_active_t, color="#85B7EB", alpha=0.4, label="P(active)")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].set_ylim(0, 1); axes[1].set_title("Smoothed regime probabilities")
    axes[1].legend(); axes[1].grid(True, alpha=0.2)

    axes[2].plot(dates, phi_bar_t, color="#2C2C2A", linewidth=2, label="phi_bar(t)")
    axes[2].axhline(1.0, color="#E24B4A", linestyle="--", linewidth=1.5, alpha=0.7,
                    label="Taylor principle (phi_bar = 1)")
    axes[2].fill_between(dates, phi_bar_t, 1, where=np.array(phi_bar_t) < 1,
                         color="#E24B4A", alpha=0.15, label="Determinacy violation")
    axes[2].set_title("Time-varying Davig-Leeper determinacy condition")
    axes[2].set_xlabel("Date"); axes[2].legend(); axes[2].grid(True, alpha=0.2)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ms_taylor_rule.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _robustness_ar1(master, baseline_dates, baseline_p_passive):
    """Re-estimate replacing realised lag with a rolling AR(1) forecast."""
    headline = master.select("inf_TP_FG_J0").to_numpy().flatten()
    valid_mask = ~np.isnan(headline)
    valid_idx = np.where(valid_mask)[0]
    ar1_full = np.full(len(headline), np.nan)
    min_window = 24
    for j in range(min_window, len(valid_idx)):
        t = valid_idx[j]
        y = headline[valid_idx[:j]]
        X_ar = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        beta_ar = np.linalg.lstsq(X_ar, y[1:], rcond=None)[0]
        ar1_full[t] = beta_ar[0] + beta_ar[1] * headline[valid_idx[j - 1]]

    df = (
        master.select(["date", "overnight_borrowing", "output_gap"])
        .with_columns([
            pl.col("output_gap").fill_null(0.0),
            pl.Series("ar1_forecast", ar1_full),
        ])
        .with_columns(
            pl.when(pl.col("ar1_forecast").is_nan()).then(None)
              .otherwise(pl.col("ar1_forecast")).alias("ar1_forecast")
        )
        .drop_nulls()
    )
    dates_r = df["date"].to_list()
    fit_r = _fit_ms(
        df["overnight_borrowing"].to_numpy(),
        np.column_stack([df["ar1_forecast"].to_numpy(), df["output_gap"].to_numpy()]),
    )
    p = fit_r.params
    passive_r = 1 if p[5] > p[4] else 0  # passive has the smaller phi_pi
    p_passive_r, _ = _smoothed_probs(fit_r, passive_r)

    common = sorted(set(baseline_dates) & set(dates_r))
    baseline_vals = [baseline_p_passive[baseline_dates.index(d)] for d in common]
    robust_vals = [p_passive_r[dates_r.index(d)] for d in common]
    corr = np.corrcoef(baseline_vals, robust_vals)[0, 1]

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(common, baseline_vals, color="#A32D2D", linewidth=2,
            label="P(passive) - baseline (lagged realised inflation)")
    ax.plot(common, robust_vals, color="#185FA5", linewidth=2, linestyle="--",
            label="P(passive) - robustness (AR(1) forecast)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_title(f"Robustness: baseline vs. AR(1) forecast (correlation = {corr:.3f})")
    ax.legend(); ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ms_taylor_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Robustness correlation of P(passive): {corr:.4f}")


if __name__ == "__main__":
    run()
