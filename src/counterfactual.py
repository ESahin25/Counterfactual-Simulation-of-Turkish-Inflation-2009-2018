"""
Stage 6 — Counterfactual simulation.

Asks: what would inflation have looked like over the 2009-02 to 2018-05 passive
episode if monetary policy had instead followed a gradualist active rule?

Uses the sign-restricted IRFs from Stage 5 to identify the monetary policy
shock. At each period in the counterfactual window we:

  1. Compute the rate the active rule prescribes, given the path that the
     economy has taken so far under the cumulative effect of past CF shocks.
  2. Back out the MP shock v_t needed to hit that prescribed rate, accounting
     for the rate already implied by past shocks (no rebalancing magic, no
     anchoring to the actual rate).
  3. Propagate v_t through ALL seven variables via the IRF, accumulating
     effects on output, FX, factors, CPI, and the rate itself.

This is a structural full-system counterfactual: every variable evolves
according to the same identified shock, not just CPI.

Outputs:
  results/figures/counterfactual_final.png  (cumulative inflation gap)
  results/figures/counterfactual_factors.png  (F_iner, F_imp under CF)
  results/figures/counterfactual_system_panel.png  (full 4-var panel)
  results/figures/robustness_gradualist_cpi.png, robustness_gradualist_rate.png
  results/csv/counterfactual_rate_annual.csv, counterfactual_cpi_annual.csv
"""
import json
from datetime import datetime, date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .config import DATA_DIR, FIG_DIR, CSV_DIR

# Gradualist active-rule parameters (matches the paper's specification).
R_REAL = 2.0       # neutral real rate
TARGET_ANN = 3.0   # annual inflation target (%)
UPPER_ANN = 10.0   # ceiling above which response saturates
A_HIGH = 1.0       # max additional aggressiveness on gap (pi - target)
RHO_R = 0.3        # rate-smoothing
USE_OUTPUT_RULE = False
PHI_Y_RULE = 0.5

CF_START = date(2009, 2, 1)
CF_END = date(2018, 5, 1)


def _gap_response(pi_annual):
    if pi_annual >= UPPER_ANN:
        return A_HIGH
    if pi_annual <= TARGET_ANN:
        return 0.0
    return A_HIGH * (pi_annual - TARGET_ANN) / (UPPER_ANN - TARGET_ANN)


def _taylor_rate(pi_annual, ygap):
    r = R_REAL + pi_annual + _gap_response(pi_annual) * (pi_annual - TARGET_ANN)
    if USE_OUTPUT_RULE:
        r += PHI_Y_RULE * ygap
    return r


def _load_artefacts():
    draws = np.load(DATA_DIR / "ms_svar_draws_7var.npz", allow_pickle=True)
    irfs_npz = np.load(DATA_DIR / "irfs_sign_restricted.npz")
    Y_data = draws["Y"]
    dates_str = draws["dates"]
    dates_est = [datetime.strptime(str(d)[:10], "%Y-%m-%d").date() for d in dates_str]
    K = int(draws["K"])
    T = int(draws["T"])
    idx = {
        "output": int(draws["output_idx"]),
        "cpi": int(draws["cpi_idx"]),
        "policy": int(draws["policy_idx"]),
        "fx": int(draws["fx_idx"]),
        "f_imp": int(draws["f_imp_idx"]),
        "f_iner": int(draws["f_iner_idx"]),
    }
    irfs_passive = irfs_npz["irf_passive"]
    return Y_data, dates_est, K, T, idx, irfs_passive


def _find_window(dates_est):
    cf_start_idx = cf_end_idx = None
    for i, d in enumerate(dates_est):
        if d >= CF_START and cf_start_idx is None:
            cf_start_idx = i
        if d <= CF_END:
            cf_end_idx = i
    return cf_start_idx, cf_end_idx


def _full_system_counterfactual(Y_data, irfs, K, T, idx, cf_start, cf_end):
    """Solve for the MP shock path that holds the active rule, propagate through all 7 vars."""
    actual_cpi = Y_data[:, idx["cpi"]]
    actual_rate = Y_data[:, idx["policy"]]
    n_g = irfs.shape[0]
    H_irf = irfs.shape[1] - 1

    counter_cpi = np.zeros((n_g, T))
    cf_rate = np.zeros((n_g, T))
    shock = np.zeros((n_g, T))
    counter_all = np.zeros((n_g, T, K))
    explode = np.zeros(n_g, dtype=bool)

    for d_i in range(n_g):
        irf = irfs[d_i]                       # (H+1, K)
        pol_impact = irf[0, idx["policy"]]    # normalised to ~1.0
        eff = np.zeros((T, K))                # cumulative effect on all vars

        for t in range(cf_start, T):
            if cf_start <= t <= cf_end:
                lb = min(12, t + 1)
                pi_ann = np.sum(actual_cpi[t - lb + 1:t + 1]
                                + eff[t - lb + 1:t + 1, idx["cpi"]])
                ygap = Y_data[t, idx["output"]] + eff[t, idx["output"]]
                r_tar = _taylor_rate(pi_ann, ygap)
                if RHO_R > 0 and t > cf_start:
                    r_tar = RHO_R * cf_rate[d_i, t - 1] + (1 - RHO_R) * r_tar
                r_tar = max(r_tar, 0.0)

                rate_now = actual_rate[t] + eff[t, idx["policy"]]
                v_t = (r_tar - rate_now) / pol_impact
                hmax = min(H_irf, T - 1 - t)
                eff[t:t + hmax + 1] += v_t * irf[:hmax + 1]
                cf_rate[d_i, t] = actual_rate[t] + eff[t, idx["policy"]]
                shock[d_i, t] = v_t
            counter_all[d_i, t] = Y_data[t] + eff[t]

        counter_cpi[d_i] = actual_cpi + eff[:, idx["cpi"]]
        seg = counter_cpi[d_i, cf_start:cf_end + 1]
        if (not np.all(np.isfinite(seg))) or np.max(np.abs(seg)) > 20:
            explode[d_i] = True

    ok = ~explode
    return counter_cpi[ok], cf_rate[ok], shock[ok], counter_all[ok], ok


def _annualise(monthly, T):
    """Rolling-12 annualised inflation from monthly log-percent changes."""
    out = np.full(T, np.nan)
    for t in range(12, T):
        out[t] = (np.exp(np.sum(monthly[t - 11:t + 1]) / 100) - 1) * 100
    return out


def _save_comparison_csvs(actual_cpi, actual_rate, ccpi, crate, dates_est, cf_start, cf_end, T):
    """Yearly and 6-monthly comparison tables for actual vs CF rate and CPI."""
    idx_cf = np.arange(cf_start, cf_end + 1)
    dates_cf = pd.to_datetime([dates_est[i] for i in idx_cf])

    aa = _annualise(actual_cpi, T)
    ac = np.array([_annualise(c, T) for c in ccpi])

    df_rate = pd.DataFrame({
        "actual": actual_rate[idx_cf],
        "cf_median": np.median(crate[:, idx_cf], 0),
        "cf_p16": np.percentile(crate[:, idx_cf], 16, 0),
        "cf_p84": np.percentile(crate[:, idx_cf], 84, 0),
    }, index=dates_cf)
    df_rate["gap"] = df_rate["cf_median"] - df_rate["actual"]

    df_cpi = pd.DataFrame({
        "actual": aa[idx_cf],
        "cf_median": np.nanmedian(ac, 0)[idx_cf],
        "cf_p16": np.nanpercentile(ac, 16, 0)[idx_cf],
        "cf_p84": np.nanpercentile(ac, 84, 0)[idx_cf],
    }, index=dates_cf)
    df_cpi["gap"] = df_cpi["cf_median"] - df_cpi["actual"]

    df_rate.round(3).to_csv(CSV_DIR / "counterfactual_rate_monthly.csv")
    df_cpi.round(3).to_csv(CSV_DIR / "counterfactual_cpi_monthly.csv")
    df_rate.resample("YE").mean().round(3).to_csv(CSV_DIR / "counterfactual_rate_annual.csv")
    df_cpi.resample("YE").mean().round(3).to_csv(CSV_DIR / "counterfactual_cpi_annual.csv")
    return aa, ac


def _make_plots(Y_data, ccpi, crate, counter_all, ok, cum_gap, aa, ac,
                dates_est, T, idx, cf_start, cf_end):
    pdl = [datetime(d.year, d.month, d.day) for d in dates_est]
    actual_cpi = Y_data[:, idx["cpi"]]
    actual_rate = Y_data[:, idx["policy"]]
    wd = [pdl[i] for i in range(cf_start, cf_end + 1)]
    W = slice(cf_start, cf_end + 1)

    # FIG A: cumulative inflation gap with bands
    fig, ax = plt.subplots(figsize=(14, 5))
    gm = np.median(cum_gap[:, W], 0)
    g16 = np.percentile(cum_gap[:, W], 16, 0)
    g84 = np.percentile(cum_gap[:, W], 84, 0)
    g05 = np.percentile(cum_gap[:, W], 5, 0)
    g95 = np.percentile(cum_gap[:, W], 95, 0)
    ax.plot(wd, gm, color="#E24B4A", linewidth=2, label="Median")
    ax.fill_between(wd, g16, g84, color="#F5A6A6", alpha=0.25, label="68%")
    ax.fill_between(wd, g05, g95, color="#F5A6A6", alpha=0.10, label="90%")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative gap (p.p.)")
    ax.set_title("Cumulative inflation gap (actual - counterfactual)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "counterfactual_final.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # FIG B: factor trajectories under CF vs actual
    fig, ax = plt.subplots(figsize=(14, 5))
    for ix, lab, c in [(idx["f_iner"], "Inertial factor (CF)", "#E24B4A"),
                       (idx["f_imp"], "Imported factor (CF)", "#185FA5")]:
        med = np.median(counter_all[:, W, ix], 0)
        ax.plot(wd, med, linewidth=2, color=c, label=lab)
        ax.fill_between(wd, np.percentile(counter_all[:, W, ix], 16, 0),
                        np.percentile(counter_all[:, W, ix], 84, 0), color=c, alpha=0.12)
    ax.plot(wd, Y_data[W, idx["f_iner"]], linewidth=1.4, color="black", linestyle="--",
            label="Inertial factor (actual)")
    ax.plot(wd, Y_data[W, idx["f_imp"]], linewidth=1.4, color="gray", linestyle=":",
            label="Imported factor (actual)")
    ax.set_ylabel("Factor level")
    ax.set_title("Latent factors: actual vs counterfactual")
    ax.legend(fontsize=9, ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "counterfactual_factors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # FIG C: 4-panel system response
    panels = [(idx["output"], "Output gap"), (idx["fx"], "d Exchange rate"),
              (idx["f_iner"], "Inertial factor"), (idx["f_imp"], "Imported factor")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    for axx, (ix, lab) in zip(axes.ravel(), panels):
        axx.plot(wd, Y_data[W, ix], "k", linewidth=1.6, label="Actual")
        axx.plot(wd, np.median(counter_all[:, W, ix], 0), "--", color="#E24B4A",
                 linewidth=1.8, label="Counterfactual")
        axx.fill_between(wd, np.percentile(counter_all[:, W, ix], 16, 0),
                         np.percentile(counter_all[:, W, ix], 84, 0),
                         color="#F5A6A6", alpha=0.15)
        axx.set_title(lab, fontsize=11)
        axx.xaxis.set_major_locator(mdates.YearLocator(2))
        axx.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes.ravel()[0].legend(fontsize=9)
    fig.suptitle("Counterfactual vs actual - full system", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "counterfactual_system_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # FIG D: annualised CPI overlay, post 12-month warmup
    idx_plot = np.arange(max(cf_start, 12), cf_end + 1)
    xd = [pdl[i] for i in idx_plot]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(xd, aa[idx_plot], "k", linewidth=2, label="Actual")
    ax.plot(xd, np.nanmedian(ac, 0)[idx_plot], "--", color="#E24B4A", linewidth=2,
            label="Counterfactual")
    ax.fill_between(xd, np.nanpercentile(ac, 16, 0)[idx_plot],
                    np.nanpercentile(ac, 84, 0)[idx_plot], color="#F5A6A6", alpha=0.15)
    ax.axhline(TARGET_ANN, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Annualised CPI: actual vs counterfactual")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "robustness_gradualist_cpi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # FIG E: policy rate overlay
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(wd, actual_rate[W], "k", linewidth=2, label="Actual")
    ax.plot(wd, np.median(crate[:, W], 0), "--", color="#E24B4A", linewidth=2,
            label="Counterfactual")
    ax.set_title("Policy rate: actual vs counterfactual")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "robustness_gradualist_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    Y_data, dates_est, K, T, idx, irfs_passive = _load_artefacts()
    cf_start, cf_end = _find_window(dates_est)
    months = cf_end - cf_start + 1
    print(f"Counterfactual window: {dates_est[cf_start]} -> {dates_est[cf_end]}  ({months} months)")

    ccpi, crate, shock, counter_all, ok = _full_system_counterfactual(
        Y_data, irfs_passive, K, T, idx, cf_start, cf_end
    )
    print(f"Stable draws: {ok.sum()}/{len(ok)}  (exploded: {(~ok).sum()})")

    actual_cpi = Y_data[:, idx["cpi"]]
    cum_actual = np.cumsum(actual_cpi)
    cum_cf = np.cumsum(ccpi, axis=1)
    cum_gap = cum_actual[np.newaxis, :] - cum_cf
    gap_end = cum_gap[:, cf_end]

    summary = {
        "cf_window": [str(dates_est[cf_start]), str(dates_est[cf_end])],
        "n_months": int(months),
        "n_stable_draws": int(ok.sum()),
        "median_cumulative_gap_pp": float(np.median(gap_end)),
        "annual_gap_pp_per_year": float(np.median(gap_end) / (months / 12)),
        "p_gap_positive": float((gap_end > 0).mean()),
        "ci_68": [float(np.percentile(gap_end, 16)), float(np.percentile(gap_end, 84))],
        "ci_90": [float(np.percentile(gap_end, 5)), float(np.percentile(gap_end, 95))],
        "target_annual": TARGET_ANN,
        "rho_r": RHO_R,
        "r_real": R_REAL,
    }
    with open(DATA_DIR / "counterfactual_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Median cumulative gap: {summary['median_cumulative_gap_pp']:+.2f} p.p.")
    print(f"  Annualised: {summary['annual_gap_pp_per_year']:+.2f} p.p./year")
    print(f"  P(gap > 0): {summary['p_gap_positive']:.3f}")
    print(f"  68% band: [{summary['ci_68'][0]:+.1f}, {summary['ci_68'][1]:+.1f}]")
    print(f"  90% band: [{summary['ci_90'][0]:+.1f}, {summary['ci_90'][1]:+.1f}]")

    aa, ac = _save_comparison_csvs(actual_cpi, Y_data[:, idx["policy"]], ccpi, crate,
                                   dates_est, cf_start, cf_end, T)
    _make_plots(Y_data, ccpi, crate, counter_all, ok, cum_gap, aa, ac,
                dates_est, T, idx, cf_start, cf_end)

    # Factor decomposition
    allf = counter_all
    W = slice(cf_start, cf_end + 1)

    def cumgap_factor(ix):
        cf = allf[:, W, ix]
        act = Y_data[W, ix]
        return np.cumsum(act[None, :] - cf, axis=1)[:, -1]
    g_iner = cumgap_factor(idx["f_iner"])
    g_imp = cumgap_factor(idx["f_imp"])
    print("\nFactor counterfactual (cumulative reduction under active policy)")
    print(f"  F_iner: median {np.median(g_iner):+.2f}  P>0 {np.mean(g_iner > 0):.3f}  "
          f"68% [{np.percentile(g_iner, 16):+.2f}, {np.percentile(g_iner, 84):+.2f}]")
    print(f"  F_imp : median {np.median(g_imp):+.2f}  P>0 {np.mean(g_imp > 0):.3f}  "
          f"68% [{np.percentile(g_imp, 16):+.2f}, {np.percentile(g_imp, 84):+.2f}]")

    print("Stage 6 complete: counterfactual saved.")


if __name__ == "__main__":
    run()
