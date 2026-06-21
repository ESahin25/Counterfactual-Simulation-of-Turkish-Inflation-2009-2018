"""
Stage 5 — Two-regime Bayesian SVAR with sign-restricted identification.

7-variable VAR with ordering [output_gap, d_commodity, d_usdtry, F_imp,
F_iner, inf_CPI, overnight_borrowing], 2 lags, Minnesota prior. Regimes are
fixed to the Stage-4 classification (p_passive > 0.5 -> regime 1). The Gibbs
sampler iterates over (B, Sigma) per regime using the inverse-Wishart for
Sigma and Normal posterior for each column of B.

Then identifies a contractionary monetary policy shock via sign restrictions
(Uhlig 2005): for h = 0..5, policy rate > 0, CPI <= 0, FX <= 0, with
Kilian-Murphy magnitude bounds and a post-filter removing draws where the
24-month cumulative CPI response turns positive.

Outputs:
  data/ms_svar_draws_7var.npz  (B / Sigma posterior draws per regime)
  data/irfs_sign_restricted.npz  (accepted post-filtered IRF draws per regime)
  results/figures/irf_7var.png, irf_factor_comparison_7var.png,
                  irf_7var_sr.png, irf_factor_7var_sr.png
"""
import warnings
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import invwishart

from .config import DATA_DIR, FIG_DIR

warnings.filterwarnings("ignore")

VAR_COLS = ["output_gap", "d_commodity", "d_usdtry", "F_imp", "F_iner",
            "inf_TP_FG_J0", "overnight_borrowing"]
VAR_NAMES = ["Output Gap", "dCommodity", "dExRate", "F_imp", "F_iner", "pi_CPI", "Policy Rate"]
P_LAG = 2

# index positions corresponding to VAR_COLS / VAR_NAMES
IDX = dict(output=0, commodity=1, fx=2, f_imp=3, f_iner=4, cpi=5, policy=6)


def _build_var_data():
    master = pl.read_parquet(DATA_DIR / "master.parquet").with_columns(
        pl.col("output_gap").fill_null(0.0)
    )
    df = master.select(["date"] + VAR_COLS)
    for col in VAR_COLS:
        df = df.with_columns(
            pl.when(pl.col(col).is_nan()).then(None).otherwise(pl.col(col)).alias(col)
        )
    df = df.drop_nulls()
    dates_var = df["date"].to_list()
    Y_raw = df.select(VAR_COLS).to_numpy()
    T_raw, K = Y_raw.shape
    Y = Y_raw[P_LAG:]
    T = Y.shape[0]
    X_list = [np.ones((T, 1))]
    for lag in range(1, P_LAG + 1):
        X_list.append(Y_raw[P_LAG - lag:T_raw - lag])
    X = np.hstack(X_list)
    dates_est = dates_var[P_LAG:]
    return Y_raw, Y, X, dates_est, T, K, T_raw


def _minnesota_prior_cov(k, sigma_sq, K, p_lag, lambda1=0.2, lambda2=0.5):
    M = K * p_lag + 1
    V = np.zeros(M)
    V[0] = 100.0  # diffuse intercept
    for lag in range(1, p_lag + 1):
        for j in range(K):
            idx = 1 + (lag - 1) * K + j
            if j == k:
                V[idx] = (lambda1 / lag) ** 2
            else:
                V[idx] = (lambda1 * lambda2 / lag) ** 2 * (sigma_sq[k] / max(sigma_sq[j], 1e-6))
    return np.diag(V)


def _gibbs_sample(Y, X, Y_raw, T, K, S_fixed, n_iter=10000, n_burn=5000):
    M = X.shape[1]
    # OLS-style residual variances for the Minnesota prior scale
    sigma_ols = np.zeros(K)
    for k in range(K):
        x_k = np.column_stack([np.ones(T)] + [Y_raw[P_LAG - l:Y_raw.shape[0] - l, k]
                                              for l in range(1, P_LAG + 1)])
        beta_k = np.linalg.lstsq(x_k, Y[:, k], rcond=None)[0]
        sigma_ols[k] = np.var(Y[:, k] - x_k @ beta_k)

    B_prior = np.zeros((M, K))
    for i in range(K):
        B_prior[1 + i, i] = 1.0  # AR(1) shrinkage

    # Initialise from per-regime OLS
    B_cur, Sig_cur = {}, {}
    for r in [0, 1]:
        mask = (S_fixed == r); Y_r = Y[mask]; X_r = X[mask]
        B_r = np.linalg.lstsq(X_r, Y_r, rcond=None)[0]
        resid = Y_r - X_r @ B_r
        B_cur[r] = B_r
        Sig_cur[r] = (resid.T @ resid) / max(resid.shape[0] - M, 1)

    n_save = n_iter - n_burn
    B_draws = {0: [], 1: []}; Sig_draws = {0: [], 1: []}
    print(f"  Gibbs sampler: {n_iter} iterations, {n_burn} burn-in...")

    for it in range(n_iter):
        for r in [0, 1]:
            mask = (S_fixed == r); n_r = mask.sum()
            Y_r = Y[mask]; X_r = X[mask]
            resid = Y_r - X_r @ B_cur[r]
            S0 = np.eye(K) * 0.01; nu0 = K + 2
            Sig_cur[r] = invwishart.rvs(df=nu0 + n_r, scale=S0 + resid.T @ resid)

            B_new = np.zeros((M, K))
            for k in range(K):
                V_pr = _minnesota_prior_cov(k, sigma_ols, K, P_LAG)
                V_pr_inv = np.linalg.inv(V_pr)
                V_post_inv = V_pr_inv + (1 / Sig_cur[r][k, k]) * (X_r.T @ X_r)
                V_post = np.linalg.inv(V_post_inv)
                b_post = V_post @ (V_pr_inv @ B_prior[:, k]
                                   + (1 / Sig_cur[r][k, k]) * (X_r.T @ Y_r[:, k]))
                B_new[:, k] = np.random.multivariate_normal(b_post, V_post)
            B_cur[r] = B_new

        if it >= n_burn:
            for r in [0, 1]:
                B_draws[r].append(B_cur[r].copy())
                Sig_draws[r].append(Sig_cur[r].copy())
        if (it + 1) % 2000 == 0:
            print(f"    iter {it + 1}/{n_iter}")

    for r in [0, 1]:
        B_draws[r] = np.array(B_draws[r])
        Sig_draws[r] = np.array(Sig_draws[r])
    return B_draws, Sig_draws, sigma_ols


def _compute_irf(B, K, p_lag, impact, H):
    Kp = K * p_lag
    F = np.zeros((Kp, Kp))
    for lag in range(p_lag):
        F[:K, lag * K:(lag + 1) * K] = B[1 + lag * K:1 + (lag + 1) * K, :].T
    if p_lag > 1:
        F[K:, :K * (p_lag - 1)] = np.eye(K * (p_lag - 1))
    state = np.zeros(Kp); state[:K] = impact
    irf = np.zeros((H + 1, K)); irf[0] = state[:K]
    for h in range(1, H + 1):
        state = F @ state; irf[h] = state[:K]
    return irf


def _cholesky_irfs(B_draws, Sig_draws, K, H=24, shock_size=1.0, thin=5):
    shock_var = K - 1  # policy rate is last
    n_save = B_draws[0].shape[0]
    idx = np.arange(0, n_save, thin)
    irfs = {0: np.zeros((len(idx), H + 1, K)), 1: np.zeros((len(idx), H + 1, K))}
    for d_i, d in enumerate(idx):
        for r in [0, 1]:
            Sigma = Sig_draws[r][d]
            try:
                P_chol = np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                P_chol = np.linalg.cholesky(Sigma + np.eye(K) * 0.001)
            impact = P_chol[:, shock_var] * shock_size
            irfs[r][d_i] = _compute_irf(B_draws[r][d], K, P_LAG, impact, H)
    return irfs


def _sign_restricted_irfs(B_draws, Sig_draws, K, H=24, H_restrict=6,
                          target_accepted=500, max_rotations=10000):
    """Uhlig (2005) Q-rotation accept-reject with Kilian-Murphy magnitude bounds."""
    max_cpi, max_out = 0.5, 5.0
    n_save = B_draws[0].shape[0]
    out = {0: [], 1: []}
    for r in [0, 1]:
        lab = "ACTIVE" if r == 0 else "PASSIVE"
        accepted = total_tried = draw_cycle = 0
        print(f"  {lab}: searching for {target_accepted} accepted draws...")
        while accepted < target_accepted:
            d = draw_cycle % n_save; draw_cycle += 1
            B = B_draws[r][d]; Sigma = Sig_draws[r][d]
            try:
                P_chol = np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                P_chol = np.linalg.cholesky(Sigma + np.eye(K) * 0.001)
            for _ in range(max_rotations):
                total_tried += 1
                q = np.random.randn(K); q = q / np.linalg.norm(q)
                impact = P_chol @ q
                if impact[IDX["policy"]] < 0:
                    impact = -impact
                impact = impact * (1.0 / impact[IDX["policy"]])
                irf = _compute_irf(B, K, P_LAG, impact, H)
                if (np.all(irf[:H_restrict, IDX["policy"]] > 0)
                        and np.all(irf[:H_restrict, IDX["cpi"]] <= 0)
                        and np.all(irf[:H_restrict, IDX["fx"]] <= 0)
                        and np.all(np.abs(irf[:H_restrict, IDX["cpi"]]) < max_cpi)
                        and np.all(np.abs(irf[:H_restrict, IDX["output"]]) < max_out)):
                    out[r].append(irf); accepted += 1; break
            if draw_cycle > n_save * 50:
                print(f"    exhausted after {accepted} accepted")
                break
        rate = accepted / max(total_tried, 1) * 100
        print(f"    {lab}: accepted {accepted}/{total_tried} ({rate:.2f}%)")
    # post-filter: drop draws where cumulative 24-month CPI response turns positive
    for r in [0, 1]:
        arr = np.array(out[r])
        keep = np.sum(arr[:, :, IDX["cpi"]], axis=1) <= 0
        out[r] = arr[keep]
        print(f"  {'ACTIVE' if r == 0 else 'PASSIVE'}: {arr.shape[0]} -> {out[r].shape[0]} after post-filter")
    return out


def _plot_irfs(irfs, filename, title, K, H=24, H_restrict=None):
    horizons = np.arange(H + 1)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    for k in range(K):
        ax = axes[k]
        for r, label, color, fill in [(0, "Active", "#185FA5", "#85B7EB"),
                                       (1, "Passive", "#A32D2D", "#E24B4A")]:
            irf = irfs[r][:, :, k]
            med = np.median(irf, axis=0)
            q16 = np.percentile(irf, 16, axis=0)
            q84 = np.percentile(irf, 84, axis=0)
            ax.plot(horizons, med, color=color, linewidth=2, label=label)
            ax.fill_between(horizons, q16, q84, color=fill, alpha=0.15)
        ax.axhline(0, color="gray", linewidth=0.5)
        if H_restrict is not None:
            ax.axvspan(0, H_restrict - 1, color="gray", alpha=0.05)
        ax.set_title(VAR_NAMES[k]); ax.set_xlabel("Months")
        ax.grid(True, alpha=0.2)
        if k == 0:
            ax.legend(fontsize=9)
    axes[7].set_visible(False)
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_factor_irf(irfs, filename, title, H=24, H_restrict=None):
    horizons = np.arange(H + 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for r, label, ax in [(0, "Active Regime", axes[0]), (1, "Passive Regime", axes[1])]:
        for var_idx, vname, color in [(IDX["f_imp"], "F_imp", "#185FA5"),
                                       (IDX["f_iner"], "F_iner", "#A32D2D"),
                                       (IDX["cpi"], "pi_CPI", "#2C2C2A")]:
            irf = irfs[r][:, :, var_idx]
            med = np.median(irf, axis=0)
            q16 = np.percentile(irf, 16, axis=0)
            q84 = np.percentile(irf, 84, axis=0)
            ls = "--" if var_idx == IDX["cpi"] else "-"
            ax.plot(horizons, med, color=color, linewidth=2, linestyle=ls, label=vname)
            ax.fill_between(horizons, q16, q84, color=color, alpha=0.08)
        ax.axhline(0, color="gray", linewidth=0.5)
        if H_restrict is not None:
            ax.axvspan(0, H_restrict - 1, color="gray", alpha=0.05)
        ax.set_title(label); ax.set_xlabel("Months"); ax.set_ylabel("Response")
        ax.legend(); ax.grid(True, alpha=0.2)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    np.random.seed(1923)

    Y_raw, Y, X, dates_est, T, K, T_raw = _build_var_data()
    print(f"VAR sample: T={T} months, K={K} variables, {dates_est[0]} -> {dates_est[-1]}")

    # Use the Stage-4 classification: regime 0 (active) / 1 (passive)
    regime_df = pl.read_parquet(DATA_DIR / "regime_probabilities.parquet").select(["date", "p_passive"])
    df_m = pl.DataFrame({"date": dates_est}).join(regime_df, on="date", how="left") \
        .with_columns(pl.col("p_passive").fill_null(0.5))
    S_fixed = (df_m["p_passive"].to_numpy() > 0.5).astype(int)
    print(f"Regimes from Stage 4: active={int((S_fixed == 0).sum())}, passive={int((S_fixed == 1).sum())}")

    B_draws, Sig_draws, _ = _gibbs_sample(Y, X, Y_raw, T, K, S_fixed)
    print(f"  Gibbs done: {B_draws[0].shape[0]} draws per regime")

    # ----- Cholesky IRFs (informal check for price puzzle) -----
    irfs_chol = _cholesky_irfs(B_draws, Sig_draws, K)
    _plot_irfs(irfs_chol, "irf_7var.png",
               "7-var MS-SVAR: Cholesky IRFs to 1 p.p. policy shock\n"
               "Blue=Active | Red=Passive | Bands = 68% credible", K)
    _plot_factor_irf(irfs_chol, "irf_factor_comparison_7var.png",
                     "F_iner vs F_imp vs pi_CPI: response to policy shock (Cholesky)")

    # ----- Sign-restricted IRFs (the identified objects we use downstream) -----
    irfs_sr = _sign_restricted_irfs(B_draws, Sig_draws, K)
    _plot_irfs(irfs_sr, "irf_7var_sr.png",
               "Sign-restricted IRFs (7-var, post-filtered): contractionary policy shock (1 p.p.)\n"
               "Blue=Active | Red=Passive | Gray=restriction window", K, H_restrict=6)
    _plot_factor_irf(irfs_sr, "irf_factor_7var_sr.png",
                     "F_iner vs F_imp vs pi_CPI: contractionary shock (sign-restricted)",
                     H_restrict=6)

    # ----- Persist artefacts -----
    np.savez_compressed(
        DATA_DIR / "ms_svar_draws_7var.npz",
        B_active=B_draws[0], B_passive=B_draws[1],
        Sigma_active=Sig_draws[0], Sigma_passive=Sig_draws[1],
        S_fixed=S_fixed, Y=Y, X=X, Y_raw=Y_raw,
        dates=[str(d) for d in dates_est],
        var_names=VAR_NAMES, var_cols=VAR_COLS,
        K=K, p=P_LAG, T=T,
        cpi_idx=IDX["cpi"], policy_idx=IDX["policy"],
        fx_idx=IDX["fx"], commodity_idx=IDX["commodity"],
        f_imp_idx=IDX["f_imp"], f_iner_idx=IDX["f_iner"],
        output_idx=IDX["output"],
    )
    np.savez_compressed(
        DATA_DIR / "irfs_sign_restricted.npz",
        irf_active=irfs_sr[0], irf_passive=irfs_sr[1],
    )
    print("Stage 5 complete: VAR draws and sign-restricted IRFs saved.")


if __name__ == "__main__":
    run()
