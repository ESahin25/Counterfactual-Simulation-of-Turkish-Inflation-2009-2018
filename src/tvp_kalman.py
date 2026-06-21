"""
Stage 2 — TVP-Kalman estimation of sub-index inflation inertia.

For each 3-digit COICOP sub-index i, estimate a state-space model where the
inertia coefficient alpha_{i,t} follows a random walk:

  Measurement:  pi_{i,t} = alpha_{i,t} * pi_{i,t-1} + beta' X_t + eps_t
  State:        alpha_{i,t} = alpha_{i,t-1} + eta_t

with X_t = [d_usdtry_t, d_commodity_t, output_gap_t].

The smoothed alpha_{i,t|T} paths and per-sector parameter estimates are saved
so the next stage can group sub-indices into high-inertia / low-inertia bins.

Outputs:
  data/kalman_results.json  (per-sector parameter estimates, post-2008 avg alpha)
  data/alpha_paths.json     (smoothed alpha_{i,t} paths and dates)
  results/csv/kalman_summary.csv
"""
import json
import warnings
import numpy as np
import polars as pl
from statsmodels.tsa.statespace.mlemodel import MLEModel

from .config import DATA_DIR, CSV_DIR

warnings.filterwarnings("ignore")


class TVPInertia(MLEModel):
    """State-space model with time-varying inertia coefficient on lagged inflation."""

    def __init__(self, endog, lagged_infl, exog):
        super().__init__(endog, k_states=1, k_posdef=1, initialization="diffuse")
        self._lagged = lagged_infl
        self._exog = exog
        self["transition", 0, 0] = 1.0
        self["selection", 0, 0] = 1.0

    @property
    def param_names(self):
        return ["beta_fx", "beta_comm", "beta_gap", "sigma2_eps", "sigma2_eta"]

    @property
    def start_params(self):
        return np.array([0.05, 0.05, 0.05, 1.0, 0.01])

    def transform_params(self, u):
        c = u.copy(); c[3] = c[3] ** 2; c[4] = c[4] ** 2
        return c

    def untransform_params(self, c):
        u = c.copy(); u[3] = u[3] ** 0.5; u[4] = u[4] ** 0.5
        return u

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        beta = params[:3]
        self["design"] = self._lagged.reshape(1, 1, -1)
        self["obs_intercept"] = (self._exog @ beta).reshape(1, -1)
        self["obs_cov", 0, 0] = params[3]
        self["state_cov", 0, 0] = params[4]


def run():
    master = pl.read_parquet(DATA_DIR / "master.parquet")
    # Fill trailing output_gap nulls with 0 (on-trend assumption) so we don't
    # lose the final months of the sample.
    master = master.with_columns(pl.col("output_gap").fill_null(0.0))

    # 3-digit sub-indices only; exclude J126 (financial services n.e.c.) which
    # has a structurally unstable variance and contaminates downstream PCA.
    sub_idx_cols = sorted(
        c for c in master.columns
        if c.startswith("inf_TP_FG_J")
        and len(c.replace("inf_TP_FG_J", "")) == 3
        and c != "inf_TP_FG_J126"
    )
    controls = ["d_usdtry", "d_commodity", "output_gap"]
    print(f"Estimating TVP-Kalman for {len(sub_idx_cols)} sub-indices")

    results = {}
    alpha_paths = {}

    for i, inf_col in enumerate(sub_idx_cols):
        label = inf_col.replace("inf_", "")
        df_est = (
            master.select(["date", inf_col] + controls)
            .with_columns(pl.col(inf_col).shift(1).alias("lagged_inf"))
            .drop_nulls()
        )
        endog = df_est[inf_col].to_numpy()
        lagged = df_est["lagged_inf"].to_numpy()
        exog = df_est[controls].to_numpy()
        dates = df_est["date"].to_list()

        try:
            model = TVPInertia(endog, lagged, exog)
            fit = model.fit(disp=False, maxiter=500)
            smoothed = fit.smoothed_state[0]
            p = fit.params
            post_2008_mask = np.array([d.year >= 2008 for d in dates])
            avg_alpha = float(smoothed[post_2008_mask].mean())
            results[label] = {
                "beta_fx": float(p[0]), "beta_comm": float(p[1]), "beta_gap": float(p[2]),
                "sigma2_eps": float(p[3]), "sigma2_eta": float(p[4]),
                "loglik": float(fit.llf), "nobs": int(len(endog)),
                "converged": bool(fit.mle_retvals["converged"]),
                "avg_alpha_post2008": avg_alpha,
            }
            alpha_paths[label] = {"dates": dates, "alpha": smoothed.tolist()}
            tick = "OK" if fit.mle_retvals["converged"] else "no-converge"
            print(f"  [{i+1:2d}/{len(sub_idx_cols)}] {label:<15} alpha_post2008={avg_alpha:>7.4f}  sigma2_eta={p[4]:.6f}  {tick}")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(sub_idx_cols)}] {label:<15} FAILED: {e}")
            results[label] = {"failed": True, "error": str(e)}

    # JSON outputs
    with open(DATA_DIR / "kalman_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(DATA_DIR / "alpha_paths.json", "w") as f:
        json.dump(alpha_paths, f, default=str)

    # CSV summary for humans
    rows = []
    for lab, r in results.items():
        if r.get("failed"):
            continue
        rows.append({
            "sector": lab,
            "avg_alpha_post2008": r["avg_alpha_post2008"],
            "sigma2_eps": r["sigma2_eps"],
            "sigma2_eta": r["sigma2_eta"],
            "beta_fx": r["beta_fx"],
            "beta_comm": r["beta_comm"],
            "beta_gap": r["beta_gap"],
            "loglik": r["loglik"],
            "converged": r["converged"],
        })
    summary = pl.DataFrame(rows).sort("avg_alpha_post2008", descending=True)
    summary.write_csv(CSV_DIR / "kalman_summary.csv")
    print(f"Stage 2 complete: kalman_summary.csv written ({len(rows)} sectors)")


if __name__ == "__main__":
    run()
