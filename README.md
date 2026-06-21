# Monetary Regimes and Inflation Inertia in Turkey

EC432 Applied Econometrics — replication code for "Monetary Regimes and Inflation Inertia in Turkey".

The pipeline estimates a three-stage framework on monthly Turkish data, 2003–2026:

1. **Sectoral inertia (TVP-Kalman).** For each 3-digit COICOP CPI sub-index, a state-space model with a random-walk inertia coefficient α_{i,t} on lagged sub-index inflation, conditional on FX, commodity, and output-gap controls. 43 sub-indices are then split into high-inertia (backward-indexed) and low-inertia (market-determined) groups at the cross-sectional mean of post-2008 average α.
2. **Latent factors (PCA).** First principal component of each group, sign-normalised to be positively correlated with headline inflation. `F_iner` captures backward-indexed pricing pressure; `F_imp` captures imported-shock pressure.
3. **Regime-switching policy & SVAR.** A 2-regime Markov-switching Taylor rule on the CBRT overnight borrowing rate identifies an active (φ_π > 1) and a passive episode (Feb 2009 – May 2018, φ_π^P ≈ 0.67). The regime sequence is then imposed on a 7-variable Bayesian SVAR with a Minnesota prior, identified via Uhlig (2005) sign restrictions with Kilian-Murphy magnitude bounds. A full-system structural counterfactual asks what inflation would have done had policy followed a gradualist active rule over the passive episode.

The headline counterfactual finding: consistently active policy would have reduced the cumulative price level by ≈34.5 p.p. over the passive episode (68% CI: 27.9–40.4; Pr(gap > 0) = 0.986), with average annual inflation falling from ≈8.3% to ≈4.5%.

## Repository structure

```
.
├── main.py                       # orchestrator, runs all stages
├── requirements.txt
├── .env.example                  # copy to .env and add your EVDS key
├── .gitignore
├── src/
│   ├── config.py                 # paths and API key loading
│   ├── data_collection.py        # Stage 0: EVDS, FRED, embedded CBRT/IPI series
│   ├── build_master.py           # Stage 1: merge + log-diff inflation, HP-filter output gap
│   ├── tvp_kalman.py             # Stage 2: 43 sectoral state-space models
│   ├── pca_factors.py            # Stage 3: F_iner and F_imp
│   ├── ms_taylor.py              # Stage 4: MS Taylor rule + AR(1) robustness
│   ├── ms_svar.py                # Stage 5: Gibbs sampler + sign-restricted IRFs
│   ├── counterfactual.py         # Stage 6: full-system structural counterfactual
│   └── export_csv.py             # Stage 7: CSV mirrors of all parquet files
├── data/                         # intermediate parquet/json/npz (gitignored)
└── results/
    ├── figures/                  # all PNGs
    └── csv/                      # all CSV tables
```

## Setup

```bash
git clone <your-repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The EVDS API key is set in `src/config.py`. If you want to use your own, register at https://evds2.tcmb.gov.tr/ and edit the `EVDS_API_KEY` constant.

## Running

```bash
python main.py                   # full pipeline end-to-end (~5–15 minutes; the Gibbs sampler dominates)
python main.py --only 5 6        # rerun the SVAR and counterfactual after editing one of them
python main.py --skip 0          # skip the data pull if data/ is already populated
```

Results land in:
- `results/figures/` — every PNG referenced in the paper (PCA factors, MS Taylor regimes and determinacy index, sign-restricted IRFs, counterfactual gap and factor decomposition, robustness overlays).
- `results/csv/` — Kalman parameter summary, regime probabilities, counterfactual rate and CPI tables at monthly / annual frequency, plus CSV mirrors of the raw parquet data.

## Notes

- **CBRT overnight borrowing rate.** The discrete decisions are embedded in `src/data_collection.py` (`ON_DECISIONS`) because EVDS does not expose the rate corridor consistently across the asymmetric-corridor period (Nov 2010 – Jul 2011) and the multiple late-liquidity-window regimes post-2018. Add new decisions there.
- **IPI series.** Seasonally and calendar adjusted, 2021=100, embedded from `EVDS_19-05-2026.xlsx` (`IPI_DATA` dict). Pinned to a fixed vintage so the paper's results are reproducible against TÜİK revisions.
- **Excluded sub-indices.** J103 (post-secondary non-tertiary education) is dropped in Stage 1 because it is discontinued in 2015. J042 (imputed rentals) is dropped because it is sparse. J126 (financial services n.e.c.) survives Stage 1 but is excluded from the PCA in Stage 3 because its σ²_η explodes.
- **RNG seeds.** Gibbs sampler and sign-restriction rejection sampler both seed `np.random.seed(1923)`. Results are bit-reproducible up to the version of `numpy` / `scipy` used.
- **Compute.** Stage 5 (Gibbs) is the slow step — 10,000 iterations × 2 regimes. Reduce `n_iter`/`n_burn` in `src/ms_svar.py` if you need a faster smoke test.

## Citation

If you use this code, please cite the paper. Replication issues / corrections welcome via PR.
