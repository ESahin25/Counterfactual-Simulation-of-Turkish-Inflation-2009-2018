"""
Pipeline orchestrator.

Usage:
    python main.py            # run the full pipeline end-to-end
    python main.py --skip 0   # skip stage 0 (e.g. you already pulled the data)
    python main.py --only 5   # run only stage 5 (e.g. iterate on the VAR)

Each stage reads its inputs from data/ and writes its outputs to data/ or
results/. They are idempotent: running stage k twice produces the same output
(modulo RNG state in the Gibbs / sign-restriction stages, which seed np.random).

Figures land in results/figures/; CSV tables land in results/csv/.
"""
import argparse
import time

from src import (
    data_collection,
    build_master,
    tvp_kalman,
    pca_factors,
    ms_taylor,
    ms_svar,
    counterfactual,
    export_csv,
)


STAGES = [
    (0, "Data collection (EVDS / FRED / embedded)", data_collection.run),
    (1, "Build master dataframe & transformations", build_master.run),
    (2, "TVP-Kalman sectoral inertia estimation",  tvp_kalman.run),
    (3, "PCA: extract F_iner and F_imp factors",   pca_factors.run),
    (4, "Markov-switching Taylor rule",            ms_taylor.run),
    (5, "MS-SVAR with sign restrictions",          ms_svar.run),
    (6, "Counterfactual simulation",               counterfactual.run),
    (7, "Export raw data to CSV",                  export_csv.run),
]


def main():
    p = argparse.ArgumentParser(description="Run the inflation-regimes pipeline.")
    p.add_argument("--skip", type=int, nargs="*", default=[],
                   help="Stage numbers to skip.")
    p.add_argument("--only", type=int, nargs="*", default=None,
                   help="If set, run only these stages (in order).")
    args = p.parse_args()

    stages_to_run = [s for s in STAGES
                     if (args.only is None or s[0] in args.only)
                     and s[0] not in args.skip]

    print(f"Will run stages: {[s[0] for s in stages_to_run]}")
    overall = time.time()
    for num, name, fn in stages_to_run:
        print(f"\n{'='*70}\nSTAGE {num}: {name}\n{'='*70}")
        t = time.time()
        fn()
        print(f"  stage {num} done in {time.time() - t:.1f}s")
    print(f"\nAll selected stages complete in {time.time() - overall:.1f}s total.")
    print("Figures: results/figures/   |   Tables: results/csv/")


if __name__ == "__main__":
    main()
