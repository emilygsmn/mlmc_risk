"""Build a HIGH-ACCURACY pseudo-true VaR reference for the portfolio model, once, and cache it to
results/blhd/pseudotrue_accurate.json for portfolio_rmse.py / portfolio_bias_variance.py to load.

Motivation: the pseudo-true was previously an order-statistic quantile from a single 300k fine
sample, whose OWN sampling standard error (~0.2 at alpha=0.95, larger in the deeper tail) is the
same size as the VaR-estimator's RMSE floor. Measuring RMSE/bias against that noisy reference
inflated the apparent 'bias floor' (a single unlucky 300k draw ~0.28 off turned a true RMSE of ~0.14
into ~0.31). A large pooled sample drives the reference SE down ~1/sqrt(N), so it stops contaminating
the RMSE and bias-variance measurements. Requires network (historical-data download).
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import RESULTS_DIR

from mlmc_risk_estimation.blhd.samplers.portfolio import portfolio_sampler
from mlmc_risk_estimation.blhd.baselines import order_statistic_quantile

ALPHAS = (0.9, 0.95, 0.99, 0.995, 0.999)
N_TOTAL = 30_000_000       # pooled fine draws; SE ~0.02 at 0.95, still small in the 0.995 tail
CHUNK = 1_000_000
N_JACK = 20                # split into blocks for a jackknife SE on each Q_full
OUT = RESULTS_DIR / "pseudotrue_accurate.json"


def main() -> None:
    """Build and cache the high-accuracy pseudo-true VaR reference for the portfolio model."""
    s = portfolio_sampler()
    print(f"Drawing {N_TOTAL:,} fine samples for the accurate pseudo-true...", flush=True)
    blocks = []
    for j in range(N_TOTAL // CHUNK):
        np.random.seed(10_000 + j)
        blocks.append(s["fine"](CHUNK))
        if (j + 1) % 5 == 0:
            print(f"  {(j + 1) * CHUNK:,}/{N_TOTAL:,}", flush=True)
    pool = np.concatenate(blocks)

    out = dict(n_total=N_TOTAL, alphas=list(ALPHAS), Q={}, Q_se={})
    # jackknife SE: quantile of each of N_JACK equal blocks, std/sqrt(N_JACK)
    split = np.array_split(pool, N_JACK)
    for a in ALPHAS:
        Q = float(order_statistic_quantile(np.sort(pool), a))
        block_qs = np.array([order_statistic_quantile(np.sort(b), a) for b in split])
        se = float(block_qs.std(ddof=1) / np.sqrt(N_JACK))
        out["Q"][f"{a}"] = Q
        out["Q_se"][f"{a}"] = se
        print(f"  alpha={a}:  Q_full={Q:.4f}  (SE~{se:.4f})", flush=True)

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
