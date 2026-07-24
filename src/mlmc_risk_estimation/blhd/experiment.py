"""Module containing RMSE-vs-cost experiment functions for the BL-HD quantile estimator.

Consists of three tools:

  run_rmse_vs_cost           Reports RMSE + its SE, for the analytical experiments (normal/student-t
                             vary-rho, vary-cost-ratio) and the classic single case.
  run_bias_variance_vs_cost  Reports the bias^2/variance decomposition of the same RMSE^2, with
                             optional bootstrap standard errors on each term (both are bootstrapped).
  run_rmse_vs_cost_portfolio Sequentially conducts the portfolio experiment. The portfolio sampler
                             holds a large precomputed pipeline context that is expensive to rebuild
                             so trials run in-process.

Cost grids and all internal budgets are in c_dg = 1 units. The plotting part later divides by c_f to
show every x-axis in c_f units (full-revaluation-equivalents).
"""

import os
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Callable

import numpy as np

from mlmc_risk_estimation.blhd.adaptive import adapt_tuning_blhd_quantile
from mlmc_risk_estimation.blhd.baselines import harrell_davis, order_statistic_quantile

__all__ = ["run_rmse_vs_cost", "run_bias_variance_vs_cost", "run_rmse_vs_cost_portfolio",
           "run_bias_variance_vs_cost_portfolio", "estimate_pseudo_true_quantile",
           "calibrate_domain", "draw_normal", "draw_studentt", "save_results", "load_results"]

_BASELINE = "__baseline__"


def _rmse_and_se(errors: np.ndarray | list[float], n_trials: int) -> tuple[float, float]:
    """Root-mean-square error and its standard error (delta-method on the MSE)."""
    errors = np.asarray(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    se = np.std(errors ** 2) / (2 * rmse * np.sqrt(n_trials)) if rmse > 0 else 0.0
    return rmse, se


def _bootstrap_bias2_var_se(errors: np.ndarray | list[float], n_boot: int = 2000,
                            rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Bootstrap SEs for bias^2 and variance."""

    if rng is None:
        rng = np.random.default_rng(0)
    errors = np.asarray(errors)
    n = len(errors)

    bias2_boot = np.empty(n_boot)
    var_boot = np.empty(n_boot)

    for b in range(n_boot):
        s = rng.choice(errors, size=n, replace=True)
        bias2_boot[b] = np.mean(s) ** 2
        var_boot[b] = np.var(s, ddof=1) if n > 1 else 0.0

    return bias2_boot.std(), var_boot.std()


# Sampler-draw functions for the baselines.

def draw_normal(n: int, mu: float, sd: float) -> np.ndarray:
    """Draw n i.i.d. normal(mu, sd) samples."""

    return np.random.normal(mu, sd, n)


def draw_studentt(n: int, dof: float, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Draw n i.i.d. location-scale Student-t(dof) samples."""

    return loc + scale * np.random.standard_t(dof, n)


def _bl_trial_worker(args: tuple) -> tuple[str, float, float]:
    """Run one BL-HD trial and return (label, C_star, error)."""

    label, C_star, trial_seed, factory, factory_args, est, alpha, true_q = args
    np.random.seed(trial_seed)
    rng = np.random.default_rng(trial_seed)
    s = factory(*factory_args)
    q_hat, _ = adapt_tuning_blhd_quantile(
        alpha, C_star, s["coarse"], s["level0"], s["level1"],
        est["c_f"], est["c_dg"], est["c_m"], est["S_0"], est["S_1"], est["r"], est["s_mh"],
        K=est.get("K", 40), ess_refresh_frac=est.get("ess_refresh_frac", 0.3),
        rng=rng, verbose=False)
    
    return (label, C_star, q_hat - true_q)


def _baseline_trial_worker(args: tuple) -> tuple[str, float, float, float]:
    """Run one HD/OS fine-only baseline trial."""

    C_star, trial_seed, draw_fn, draw_args, c_f, alpha, true_q = args
    np.random.seed(trial_seed)
    n_fine = max(int(C_star / c_f), 2)
    x = np.sort(draw_fn(n_fine, *draw_args))

    return (_BASELINE, C_star, harrell_davis(x, alpha) - true_q,
            order_statistic_quantile(x, alpha) - true_q)


def _arm_names(bl_arms: list[tuple], baseline: tuple | None) -> list[str]:
    """Series names in report order: baselines (if any) followed by the BL-HD arm labels."""
    names = (["HD (fine)", "OS (fine)"] if baseline is not None else [])
    return names + [label for label, *_ in bl_arms]


def _dg_init_cost(alpha: float, c_dg: float) -> float:
    """Cost of the mandatory delta-gamma kernel-floor draw that adapt_tuning_blhd_quantile makes at
    init, before any adaptive allocation.
    """

    n_dg_floor = int(np.ceil(2.0 / min(alpha, 1 - alpha) - 1))
    return n_dg_floor * c_dg


def _bl_arm_is_feasible(est: dict, alpha: float, C_star: float) -> bool:
    """A BL-HD arm can only satisfy a budget C_star if its mandatory DG investment fits inside it.
    Check this, because at deep alpha and large c_dg the floor alone can exceed C_star."""

    return _dg_init_cost(alpha, est["c_dg"]) <= C_star


def _run_trials_raw(bl_arms: list[tuple], baseline: tuple | None, cost_grid: list[float],
                    alpha: float, true_q: float, n_trials: int, seed: int, n_workers: int | None,
                    verbose: bool) -> tuple[list[float], dict[tuple[str, float], list[float]]]:
    """Dispatch every (cost x arm) trial to a process pool and return the raw per-trial errors.
    """
    cost_grid = list(cost_grid)
    rng_master = np.random.default_rng(seed)

    # Flatten all trials up front, drawing seeds sequentially (baseline first, then each arm) so the
    # run is reproducible regardless of pool completion order.
    tasks = []
    feasible = {}
    for label, *_ in bl_arms:
        feasible[label] = []
    for C_star in cost_grid:
        if baseline is not None:
            draw_fn, draw_args, c_f_base = baseline
            for _ in range(n_trials):
                s = int(rng_master.integers(1 << 30))
                tasks.append((_baseline_trial_worker,
                              (C_star, s, draw_fn, draw_args, c_f_base, alpha, true_q)))
        for label, factory, factory_args, est in bl_arms:
            if not _bl_arm_is_feasible(est, alpha, C_star):
                continue
            feasible[label].append(C_star)
            for _ in range(n_trials):
                s = int(rng_master.integers(1 << 30))
                tasks.append((_bl_trial_worker,
                              (label, C_star, s, factory, factory_args, est, alpha, true_q)))

    if verbose:
        for label, factory, factory_args, est in bl_arms:
            n_skip = len(cost_grid) - len(feasible[label])
            if n_skip:
                print(f"  {label}: skipping {n_skip} infeasible budget(s) "
                      f"(DG kernel floor costs {_dg_init_cost(alpha, est['c_dg']):,.0f})")

    n_total = len(tasks)
    if verbose:
        print(f"Dispatching {n_total:,} independent trials across "
              f"{n_workers or os.cpu_count()} worker process(es)...")

    errs = {("HD (fine)", C): [] for C in cost_grid} if baseline is not None else {}
    if baseline is not None:
        errs.update({("OS (fine)", C): [] for C in cost_grid})
    for label, *_ in bl_arms:
        errs.update({(label, C): [] for C in feasible[label]})

    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(fn, task) for fn, task in tasks]
        for future in as_completed(futures):
            res = future.result()
            n_done += 1
            if res[0] == _BASELINE:
                _, C_star, err_hd, err_os = res
                errs[("HD (fine)", C_star)].append(err_hd)
                errs[("OS (fine)", C_star)].append(err_os)
            else:
                label, C_star, err = res
                errs[(label, C_star)].append(err)
            if verbose and (n_done % max(1, n_total // 20) == 0 or n_done == n_total):
                print(f"  {n_done:>6,}/{n_total:,} trials done "
                      f"({100 * n_done / n_total:5.1f}%)  elapsed={time.time() - t0:.1f}s")
    if verbose:
        print(f"All trials done in {time.time() - t0:.1f}s\n")

    return cost_grid, errs


def run_rmse_vs_cost(bl_arms: list[tuple], baseline: tuple | None, cost_grid: list[float],
                     alpha: float, true_q: float, n_trials: int = 30, seed: int = 0,
                     n_workers: int | None = None, verbose: bool = True) -> dict[str, dict]:
    """Parallel RMSE-vs-cost comparison of BL-HD arms against the HD/OS fine-only baselines."""

    cost_grid, errs = _run_trials_raw(bl_arms, baseline, cost_grid, alpha, true_q, n_trials, seed,
                                      n_workers, verbose)

    results = {name: {"cost": [], "rmse": [], "se": []} for name in _arm_names(bl_arms, baseline)}
    for C_star in cost_grid:
        for name in results:
            # Infeasible (arm, C_star) combos were never dispatched. The arm has no point there,
            # so its cost list is shorter than cost_grid.
            if (name, C_star) not in errs:
                continue
            rmse, se = _rmse_and_se(errs[(name, C_star)], n_trials)
            results[name]["cost"].append(C_star)
            results[name]["rmse"].append(rmse)
            results[name]["se"].append(se)

    return results


def run_bias_variance_vs_cost(bl_arms: list[tuple], baseline: tuple | None, cost_grid: list[float],
                              alpha: float, true_q: float, n_trials: int = 30, seed: int = 0,
                              n_workers: int | None = None, verbose: bool = True,
                              n_boot: int = 2000) -> dict[str, dict]:
    """Parallel bias^2/variance decomposition of RMSE^2 vs. cost (same trial dispatch as
    run_rmse_vs_cost).

    Checking that bias^2 AND variance are both shrinking with cost to see whether the adaptive
    estimator's internal error-budget split (interpolation/smoothing/variance) is well-balanced.
    """
    cost_grid, errs = _run_trials_raw(bl_arms, baseline, cost_grid, alpha, true_q, n_trials, seed,
                                      n_workers, verbose)

    fields = ("cost", "bias", "variance", "rmse", "bias2_se", "variance_se")
    results = {name: {f: [] for f in fields} for name in _arm_names(bl_arms, baseline)}
    for C_star in cost_grid:
        for name in results:
            # Infeasible (arm, C_star) combinationss were never dispatched
            if (name, C_star) not in errs:
                continue
            e = np.asarray(errs[(name, C_star)])
            bias = np.mean(e)
            var = np.var(e, ddof=1)
            rmse = np.sqrt(bias ** 2 + var)
            bias2_se, var_se = ((_bootstrap_bias2_var_se(e, n_boot=n_boot) if n_boot else (None, None)))
            results[name]["cost"].append(C_star)
            results[name]["bias"].append(bias)
            results[name]["variance"].append(var)
            results[name]["rmse"].append(rmse)
            results[name]["bias2_se"].append(bias2_se)
            results[name]["variance_se"].append(var_se)

    return results


def calibrate_domain(fine_sampler: Callable[[int], np.ndarray], n_pilot: int = 200,
                     domain_margin: float = 1.5) -> tuple[float, float, np.ndarray]:
    """CDF/PDF estimation domain [S_0, S_1] from a pilot fine batch's observed range, widened by
    domain_margin (a small pilot can under-sample the tails). Returns (S_0, S_1, fine_pilot)."""
    fine_pilot = fine_sampler(n_pilot)
    lo, hi = fine_pilot.min(), fine_pilot.max()
    span = hi - lo
    return lo - domain_margin * span, hi + domain_margin * span, fine_pilot


def estimate_pseudo_true_quantile(fine_sampler: Callable[[int], np.ndarray], alpha: float,
                                  n_pseudo_true: int = 200_000, seed: int = 0) -> float:
    """Large-sample full-revaluation reference quantile, used as ground truth for RMSE when no
    closed form exists (a real portfolio). `fine_sampler` returns loss samples, so
    the alpha-quantile is the VaR (positive), and no sign correction is needed.

    Uses the order-statistic quantile at a very high number of samples."""
    np.random.seed(seed)
    x = np.sort(fine_sampler(n_pseudo_true))
    return order_statistic_quantile(x, alpha)


def _portfolio_cost_point_errors(sampler: dict, alpha: float, C_star: float, S_0: float, S_1: float,
                                 r: int, c_f: float, c_dg: float, c_m: float, s_mh: float,
                                 true_q: float, n_trials: int, K: int, ess_refresh_frac: float,
                                 rng_master: np.random.Generator,
                                 with_baseline: bool) -> dict[str, list[float]]:
    """Run n_trials of each estimator at one cost point and return the raw per-trial errors
    (q_hat - true_q), keyed by series name. Shared by the RMSE and bias/variance portfolio drivers
    so both draw the same seeds in the same order.
     
    With with_baseline=False, the HD/OS full-revaluation baselines are skipped.
    """

    level0, level1, dg_sampler, fine_sampler = (sampler["level0"], sampler["level1"],
                                                sampler["coarse"], sampler["fine"])
    errs = {"BL-HD": []}
    for _ in range(n_trials):
        trial_seed = int(rng_master.integers(1 << 30))
        np.random.seed(trial_seed)
        rng = np.random.default_rng(trial_seed)
        q_hat, _ = adapt_tuning_blhd_quantile(
            alpha, C_star, dg_sampler, level0, level1, c_f, c_dg, c_m, S_0, S_1, r, s_mh,
            K=K, ess_refresh_frac=ess_refresh_frac, rng=rng, verbose=False)
        errs["BL-HD"].append(q_hat - true_q)

    if with_baseline:
        errs["HD (fine)"], errs["OS (fine)"] = [], []
        n_fine = max(int(C_star / c_f), 2)
        for _ in range(n_trials):
            trial_seed = int(rng_master.integers(1 << 30))
            np.random.seed(trial_seed)
            x = np.sort(fine_sampler(n_fine))
            errs["HD (fine)"].append(harrell_davis(x, alpha) - true_q)
            errs["OS (fine)"].append(order_statistic_quantile(x, alpha) - true_q)
    return errs


def run_rmse_vs_cost_portfolio(sampler: dict, alpha: float, cost_grid: list[float], S_0: float,
                               S_1: float, r: int, c_f: float, c_dg: float, c_m: float, s_mh: float,
                               true_q: float, n_trials: int = 30, K: int = 20,
                               ess_refresh_frac: float = 0.3, seed: int = 0,
                               verbose: bool = True) -> dict[str, dict]:
    """Sequential RMSE-vs-cost comparison for the portfolio sampler."""

    results = {name: {"cost": [], "rmse": [], "se": []}
               for name in ("BL-HD", "HD (fine)", "OS (fine)")}
    rng_master = np.random.default_rng(seed)

    for C_star in cost_grid:
        t0 = time.time()
        errs = _portfolio_cost_point_errors(sampler, alpha, C_star, S_0, S_1, r, c_f, c_dg, c_m,
                                            s_mh, true_q, n_trials, K, ess_refresh_frac, rng_master,
                                            with_baseline=True)
        for name in results:
            rmse, se = _rmse_and_se(errs[name], n_trials)
            results[name]["cost"].append(C_star)
            results[name]["rmse"].append(rmse)
            results[name]["se"].append(se)

        if verbose:
            print(f"C_star={C_star:>14,.0f}  N_fine={max(int(C_star / c_f), 2):>7,}  "
                  f"RMSE: BL-HD={results['BL-HD']['rmse'][-1]:.5f}  "
                  f"HD={results['HD (fine)']['rmse'][-1]:.5f}  "
                  f"OS={results['OS (fine)']['rmse'][-1]:.5f}  ({time.time() - t0:.1f}s)")

    return results


def run_bias_variance_vs_cost_portfolio(sampler: dict, alpha: float, cost_grid: list[float],
                                        S_0: float, S_1: float, r: int, c_f: float, c_dg: float,
                                        c_m: float, s_mh: float, true_q: float, n_trials: int = 30,
                                        K: int = 20, ess_refresh_frac: float = 0.3, seed: int = 0,
                                        verbose: bool = True, n_boot: int = 2000,
                                        with_baseline: bool = True) -> dict[str, dict]:
    """Sequential bias^2/variance decomposition of RMSE^2 vs. cost for the portfolio sampler."""

    fields = ("cost", "bias", "variance", "rmse", "bias2_se", "variance_se")
    names = ["BL-HD"] + (["HD (fine)", "OS (fine)"] if with_baseline else [])
    results = {name: {f: [] for f in fields} for name in names}
    rng_master = np.random.default_rng(seed)

    for C_star in cost_grid:
        t0 = time.time()
        errs = _portfolio_cost_point_errors(sampler, alpha, C_star, S_0, S_1, r, c_f, c_dg, c_m,
                                            s_mh, true_q, n_trials, K, ess_refresh_frac, rng_master,
                                            with_baseline=with_baseline)
        for name in results:
            e = np.asarray(errs[name])
            bias = np.mean(e)
            var = np.var(e, ddof=1)
            bias2_se, var_se = (_bootstrap_bias2_var_se(e, n_boot=n_boot) if n_boot else (None, None))
            results[name]["cost"].append(C_star)
            results[name]["bias"].append(bias)
            results[name]["variance"].append(var)
            results[name]["rmse"].append(np.sqrt(bias ** 2 + var))
            results[name]["bias2_se"].append(bias2_se)
            results[name]["variance_se"].append(var_se)

        if verbose:
            d = results["BL-HD"]
            print(f"C_star={C_star:>14,.0f}  BL-HD: bias={d['bias'][-1]:+.5f} "
                  f"var={d['variance'][-1]:.6f} rmse={d['rmse'][-1]:.5f}  ({time.time() - t0:.1f}s)")

    return results


def _json_default(obj: object) -> object:
    """Convert numpy arrays/scalars to plain Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_results(results: dict, meta: dict, out_dir: str, basename: str) -> tuple[str, str]:
    """Save the results to file JSON and CSV files."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{basename}.json")
    csv_path = os.path.join(out_dir, f"{basename}.csv")
    with open(json_path, "w") as f:
        json.dump({"results": results, "meta": meta}, f, default=_json_default, indent=2)

    fields = [k for d in results.values() for k in d if k != "cost"]
    fields = list(dict.fromkeys(fields))  # de-dup, preserve first-seen order
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "cost", *fields])
        for series, d in results.items():
            for i, cost in enumerate(d["cost"]):
                writer.writerow([series, cost, *(d[f][i] for f in fields)])
    return json_path, csv_path


def load_results(json_path: str) -> tuple[dict, dict]:
    """Read a JSON written by save_results."""
    with open(json_path) as f:
        payload = json.load(f)
    return payload["results"], payload["meta"]
