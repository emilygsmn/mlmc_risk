""""Module providing functions for risk aggregation."""

import pandas as pd

__all__ = ["calc_instr_pnls"]

def calc_instr_pnls(prices_at_t1: pd.DataFrame,
                    prices_at_t2: pd.DataFrame
                    ) -> pd.DataFrame:
    """Function calculating the scenario profit-and-loss per instrument."""

    if prices_at_t1.shape != prices_at_t2.shape:
        raise ValueError(
            f"Shape mismatch: df1.shape={prices_at_t1.shape}, df2.shape={prices_at_t2.shape}"
        )

    if not prices_at_t1.columns.equals(prices_at_t2.columns):
        raise ValueError("Column mismatch between DataFrames")

    if not prices_at_t1.index.equals(prices_at_t2.index):
        raise ValueError("Index mismatch between DataFrames")

    return prices_at_t2.subtract(prices_at_t1)
