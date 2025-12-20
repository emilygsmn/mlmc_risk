""""Module providing functions for risk aggregation."""

import pandas as pd

__all__ = ["calc_instr_pnls", "calc_portfolio_pnl"]

def calc_instr_pnls(prices_at_t1: pd.DataFrame,
                    prices_at_t2: pd.DataFrame
                    ) -> pd.DataFrame:
    """Function calculating the scenario profit-and-loss per instrument."""

    # Ensure the two DataFrames have the same shape
    if prices_at_t1.shape != prices_at_t2.shape:
        raise ValueError(
            f"Shape mismatch: df1.shape={prices_at_t1.shape}, df2.shape={prices_at_t2.shape}"
        )

    # Ensure the two DataFrames contain the same columns
    if not prices_at_t1.columns.equals(prices_at_t2.columns):
        raise ValueError("Column mismatch between DataFrames")

    # Ensure the two DataFrames contain the same row indices
    if not prices_at_t1.index.equals(prices_at_t2.index):
        raise ValueError("Index mismatch between DataFrames")

    return prices_at_t2.subtract(prices_at_t1)

def calc_portfolio_pnl(instr_pnls: pd.DataFrame
                       ) -> pd.DataFrame:
    """Function calculating the total portfolio scenario profit-and-loss."""

    # Ensure all values in the DataFrame are numeric
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in instr_pnls.dtypes):
        raise ValueError(f"Not all columns in the DataFrame are numeric: {instr_pnls}.")
    
    return instr_pnls.sum(axis=1).to_frame(name="total_pnl")
