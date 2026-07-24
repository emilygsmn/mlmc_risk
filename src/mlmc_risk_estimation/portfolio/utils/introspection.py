"""Module providing runtime introspection helper functions."""

import inspect
from collections.abc import Callable
from typing import Any

__all__ = ["get_pricing_arg_spec", "get_pricing_func"]

def get_pricing_arg_spec(module: object = None,
                         prefix: str = "_calc_",
                         suffix: str = "_price"
                         ) -> dict[str, tuple[str, ...]]:
    """Return a dict of all pricing function argument specifications."""

    if module is None:
        from mlmc_risk_estimation import full_valuation
        module = full_valuation

    arg_spec = {}

    for name, obj in inspect.getmembers(module, inspect.isfunction):
        # Pricing functions are identified by the "_calc_..._price" naming convention
        if name.startswith(prefix) and name.endswith(suffix):
            val_tag = name[len(prefix):-len(suffix)]
            arg_names = tuple(inspect.signature(obj).parameters)
            arg_spec[val_tag] = arg_names

    return arg_spec

def get_pricing_func(tag: str,
                     module: object = None
                     ) -> Callable[..., Any] | None:
    """Return the pricing function for a given tag, using the naming convention."""

    if module is None:
        from mlmc_risk_estimation import full_valuation
        module = full_valuation

    func_name = f"_calc_{tag}_price"
    func = getattr(module, func_name, None)

    if func is None:
        raise NotImplementedError(f"No pricing function found for val_tag='{tag}'")

    return func
