"""Module providing runtime introspection helper functions."""

import inspect
from typing import Any, Callable

__all__ = ["get_pricing_arg_spec", "get_pricing_func"]

def get_pricing_arg_spec(module: object = None,
                         prefix: str = "_calc_",
                         suffix: str = "_price"
                         ) -> dict[str, tuple[str, ...]]:
    """Function returning a dict of all pricing function argument specifications."""

    # Import full_valuation module if not passed as input
    if module is None:
        import full_valuation
        module = full_valuation

    # Initialize dict to be filled
    arg_spec = {}

    # Iterate over all attributes of the module
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        # Only consider functions with the naming pattern
        if name.startswith(prefix) and name.endswith(suffix):
            # Extract val_tag from function name
            val_tag = name[len(prefix):-len(suffix)]

            # Get the argument names
            sig = inspect.signature(obj)
            params = sig.parameters

            # Collect all argument names in a tuple
            arg_names = tuple(p for p in params)

            arg_spec[val_tag] = arg_names

    return arg_spec

def get_pricing_func(tag: str,
                     module: object = None
                     ) -> Callable[..., Any] | None:
    """Function returning the pricing function for a given tag, using the naming convention."""

    # Import full_valuation module if not passed as input
    if module is None:
        import full_valuation
        module = full_valuation

    # Build the function name from the valuation tag
    func_name = f"_calc_{tag}_price"

    # Get the function object from this module
    func = getattr(module, func_name, None)

    # Raise error if function is not available
    if func is None:
        raise NotImplementedError(f"No pricing function found for val_tag='{tag}'")

    return func
