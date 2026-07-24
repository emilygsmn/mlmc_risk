"""The sampler contract shared by every fine/coarse model source.

A "sampler" is a plain dict of callables describing a coupled fine/coarse model
It is the only interface the BL-HD estimator depends on, s.t. the same algorithm can
run unchanged on the portfolio, a bivariate normal, or a bivariate Student-t.
We use:

    "fine"   : n iid fine draws
    "coarse" : n iid coarse (DG) draws
    "level0" : MLMC level 0 (coarse only)
    "level1" : MLMC level 1 (fine & coarse)

The level-1 sampler includes coupling. The fine and coarse model samples share the underlying
randomness, so they are correlated.
"""

from collections.abc import Callable

import numpy as np

__all__ = ["SAMPLER_KEYS", "make_sampler"]

SAMPLER_KEYS = ("fine", "coarse", "level0", "level1")

def make_sampler(fine: Callable[[int], np.ndarray],
                 coarse: Callable[[int], np.ndarray],
                 level0: Callable[[int], tuple[np.ndarray, np.ndarray]],
                 level1: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]]
                 ) -> dict[str, Callable]:
    """Combine the four coupled-model callables into the sampler dict."""
    
    return {"fine": fine, "coarse": coarse, "level0": level0, "level1": level1}
