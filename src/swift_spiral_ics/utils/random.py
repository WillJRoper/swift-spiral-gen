"""Random number generation utilities."""

import numpy as np


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Create a numpy random number generator.

    Args:
        seed: Random seed for reproducibility. If None, uses system entropy.

    Returns:
        NumPy random Generator instance.
    """
    return np.random.default_rng(seed)
