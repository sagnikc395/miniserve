"""Small helpers shared by the benchmark runs."""

import numpy as np


def percentile(values, p) -> float:
    return float(np.percentile(values, p)) if len(values) else 0.0


def per_second(count, elapsed) -> float:
    return count / elapsed if elapsed > 0 else 0.0
