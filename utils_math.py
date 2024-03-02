import math
import numpy as np
from scipy.stats import beta


def split_integer_into_parts(cnt, ratios):
    ratios = np.array(ratios, dtype=float)

    assert cnt >= 0
    assert (ratios >= 0).all()
    assert ratios.sum() > 0

    # Sort ratios.
    sort_idx = np.argsort(-ratios)
    ratios = ratios[sort_idx]

    # Compute fractions of the remainders.
    ratios_cumsum = np.cumsum(ratios[::-1])[::-1]
    fracs = np.divide(
        ratios, ratios_cumsum, out=np.ones_like(ratios), where=(ratios_cumsum != 0)
    )

    # Split integer into parts.
    remainder = cnt
    parts = np.zeros_like(fracs, dtype=int)
    for i, frac in enumerate(fracs):
        parts[i] = round(remainder * frac)
        remainder -= parts[i]

    assert parts.sum() == cnt

    # Unsort parts.
    parts = parts[np.argsort(sort_idx)]
    return parts


def binom_ci(k, n, confidence):
    alpha = 1 - confidence
    l, u = beta.ppf([alpha/2, 1 - alpha/2], [k, k + 1], [n - k + 1, n - k])
    if math.isnan(l):
        l = 0
    return l, u


def binom_threshold(s, n, acc_diff, confidence):
    low = s
    high = n
    while low < high:
        mid = (low + high) // 2
        acc, _ = binom_ci(mid, n, confidence)
        if acc < 1 - acc_diff:
            low = mid + 1
        else:
            high = mid
    # Using low ensures we get zero probability when it is impossible to satisfy the accuracy.
    return low - s


def binom_threshold_gaussian(s, n, t, z):
    a = n + (z ** 2)
    b = - n * (z ** 2) - 2 * (n ** 2) * t
    c = (n ** 3) * (t ** 2)
    # Quadratic equation.
    k = (- b - ((b ** 2) - 4 * a * c) ** 0.5) / (2 * a)
    # Can be negative which is also valid.
    return k - s


def func_cost(n, a, z, cost_small, cost_large, cost_satisfied, x):
    if math.isclose(x, a, rel_tol=1e-3):
        return (cost_small + cost_large) * n
    k = x * (1 - x) * (z**2) / ((x - a) ** 2)
    if x < a:
        return (cost_small + cost_large) * k + cost_satisfied * (n - k)
    else:
        if k >= n:
            k = n
        return cost_small * n + cost_large * k
