"""
my_ops.py  -  your own operators (also where AI-generated operators are saved).

Every function here is usable in expressions exactly like the built-ins, and a
function with the same name as a built-in replaces it.

Naming conventions decide how the compiler feeds data in:
  ts_<name>(x, ..., d=...)   time-series: gets a window, returns today's vector.
                             The window parameter MUST be called d.
  group_<name>(x, group)     per-day group op on vectors.
  anything else              row op: elementwise or per-day cross-sectional;
                             must work on a window (rows x instruments).

Rules: only `np`, no imports, never modify inputs, keep it NaN-safe.
You may use helpers from python_ops.py: _tail, _safe_div, _nanmean0,
_nanstd0, _cs_mean, _cs_std, today, rank, ...

Only exceptions to the conventions need an OP_META entry, e.g. a ts op that
needs one row more than d:  OP_META = {"ts_my_op": {"extra": 1}}
"""

import numpy as np


def ts_up_ratio(x, d=20):
    """Share of the last d days where x rose."""
    w = _tail(x, int(d) + 1)
    if w.shape[0] < 2:
        return np.full(w.shape[1], np.nan)
    chg = np.diff(w, axis=0)
    m = np.isfinite(chg)
    return _safe_div(np.where(m, chg > 0, False).sum(axis=0), m.sum(axis=0))


def ts_efficiency_ratio(x, d=20):
    """Kaufman efficiency: |net change| / sum of |daily changes| over d days.
    Near 1 = clean trend, near 0 = noise."""
    w = _tail(x, int(d) + 1)
    if w.shape[0] < 3:
        return np.full(w.shape[1], np.nan)
    path = np.where(np.isfinite(np.diff(w, axis=0)), np.abs(np.diff(w, axis=0)), 0.0).sum(axis=0)
    return _safe_div(np.abs(w[-1] - w[0]), path)


OP_META = {
    "ts_up_ratio": {"extra": 1},
    "ts_efficiency_ratio": {"extra": 1},
}
