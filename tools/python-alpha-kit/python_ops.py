"""
python_ops.py  -  operator library for BRAIN Python Alphas.

You normally never call these directly: write a Fast Expression style
expression and compiler.py turns it into code, inlining only the operators
it uses. This file is the single source of truth for what they do.

Rules (so functions can be inlined into a self-contained Alpha):
  * use only `np`; no imports, no module-level state used by functions
  * never modify inputs in place (engine arrays are read-only)
  * NaN-aware and warning-free

Kinds (decides how the compiler feeds data in; see OP_META at the bottom)
  ts      window [rows, N] -> today's vector. Window parameter is `d`.
          Name starts with ts_. The compiler handles nesting automatically.
  row     works on whatever rows it gets (elementwise or per-day
          cross-sectional), returns the same rows. Default for other names.
  group   (value, group codes) per day. Name starts with group_.
  series  window -> window along time (e.g. ffill). Declared in OP_META.
  internal  plumbing used by generated code; not callable from expressions.
"""

import numpy as np


# ==========================================================================
# Internal plumbing
# ==========================================================================

def clean(x):
    """Writable float64 copy; int missing sentinel and +/-inf become NaN."""
    a = np.asarray(x)
    if a.dtype.kind == "i":
        out = a.astype(np.float64)
        out[a == np.iinfo(a.dtype).min] = np.nan
        return out
    out = a.astype(np.float64)
    out[~np.isfinite(out)] = np.nan
    return out


def finalize(signal, universe_today):
    """Today's row, inf -> NaN, out-of-universe -> NaN, float32."""
    s = np.asarray(signal, dtype=np.float64)
    if s.ndim == 0:
        s = np.full(np.asarray(universe_today).shape, float(s))
    if s.ndim == 2:
        s = s[-1]
    s = s.copy()
    s[~np.isfinite(s)] = np.nan
    s[~np.asarray(universe_today).astype(bool)] = np.nan
    return s.astype(np.float32)


def today(x):
    a = np.asarray(x, dtype=np.float64)
    return a[-1] if a.ndim == 2 else a


def adjust_per_share(x, adjfactor, inverse=False):
    """Split-adjust a window with the adjfactor field, rebased so today's value
    is unchanged. inverse=True for share counts (volume, sharesout)."""
    a = _as2d(clean(adjfactor))
    a = np.where(np.isfinite(a), a, 1.0)
    cum = np.cumsum(a - 1.0, axis=0) + 1.0
    xa = _as2d(x)
    rows = min(xa.shape[0], cum.shape[0])
    xa, cum = xa[-rows:], cum[-rows:]
    if inverse:
        return xa * cum / cum[-1]
    return xa / cum * cum[-1]


def _as2d(x):
    a = np.asarray(x, dtype=np.float64)
    return a[np.newaxis, :] if a.ndim == 1 else a


def _tail(x, d):
    a = _as2d(x)
    return a[-max(int(d), 1):]


def _align(*args):
    rows = [a.shape[0] for a in args if isinstance(a, np.ndarray) and a.ndim == 2]
    if not rows:
        return list(args)
    r = min(rows)
    return [a[-r:] if (isinstance(a, np.ndarray) and a.ndim == 2) else a for a in args]


def call_row(op, *args, **kw):
    """Call a row op after aligning all windows to their common last rows."""
    return op(*_align(*args), **kw)


def rows_apply(op, *args, **kw):
    """Apply a per-day (vector) op to each aligned row of the windows."""
    args = _align(*args)
    rows = next((a.shape[0] for a in args if isinstance(a, np.ndarray) and a.ndim == 2), None)
    if rows is None:
        return _as2d(op(*args, **kw))
    out = [op(*[a[r] if (isinstance(a, np.ndarray) and a.ndim == 2) else a for a in args], **kw)
           for r in range(rows)]
    return np.vstack(out)


def roll(op, length, *args, **kw):
    """Last `length` daily values of a ts op, as a window (oldest first)."""
    args = _align(*args)
    T = next((a.shape[0] for a in args if isinstance(a, np.ndarray) and a.ndim == 2), 1)
    L = max(1, min(int(length), T))
    rows = []
    for k in range(L - 1, -1, -1):
        end = T - k
        sl = [a[:end] if (isinstance(a, np.ndarray) and a.ndim == 2) else a for a in args]
        rows.append(op(*sl, **kw))
    return np.vstack(rows)


def _nanmean0(w):
    m = np.isfinite(w)
    cnt = m.sum(axis=0)
    tot = np.where(m, w, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = tot / cnt
    out = np.asarray(out, dtype=np.float64)
    out[cnt == 0] = np.nan
    return out


def _nanstd0(w, ddof=0):
    m = np.isfinite(w)
    cnt = m.sum(axis=0)
    dev = np.where(m, w - _nanmean0(w), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt((dev * dev).sum(axis=0) / (cnt - ddof))
    out = np.asarray(out, dtype=np.float64)
    out[cnt - ddof <= 0] = np.nan
    return out


def _nanmoment0(w, k):
    m = np.isfinite(w)
    dev = np.where(m, w - _nanmean0(w), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (dev ** k).sum(axis=0) / m.sum(axis=0)


def _nanquantile0(w, q):
    s = np.sort(np.where(np.isfinite(w), w, np.inf), axis=0)
    cnt = np.isfinite(w).sum(axis=0)
    pos = q * np.maximum(cnt - 1, 0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(cnt - 1, 0))
    cols = np.arange(w.shape[1])
    vlo, vhi = s[lo, cols], s[hi, cols]
    with np.errstate(invalid="ignore"):
        out = np.where(np.isfinite(vhi), vlo + (vhi - vlo) * (pos - lo), vlo)
    out = np.asarray(out, dtype=np.float64)
    out[cnt == 0] = np.nan
    return out


def _cs_mean(x):
    m = np.isfinite(x)
    cnt = m.sum(axis=-1, keepdims=True)
    tot = np.where(m, x, 0.0).sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = tot / cnt
    return np.where(cnt == 0, np.nan, out)


def _cs_std(x):
    m = np.isfinite(x)
    cnt = m.sum(axis=-1, keepdims=True)
    dev = np.where(m, x - _cs_mean(x), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt((dev * dev).sum(axis=-1, keepdims=True) / cnt)
    return np.where(cnt == 0, np.nan, out)


def _safe_div(a, b):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out = np.asarray(np.asarray(a, dtype=np.float64) / b, dtype=np.float64)
    out = np.array(out, dtype=np.float64)
    out[~np.isfinite(out)] = np.nan
    return out


def _f(x):
    return np.asarray(x, dtype=np.float64)


def _nan_where_any(out, *xs):
    out = np.array(out, dtype=np.float64)
    bad = np.zeros(out.shape, dtype=bool)
    for x in xs:
        bad |= ~np.isfinite(np.broadcast_to(_f(x), out.shape))
    out[bad] = np.nan
    return out


def _norm_ppf(p):
    """Inverse standard normal CDF (Acklam approximation)."""
    p = np.clip(_f(p), 1e-9, 1 - 1e-9)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    out = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    out[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = p[mid] - 0.5
    r = q * q
    out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return out


def _fft_lowpass(w, keep):
    a = ffill(w)
    T = a.shape[0]
    if T < 4:
        return a
    a = np.where(np.isfinite(a), a, _nanmean0(a))
    a = np.where(np.isfinite(a), a, 0.0)
    t = np.arange(T, dtype=np.float64)[:, None]
    tc = t - t.mean()
    slope = (tc * (a - a.mean(axis=0))).sum(axis=0) / (tc * tc).sum()
    trend = a.mean(axis=0) + slope * tc
    spec = np.fft.rfft(a - trend, axis=0)
    spec[int(keep) + 1:] = 0
    return np.fft.irfft(spec, n=T, axis=0) + trend


# ==========================================================================
# Arithmetic (elementwise)
# ==========================================================================

def add(*args, filter=False):
    """Sum of inputs. filter=True treats NaN as 0."""
    tot = 0.0
    for a in args:
        a = _f(a)
        tot = tot + (np.where(np.isfinite(a), a, 0.0) if filter else a)
    return _f(tot)


def subtract(x, y, filter=False):
    """x - y. filter=True treats NaN as 0."""
    x, y = _f(x), _f(y)
    if filter:
        x, y = np.where(np.isfinite(x), x, 0.0), np.where(np.isfinite(y), y, 0.0)
    return x - y


def multiply(*args, filter=False):
    """Product of inputs. filter=True treats NaN as 1."""
    tot = 1.0
    for a in args:
        a = _f(a)
        tot = tot * (np.where(np.isfinite(a), a, 1.0) if filter else a)
    return _f(tot)


def divide(x, y):
    """x / y; division by 0 gives NaN."""
    return _safe_div(x, y)


def neg(x):
    """-x (FE reverse)."""
    return -_f(x)


def inverse(x):
    """1 / x."""
    return _safe_div(1.0, x)


def op_abs(x):
    """Absolute value (FE abs)."""
    return np.abs(_f(x))


def sign(x):
    """-1, 0, +1."""
    return np.sign(_f(x))


def op_log(x):
    """Natural log; NaN for x <= 0 (FE log)."""
    x = _f(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.log(np.where(x > 0, x, np.nan))
    return out


def signed_log(x):
    """sign(x) * log(1 + |x|): tames heavy tails, keeps sign."""
    x = _f(x)
    return np.sign(x) * np.log1p(np.abs(x))


def sqrt(x):
    """Square root; NaN for negatives."""
    x = _f(x)
    return np.sqrt(np.where(x >= 0, x, np.nan))


def power(x, e):
    """x ** e (NaN where undefined)."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out = np.power(_f(x), e)
    return _safe_div(out, 1.0)


def signed_power(x, e):
    """sign(x) * |x| ** e."""
    x = _f(x)
    return np.sign(x) * np.abs(x) ** e


def exp(x):
    """e ** x (input clipped to avoid overflow)."""
    return np.exp(np.clip(_f(x), -50, 50))


def op_max(*args):
    """Elementwise max (FE max)."""
    out = _f(args[0])
    for a in args[1:]:
        out = np.maximum(out, _f(a))
    return out


def op_min(*args):
    """Elementwise min (FE min)."""
    out = _f(args[0])
    for a in args[1:]:
        out = np.minimum(out, _f(a))
    return out


def clip(x, lower=-1.0, upper=1.0):
    """Clamp into [lower, upper]."""
    return np.clip(_f(x), lower, upper)


def tanh(x):
    """Hyperbolic tangent squash."""
    return np.tanh(_f(x))


def sigmoid(x):
    """Logistic squash to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(_f(x), -50, 50)))


# ==========================================================================
# Logic (1.0 / 0.0, NaN where an input is NaN)
# ==========================================================================

def gt(x, y):
    """x > y."""
    with np.errstate(invalid="ignore"):
        return _nan_where_any(_f(x) > _f(y), x, y)


def lt(x, y):
    """x < y."""
    with np.errstate(invalid="ignore"):
        return _nan_where_any(_f(x) < _f(y), x, y)


def ge(x, y):
    """x >= y."""
    with np.errstate(invalid="ignore"):
        return _nan_where_any(_f(x) >= _f(y), x, y)


def le(x, y):
    """x <= y."""
    with np.errstate(invalid="ignore"):
        return _nan_where_any(_f(x) <= _f(y), x, y)


def eq(x, y):
    """x == y."""
    return _nan_where_any(_f(x) == _f(y), x, y)


def ne(x, y):
    """x != y."""
    return _nan_where_any(_f(x) != _f(y), x, y)


def and_(x, y):
    """Logical and (FE and / &&)."""
    return _nan_where_any((_f(x) > 0) & (_f(y) > 0), x, y)


def or_(x, y):
    """Logical or (FE or / ||)."""
    return _nan_where_any((_f(x) > 0) | (_f(y) > 0), x, y)


def not_(x):
    """Logical not (FE not / !)."""
    return _nan_where_any(~(_f(x) > 0), x)


def if_else(cond, a, b):
    """a where cond > 0 else b (FE cond ? a : b). NaN cond -> NaN."""
    c = _f(cond)
    out = _f(np.where(c > 0, _f(a), _f(b)))
    out = np.array(np.broadcast_to(out, np.broadcast(c, out).shape), dtype=np.float64)
    out[~np.isfinite(np.broadcast_to(c, out.shape))] = np.nan
    return out


def is_nan(x):
    """1 where x is NaN, else 0."""
    return (~np.isfinite(_f(x))).astype(np.float64)


# ==========================================================================
# Cleaning / prep (also usable inside expressions)
# ==========================================================================

def fill_zero(x):
    """NaN -> 0 (for event data where missing means 'nothing happened')."""
    x = _f(x)
    return np.where(np.isfinite(x), x, 0.0)


def fill_value(x, value=0.0):
    """NaN -> value."""
    x = _f(x)
    return np.where(np.isfinite(x), x, value)


def purify(x):
    """+/-inf -> NaN."""
    return _safe_div(x, 1.0)


def to_nan(x, value=0.0):
    """value -> NaN (e.g. treat exact zeros as missing)."""
    x = _f(x)
    return np.where(x == value, np.nan, x)


def ffill(x, limit=0):
    """Forward-fill along time (limit=0 means unlimited within the window)."""
    a = _as2d(x).copy()
    run = np.zeros(a.shape[1])
    for t in range(1, a.shape[0]):
        miss = ~np.isfinite(a[t])
        run = np.where(miss, run + 1, 0)
        fill = miss & np.isfinite(a[t - 1])
        if limit:
            fill &= run <= limit
        a[t, fill] = a[t - 1, fill]
    return a


def coverage_mask(x, d=60, min_frac=0.5):
    """NaN where the trailing d-day share of valid values is below min_frac."""
    a = _as2d(x)
    d = int(d)
    c = np.cumsum(np.isfinite(a), axis=0).astype(np.float64)
    lagged = np.vstack([np.zeros((d, a.shape[1])), c])[:a.shape[0]]
    n = np.minimum(np.arange(1, a.shape[0] + 1), d)[:, None]
    return np.where((c - lagged) / n >= min_frac, a, np.nan)


# ==========================================================================
# Cross-sectional (per day)
# ==========================================================================

def _rank1d(v):
    out = np.full(v.shape, np.nan)
    m = np.isfinite(v)
    n = int(m.sum())
    if n == 0:
        return out
    if n == 1:
        out[m] = 0.5
        return out
    vv = v[m]
    s = np.sort(vv)
    lo = np.searchsorted(s, vv, side="left")
    hi = np.searchsorted(s, vv, side="right")
    out[m] = (lo + hi - 1) / 2.0 / (n - 1)
    return out


def rank(x):
    """Cross-sectional rank 0..1, ties averaged."""
    a = _f(x)
    if a.ndim == 1:
        return _rank1d(a)
    return np.vstack([_rank1d(r) for r in a])


def zscore(x):
    """Cross-sectional z-score."""
    a = _f(x)
    return _safe_div(a - _cs_mean(a), _cs_std(a))


def demean(x):
    """Subtract the cross-sectional mean."""
    a = _f(x)
    return a - _cs_mean(a)


def normalize(x, useStd=False, limit=0.0):
    """FE normalize: demean, optionally divide by std, optionally clip at +/-limit."""
    a = demean(x)
    if useStd:
        a = _safe_div(a, _cs_std(_f(x)))
    if limit:
        a = np.clip(a, -limit, limit)
    return a


def scale(x, scale=1.0, longscale=0.0, shortscale=0.0):
    """Scale so sum(|x|) = scale; or longs to longscale and shorts to shortscale."""
    a = _f(x)
    if longscale or shortscale:
        pos = np.where(a > 0, a, 0.0)
        negp = np.where(a < 0, -a, 0.0)
        ps = pos.sum(axis=-1, keepdims=True)
        ns = negp.sum(axis=-1, keepdims=True)
        out = _safe_div(pos * (longscale or 1.0), ps) - _safe_div(negp * (shortscale or 1.0), ns)
        return np.where(np.isfinite(a), np.nan_to_num(out), np.nan)
    tot = np.where(np.isfinite(a), np.abs(a), 0.0).sum(axis=-1, keepdims=True)
    return _safe_div(a * scale, tot)


def winsorize(x, std=4.0):
    """Clip to mean +/- std * sd."""
    a = _f(x)
    mu, sd = _cs_mean(a), _cs_std(a)
    return np.clip(a, mu - std * sd, mu + std * sd)


def truncate(x, maxPercent=0.01):
    """Cap each |value| at maxPercent of the day's total |x|."""
    a = _f(x)
    cap = maxPercent * np.where(np.isfinite(a), np.abs(a), 0.0).sum(axis=-1, keepdims=True)
    return np.clip(a, -cap, cap)


def quantile(x, driver="gaussian", sigma=1.0):
    """Rank then map to a distribution: gaussian | uniform | cauchy."""
    r = rank(x)
    p = 0.001 + 0.998 * r
    if driver == "uniform":
        return (p - 0.5) * 2 * sigma
    if driver == "cauchy":
        return np.tan(np.pi * (p - 0.5)) * sigma
    out = _norm_ppf(np.where(np.isfinite(p), p, 0.5)) * sigma
    return np.where(np.isfinite(r), out, np.nan)


def regression_neut(y, x):
    """Residual of y after a cross-sectional regression on x (per day)."""
    y, x = _f(y), _f(x)
    m = np.isfinite(y) & np.isfinite(x)
    n = m.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.where(m, x, 0.0).sum(axis=-1, keepdims=True) / n
        my = np.where(m, y, 0.0).sum(axis=-1, keepdims=True) / n
        dx, dy = np.where(m, x - mx, 0.0), np.where(m, y - my, 0.0)
        beta = (dx * dy).sum(axis=-1, keepdims=True) / (dx * dx).sum(axis=-1, keepdims=True)
    out = y - (my + beta * (x - mx))
    return np.where(m, out, np.nan)


def vector_neut(x, y):
    """x minus its projection on y (per day, no intercept)."""
    x, y = _f(x), _f(y)
    m = np.isfinite(x) & np.isfinite(y)
    xm, ym = np.where(m, x, 0.0), np.where(m, y, 0.0)
    k = _safe_div((xm * ym).sum(axis=-1, keepdims=True), (ym * ym).sum(axis=-1, keepdims=True))
    return np.where(m, x - np.nan_to_num(k) * y, np.nan)


def bucket(x, n=10, range=""):
    """Bucket codes for grouping. Default: n quantile buckets. FE style
    range="lo,hi,step" cuts the values of x (usually a rank) at fixed edges."""
    a = _f(x)
    if range:
        lo, hi, st = [float(v) for v in str(range).split(",")]
        edges = np.arange(lo, hi + 1e-12, st)
        codes = np.digitize(a, edges).astype(np.float64)
        return np.where(np.isfinite(a), codes, np.nan)
    r = rank(a)
    return np.where(np.isfinite(r), np.minimum(np.floor(r * n), n - 1), np.nan)


def tail(x, lower=0.0, upper=0.0, newval=0.0):
    """Replace values inside [lower, upper] with newval."""
    a = _f(x)
    return np.where((a >= lower) & (a <= upper), newval, a)


# ==========================================================================
# Group (per day; group codes are any numbers, NaN = no group)
# Note: GROUP-type fields may not load in Python Alphas. Build groups with
# bucket(rank(cap), 10), cluster_neutralize(...) or `market`.
# ==========================================================================

def _group_codes(g):
    g = _f(g)
    codes = np.full(g.shape, -1, dtype=np.int64)
    ok = np.isfinite(g)
    if ok.any():
        _, inv = np.unique(g[ok], return_inverse=True)
        codes[ok] = inv.ravel()
    return codes


def _group_stat(x, c, vals):
    k = int(c.max()) + 1 if (c >= 0).any() else 0
    out = np.full(x.shape, np.nan)
    if k == 0:
        return out
    m = np.isfinite(vals) & (c >= 0)
    s = np.bincount(c[m], weights=vals[m], minlength=k)
    n = np.bincount(c[m], minlength=k)
    gm = np.full(k, np.nan)
    gm[n > 0] = s[n > 0] / n[n > 0]
    has = c >= 0
    out[has] = gm[c[has]]
    return out


def group_mean(x, group):
    """Mean of x within each group."""
    x = today(x)
    g = np.broadcast_to(today(group), x.shape)
    return _group_stat(x, _group_codes(g), x)


def group_neutralize(x, group):
    """x minus its group mean."""
    return today(x) - group_mean(x, group)


def group_zscore(x, group):
    """z-score within each group."""
    x = today(x)
    dev = x - group_mean(x, group)
    return _safe_div(dev, np.sqrt(group_mean(dev * dev, group)))


def group_rank(x, group):
    """Rank 0..1 within each group."""
    x = today(x)
    c = _group_codes(np.broadcast_to(today(group), x.shape))
    out = np.full(x.shape, np.nan)
    for k in np.unique(c[c >= 0]):
        idx = np.where(c == k)[0]
        out[idx] = _rank1d(x[idx])
    return out


def group_scale(x, group):
    """x / sum(|x|) within each group."""
    x = today(x)
    c = _group_codes(np.broadcast_to(today(group), x.shape))
    mean_abs = _group_stat(x, c, np.abs(x))
    return _safe_div(x, mean_abs * _group_count_raw(x, c))


def _group_count_raw(x, c):
    k = int(c.max()) + 1 if (c >= 0).any() else 0
    out = np.full(x.shape, np.nan)
    if k == 0:
        return out
    m = np.isfinite(x) & (c >= 0)
    n = np.bincount(c[m], minlength=k).astype(np.float64)
    has = c >= 0
    out[has] = n[c[has]]
    return out


def group_count(x, group):
    """Number of valid values in each instrument's group."""
    x = today(x)
    return _group_count_raw(x, _group_codes(np.broadcast_to(today(group), x.shape)))


# ==========================================================================
# Time-series (window -> today). Window parameter is always `d`.
# ==========================================================================

def ts_delay(x, d=1):
    """Value d days ago."""
    a = _as2d(x)
    d = int(d)
    if a.shape[0] <= d:
        return np.full(a.shape[1], np.nan)
    return a[-1 - d].copy()


def ts_delta(x, d=1):
    """Today minus d days ago."""
    return _as2d(x)[-1] - ts_delay(x, d)


def ts_returns(x, d=1):
    """x / x d days ago - 1."""
    return _safe_div(_as2d(x)[-1], ts_delay(x, d)) - 1.0


def ts_mean(x, d=20):
    """Mean over d days."""
    return _nanmean0(_tail(x, d))


def ts_sum(x, d=20):
    """Sum over d days."""
    w = _tail(x, d)
    m = np.isfinite(w)
    out = np.where(m, w, 0.0).sum(axis=0)
    out[m.sum(axis=0) == 0] = np.nan
    return out


def ts_product(x, d=20):
    """Product over d days."""
    w = _tail(x, d)
    m = np.isfinite(w)
    out = np.where(m, w, 1.0).prod(axis=0)
    out[m.sum(axis=0) == 0] = np.nan
    return out


def ts_std(x, d=20):
    """Standard deviation over d days (FE ts_std_dev)."""
    return _nanstd0(_tail(x, d))


def ts_var(x, d=20):
    """Variance over d days."""
    return _nanstd0(_tail(x, d)) ** 2


def ts_median(x, d=20):
    """Median over d days."""
    return _nanquantile0(_tail(x, d), 0.5)


def ts_percentile(x, d=20, q=0.5):
    """q-th quantile (0..1) of the last d days."""
    return _nanquantile0(_tail(x, d), q)


def ts_min(x, d=20):
    """Minimum over d days."""
    w = _tail(x, d)
    out = np.where(np.isfinite(w), w, np.inf).min(axis=0)
    out[~np.isfinite(out)] = np.nan
    return out


def ts_max(x, d=20):
    """Maximum over d days."""
    w = _tail(x, d)
    out = np.where(np.isfinite(w), w, -np.inf).max(axis=0)
    out[~np.isfinite(out)] = np.nan
    return out


def ts_range(x, d=20):
    """Max minus min over d days."""
    return ts_max(x, d) - ts_min(x, d)


def ts_max_diff(x, d=20):
    """Today minus the d-day max."""
    return _as2d(x)[-1] - ts_max(x, d)


def ts_min_diff(x, d=20):
    """Today minus the d-day min."""
    return _as2d(x)[-1] - ts_min(x, d)


def ts_drawdown(x, d=60):
    """Today / d-day max - 1 (how far below the recent peak)."""
    return _safe_div(_as2d(x)[-1], ts_max(x, d)) - 1.0


def ts_zscore(x, d=20):
    """(today - mean) / std over d days."""
    w = _tail(x, d)
    return _safe_div(w[-1] - _nanmean0(w), _nanstd0(w))


def ts_av_diff(x, d=20):
    """Today minus the d-day mean."""
    w = _tail(x, d)
    return w[-1] - _nanmean0(w)


def ts_scale(x, d=20):
    """(today - min) / (max - min) over d days."""
    w = _tail(x, d)
    lo, hi = ts_min(w, d), ts_max(w, d)
    return _safe_div(w[-1] - lo, hi - lo)


def ts_rank(x, d=20, constant=0.0):
    """Where today sits in the last d days, 0..1, ties averaged."""
    w = _tail(x, d)
    cur = w[-1]
    m = np.isfinite(w)
    valid = m.sum(axis=0)
    less = (np.where(m, w, np.inf) < cur).sum(axis=0)
    eq_ = (np.where(m, w, np.nan) == cur).sum(axis=0) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (less + eq_ / 2.0) / (valid - 1)
    out = np.asarray(out, dtype=np.float64)
    out[~np.isfinite(cur) | (valid < 2)] = np.nan
    return out + constant


def ts_quantile(x, d=20, driver="gaussian"):
    """ts_rank mapped to a gaussian (or uniform) distribution."""
    r = ts_rank(x, d)
    p = 0.001 + 0.998 * r
    if driver == "uniform":
        return (p - 0.5) * 2
    out = _norm_ppf(np.where(np.isfinite(p), p, 0.5))
    return np.where(np.isfinite(r), out, np.nan)


def ts_arg_max(x, d=20):
    """Days since the d-day max (0 = today)."""
    w = _tail(x, d)
    out = (w.shape[0] - 1 - np.where(np.isfinite(w), w, -np.inf).argmax(axis=0)).astype(np.float64)
    out[~np.isfinite(w).any(axis=0)] = np.nan
    return out


def ts_arg_min(x, d=20):
    """Days since the d-day min (0 = today)."""
    w = _tail(x, d)
    out = (w.shape[0] - 1 - np.where(np.isfinite(w), w, np.inf).argmin(axis=0)).astype(np.float64)
    out[~np.isfinite(w).any(axis=0)] = np.nan
    return out


def ts_decay_linear(x, d=10):
    """Linearly weighted mean (newest weight d, oldest 1)."""
    w = _tail(x, d)
    rows = w.shape[0]
    wt = np.arange(int(d) - rows + 1, int(d) + 1, dtype=np.float64)[:, None]
    m = np.isfinite(w)
    return _safe_div(np.where(m, w * wt, 0.0).sum(axis=0), np.where(m, wt, 0.0).sum(axis=0))


def ts_decay_exp(x, d=20, factor=0.9):
    """Exponentially weighted mean (FE ts_decay_exp_window)."""
    w = _tail(x, d)
    rows = w.shape[0]
    wt = (factor ** np.arange(rows - 1, -1, -1, dtype=np.float64))[:, None]
    m = np.isfinite(w)
    return _safe_div(np.where(m, w * wt, 0.0).sum(axis=0), np.where(m, wt, 0.0).sum(axis=0))


def ts_count_nans(x, d=20):
    """Number of NaN days in the last d."""
    return (~np.isfinite(_tail(x, d))).sum(axis=0).astype(np.float64)


def ts_count(x, d=20):
    """Number of valid days in the last d."""
    return np.isfinite(_tail(x, d)).sum(axis=0).astype(np.float64)


def ts_backfill(x, d=250):
    """Most recent non-NaN value within the last d days."""
    w = _tail(x, d)
    out = w[-1].copy()
    for row in w[-2::-1]:
        miss = ~np.isfinite(out)
        if not miss.any():
            break
        out[miss] = row[miss]
    return out


def ts_skew(x, d=60):
    """Skewness over d days."""
    w = _tail(x, d)
    return _safe_div(_nanmoment0(w, 3), _nanmoment0(w, 2) ** 1.5)


def ts_kurt(x, d=60):
    """Excess kurtosis over d days."""
    w = _tail(x, d)
    return _safe_div(_nanmoment0(w, 4), _nanmoment0(w, 2) ** 2) - 3.0


def ts_mad(x, d=20):
    """Mean absolute deviation over d days."""
    w = _tail(x, d)
    return _nanmean0(np.abs(w - _nanmean0(w)))


def ts_ir(x, d=20):
    """Mean / std over d days (information-ratio style)."""
    w = _tail(x, d)
    return _safe_div(_nanmean0(w), _nanstd0(w))


def _pair_stats(x, y, d):
    wx, wy = _tail(x, d), _tail(y, d)
    r = min(wx.shape[0], wy.shape[0])
    wx, wy = wx[-r:], wy[-r:]
    m = np.isfinite(wx) & np.isfinite(wy)
    n = m.sum(axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.where(m, wx, 0.0).sum(axis=0) / n
        my = np.where(m, wy, 0.0).sum(axis=0) / n
        dx = np.where(m, wx - mx, 0.0)
        dy = np.where(m, wy - my, 0.0)
        cov = (dx * dy).sum(axis=0) / n
        vx = (dx * dx).sum(axis=0) / n
        vy = (dy * dy).sum(axis=0) / n
    return wx, wy, n, mx, my, cov, vx, vy


def ts_corr(x, y, d=20):
    """Correlation of x and y over d days."""
    _, _, n, _, _, cov, vx, vy = _pair_stats(x, y, d)
    out = _safe_div(cov, np.sqrt(vx * vy))
    out[n < 3] = np.nan
    return out


def ts_covariance(x, y, d=20):
    """Covariance of x and y over d days."""
    _, _, n, _, _, cov, _, _ = _pair_stats(x, y, d)
    out = np.array(cov, dtype=np.float64)
    out[n < 2] = np.nan
    return out


def ts_regression(y, x, d=60, rettype=0):
    """Regress y on x over d days. rettype 0 residual, 1 intercept, 2 slope, 3 R^2."""
    wx, wy, n, mx, my, cov, vx, vy = _pair_stats(x, y, d)
    beta = _safe_div(cov, vx)
    a0 = my - beta * mx
    if rettype == 1:
        out = a0
    elif rettype == 2:
        out = beta
    elif rettype == 3:
        out = _safe_div(cov * cov, vx * vy)
    else:
        out = wy[-1] - (a0 + beta * wx[-1])
    out = np.array(out, dtype=np.float64)
    out[n < 3] = np.nan
    return out


def ts_beta(y, x, d=60):
    """Slope of y on x over d days."""
    return ts_regression(y, x, d, 2)


def ts_trend(x, d=20):
    """Slope of x against time over d days (per day)."""
    w = _tail(x, d)
    t = np.broadcast_to(np.arange(w.shape[0], dtype=np.float64)[:, None], w.shape)
    return ts_regression(w, t, d, 2)


def ts_autocorr(x, d=20, lag=1):
    """Correlation of x with itself `lag` days earlier, over d days."""
    w = _tail(x, int(d) + int(lag))
    if w.shape[0] <= int(lag) + 2:
        return np.full(w.shape[1], np.nan)
    return ts_corr(w[int(lag):], w[:-int(lag)], d)


def days_from_last_change(x, d=250):
    """Days since x last changed (within d days)."""
    w = _tail(x, d)
    cur = w[-1]
    out = np.zeros(w.shape[1])
    same = np.isfinite(cur)
    for k in range(2, w.shape[0] + 1):
        row = w[-k]
        same = same & (row == cur)
        out = out + same
    out[~np.isfinite(cur)] = np.nan
    return out


def last_diff_value(x, d=250):
    """Most recent value (within d days) that differs from today's."""
    w = _tail(x, d)
    cur = w[-1]
    out = np.full(w.shape[1], np.nan)
    for row in w[-2::-1]:
        hit = np.isnan(out) & np.isfinite(row) & (row != cur)
        out[hit] = row[hit]
    return out


def ts_fft_smooth(x, d=60, keep=4):
    """Today's value of a Fourier low-pass (keep lowest frequencies) of d days."""
    return _fft_lowpass(_tail(x, d), keep)[-1]


def ts_pca_neutralize(x, r, d=60, k=3):
    """Remove today's x exposure to the top-k principal components of r's
    last d days (statistical risk factors, e.g. r = returns)."""
    xv = today(x)
    w = _tail(r, d)
    z = np.where(np.isfinite(w), w - _nanmean0(w), 0.0)
    sd = np.sqrt((z * z).mean(axis=0))
    z = _safe_div(z, sd)
    z = np.where(np.isfinite(z), z, 0.0)
    if z.shape[0] < int(k) + 2:
        return np.full(xv.shape, np.nan)
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    load = vt[: int(k)].T
    ok = np.isfinite(xv)
    out = np.full(xv.shape, np.nan)
    if ok.sum() <= int(k) + 1:
        return out
    A = np.column_stack([np.ones(ok.sum()), load[ok]])
    coef = np.linalg.lstsq(A, xv[ok], rcond=None)[0]
    out[ok] = xv[ok] - A @ coef
    return out


# ==========================================================================
# Stateful steps (the compiler wires these to `store`; use hump(...),
# ema(...), trade_when(...), cluster_neutralize(...) in expressions)
# ==========================================================================

def hump_step(x, prev, hump=0.01):
    x = today(x)
    if prev is None:
        return x.copy()
    prev = _f(prev)
    thr = hump * np.where(np.isfinite(x), np.abs(x), 0.0).sum()
    keep = np.isfinite(prev) & np.isfinite(x) & (np.abs(x - prev) <= thr)
    return np.where(keep, prev, x)


def ema_step(x, prev, a=0.1):
    x = today(x)
    if prev is None:
        return x.copy()
    prev = _f(prev)
    upd = np.where(np.isfinite(prev), a * x + (1 - a) * prev, x)
    return np.where(np.isfinite(x), upd, prev)


def trade_when_step(trigger, x, exit_cond, prev):
    x = today(x)
    out = np.full(x.shape, np.nan) if prev is None else _f(prev).copy()
    trig = np.broadcast_to(np.nan_to_num(today(trigger), nan=0.0) > 0, x.shape)
    ex = np.broadcast_to(np.nan_to_num(today(exit_cond), nan=0.0) > 0, x.shape)
    out[trig] = x[trig]
    out[ex] = np.nan
    return out


def kmeans_labels(features, k=20, iters=15, seed=0, min_valid=0.6):
    f = _f(features)
    ok = np.isfinite(f).mean(axis=1) >= min_valid
    labels = np.full(f.shape[0], np.nan)
    if ok.sum() < k:
        return labels
    z = f[ok]
    z = np.where(np.isfinite(z), z - _nanmean0(z.T)[:, None], 0.0)
    z = _safe_div(z, np.sqrt((z * z).mean(axis=1, keepdims=True)))
    z = np.where(np.isfinite(z), z, 0.0)
    rng = np.random.default_rng(seed)
    cent = z[rng.choice(z.shape[0], size=k, replace=False)]
    lab = np.zeros(z.shape[0], dtype=np.int64)
    for _ in range(iters):
        d2 = ((z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        lab = d2.argmin(axis=1)
        for j in range(k):
            sel = lab == j
            if sel.any():
                cent[j] = z[sel].mean(axis=0)
    labels[ok] = lab
    return labels


# ==========================================================================
# Metadata read by oplib.py (never inlined). Only exceptions to the naming
# conventions need an entry.
# ==========================================================================

OP_META = {
    # plumbing
    "clean": {"kind": "internal"}, "finalize": {"kind": "internal"},
    "today": {"kind": "internal"}, "adjust_per_share": {"kind": "internal"},
    "call_row": {"kind": "internal"}, "rows_apply": {"kind": "internal"},
    "roll": {"kind": "internal"}, "hump_step": {"kind": "internal"},
    "ema_step": {"kind": "internal"}, "trade_when_step": {"kind": "internal"},
    "kmeans_labels": {"kind": "internal"},
    # ts ops that need one row more than d
    "ts_delay": {"extra": 1}, "ts_delta": {"extra": 1}, "ts_returns": {"extra": 1},
    "ts_autocorr": {"extra": "lag"},
    # series (window -> window)
    "ffill": {"kind": "series", "window": "limit", "unbounded": 250, "category": "prep"},
    "coverage_mask": {"kind": "series", "window": "d", "category": "prep"},
    # categories for row ops
    "add": {"category": "arithmetic"}, "subtract": {"category": "arithmetic"},
    "multiply": {"category": "arithmetic"}, "divide": {"category": "arithmetic"},
    "neg": {"category": "arithmetic"}, "inverse": {"category": "arithmetic"},
    "op_abs": {"category": "arithmetic"}, "sign": {"category": "arithmetic"},
    "op_log": {"category": "arithmetic"}, "signed_log": {"category": "arithmetic"},
    "sqrt": {"category": "arithmetic"}, "power": {"category": "arithmetic"},
    "signed_power": {"category": "arithmetic"}, "exp": {"category": "arithmetic"},
    "op_max": {"category": "arithmetic"}, "op_min": {"category": "arithmetic"},
    "clip": {"category": "arithmetic"}, "tanh": {"category": "arithmetic"},
    "sigmoid": {"category": "arithmetic"},
    "gt": {"category": "logic"}, "lt": {"category": "logic"}, "ge": {"category": "logic"},
    "le": {"category": "logic"}, "eq": {"category": "logic"}, "ne": {"category": "logic"},
    "and_": {"category": "logic"}, "or_": {"category": "logic"}, "not_": {"category": "logic"},
    "if_else": {"category": "logic"}, "is_nan": {"category": "logic"},
    "fill_zero": {"category": "prep"}, "fill_value": {"category": "prep"},
    "purify": {"category": "prep"}, "to_nan": {"category": "prep"},
    "ts_fft_smooth": {"category": "ml"}, "ts_pca_neutralize": {"category": "ml"},
}
