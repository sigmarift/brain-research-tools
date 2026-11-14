"""
local_runner.py  -  dry-run a Python Alpha locally before spending a simulation.

It mimics the engine contract from the BRAIN docs on synthetic data:
  * called once per day with windows of up to lookback+1 rows (warm-up included)
  * every data array is read-only; universe is int 1/0 and never listed in data
  * the universe GROWS during the run, and typed store entries are auto-extended
    along their "i" axes with their `extend` value (checked for exact dtype)
  * int fields carry the int32-min missing sentinel, floats carry NaN gaps
  * the return must be a 1-D float32 array of shape (n_instruments,)

It does NOT tell you whether the alpha is good (data is random). It tells you
whether it will run, how much of the universe it covers, and how slow it is.
For real-data local checks use BrainLabs (brain.generate_alpha_matrix).

    from local_runner import dry_run
    r = dry_run(spec.code, lookback=spec.lookback)
    print(r)
"""

from __future__ import annotations

import sys
import time
import traceback
import types
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

PRICE_HINTS = ("close", "open", "high", "low", "vwap", "price")
RETURN_HINTS = ("returns", "ret")
VOLUME_HINTS = ("volume", "adv", "sharesout", "cap")


@dataclass
class DryRunResult:
    ok: bool
    name: str
    days_run: int
    error: Optional[str] = None
    error_day: Optional[int] = None
    coverage: float = 0.0          # mean share of in-universe names with a value
    ms_per_call: float = 0.0
    est_minutes_full_sim: float = 0.0
    warnings: list = field(default_factory=list)

    def __str__(self) -> str:
        if not self.ok:
            return f"FAIL {self.name} on day {self.error_day}: {self.error}"
        w = f"  warnings: {self.warnings}" if self.warnings else ""
        return (f"OK   {self.name}: coverage {self.coverage:.0%}, "
                f"{self.ms_per_call:.1f} ms/day, ~{self.est_minutes_full_sim:.1f} min "
                f"for 10y x 3000 names (rough){w}")


# --------------------------------------------------------------------------
# A stand-in for brain.alphas so the code runs without the BRAIN package
# --------------------------------------------------------------------------

def _stub_alpha(data=None, store=None):
    def deco(fn):
        fn._alpha_data = list(data or [])
        fn._alpha_store = list(store or [])
        return fn
    return deco


def load_alpha(code: str) -> Callable:
    """exec the module text with a stub `brain.alphas` and return the @alpha fn."""
    stub_brain = types.ModuleType("brain")
    stub_alphas = types.ModuleType("brain.alphas")
    stub_alphas.alpha = _stub_alpha
    stub_brain.alphas = stub_alphas
    saved = {k: sys.modules.get(k) for k in ("brain", "brain.alphas")}
    sys.modules["brain"], sys.modules["brain.alphas"] = stub_brain, stub_alphas
    try:
        ns: dict = {"__name__": "alpha_module"}
        exec(compile(code, "<alpha>", "exec"), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    fns = [v for v in ns.values() if callable(v) and hasattr(v, "_alpha_data")]
    if len(fns) != 1:
        raise ValueError(f"expected exactly one @alpha function, found {len(fns)}")
    return fns[0]


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

def _kind(name: str, field_kinds: dict) -> str:
    if name in field_kinds:
        return field_kinds[name]
    n = name.lower()
    if n == "adjfactor":
        return "adjfactor"
    if any(h == n or n.endswith("_" + h) for h in RETURN_HINTS):
        return "returns"
    if any(h in n for h in PRICE_HINTS):
        return "price"
    if any(h in n for h in VOLUME_HINTS):
        return "volume"
    if n.startswith(("fnd", "fn_", "mdl", "anl")):
        return "sparse"
    return "dense"


def _make_field(kind: str, T: int, N: int, rng: np.random.Generator, nan_rate: float):
    if kind == "returns":
        a = rng.normal(0, 0.02, (T, N))
    elif kind == "price":
        a = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    elif kind == "volume":
        a = np.exp(rng.normal(13, 1, (T, N)))
    elif kind == "adjfactor":
        a = np.ones((T, N))
        hits = rng.random((T, N)) < 0.0005
        a[hits] = rng.choice([0.5, 2.0, 5.0], size=hits.sum())
        return a.astype(np.float32)
    elif kind == "sparse":
        a = np.full((T, N), np.nan)
        base = rng.normal(0, 1, N)
        for t in range(0, T, 63):
            off = rng.integers(0, 10, N)
            for j in range(N):
                if t + off[j] < T:
                    a[t + off[j], j] = base[j] + rng.normal(0, 0.3)
        return a.astype(np.float32)
    elif kind == "int":
        a = rng.integers(0, 5, (T, N)).astype(np.int32)
        a[rng.random((T, N)) < nan_rate] = np.iinfo(np.int32).min
        return a
    else:
        a = rng.normal(0, 1, (T, N)).cumsum(axis=0) * 0.1
    a[rng.random((T, N)) < nan_rate] = np.nan
    return a.astype(np.float32)


# --------------------------------------------------------------------------
# Store emulation
# --------------------------------------------------------------------------

class _Store:
    pass


def _parse_store(decls):
    typed, names = {}, []
    for d in decls:
        if isinstance(d, str):
            names.append(d)
        elif isinstance(d, dict):
            names.append(d["name"])
            ext = d.get("extend", None)
            if ext is not None and not isinstance(ext, np.generic):
                raise TypeError(
                    f"Store item '{d['name']}' extend value '{ext}' has wrong type: "
                    f"{type(ext)} (use np.float64(...), np.float32(...), ...)")
            typed[d["name"]] = (d.get("dims", ""), ext)
        else:
            raise TypeError(f"bad store declaration: {d!r}")
    return typed, names


def _extend(val, dims: str, ext, n: int, name: str):
    if val is None or "i" not in dims:
        return val
    a = np.asarray(val)
    if a.ndim != len(dims):
        raise ValueError(f"store '{name}' has {a.ndim} dims but declared dims='{dims}'")
    if ext is not None and a.dtype != np.asarray(ext).dtype:
        raise ValueError(f"store '{name}' dtype {a.dtype} does not match extend dtype "
                         f"{np.asarray(ext).dtype}")
    fill = ext if ext is not None else (np.nan if a.dtype.kind == "f" else np.iinfo(a.dtype).min)
    for ax, c in enumerate(dims):
        if c == "i" and a.shape[ax] < n:
            pad = [(0, 0)] * a.ndim
            pad[ax] = (0, n - a.shape[ax])
            a = np.pad(a, pad, mode="constant", constant_values=fill)
    return a


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def dry_run(
    code_or_fn,
    lookback: int = 5,
    n_days: int = 300,
    n_instruments: int = 400,
    grow_from: float = 0.8,
    nan_rate: float = 0.05,
    field_kinds: Optional[dict] = None,
    seed: int = 0,
    return_matrix: bool = False,
    data_override: Optional[dict] = None,
    full_universe: bool = False,
):
    """Run the alpha day by day on synthetic data. field_kinds lets you force a
    generator per field: returns | price | volume | sparse | int | dense | adjfactor."""
    field_kinds = field_kinds or {}
    try:
        fn = load_alpha(code_or_fn) if isinstance(code_or_fn, str) else code_or_fn
    except Exception as e:  # noqa: BLE001
        return DryRunResult(False, "<module>", 0, f"{type(e).__name__}: {e}", -1)
    name = fn.__name__
    src_lines = code_or_fn.splitlines() if isinstance(code_or_fn, str) else []
    warns = []
    if "universe" in fn._alpha_data:
        return DryRunResult(False, name, 0, "'universe' listed in data=[...]", -1)
    try:
        typed, store_names = _parse_store(fn._alpha_store)
    except Exception as e:  # noqa: BLE001
        return DryRunResult(False, name, 0, f"{type(e).__name__}: {e}", -1)

    rng = np.random.default_rng(seed)
    T, N = n_days, n_instruments
    data_override = data_override or {}
    fields = {f: (np.asarray(data_override[f], dtype=np.float32) if f in data_override
                  else _make_field(_kind(f, field_kinds), T, N, rng, nan_rate)) for f in fn._alpha_data}
    univ = np.ones((T, N), np.int32) if full_universe else (rng.random((T, N)) < 0.9).astype(np.int32)
    n_t = np.full(T, N) if full_universe else np.linspace(int(N * grow_from), N, T).astype(int)

    store = _Store()
    for s in store_names:
        setattr(store, s, None)

    cover, times, out_rows = [], [], []
    for t in range(T):
        n = int(n_t[t])
        rows = min(t + 1, lookback + 1)
        data = types.SimpleNamespace()
        for f, arr in fields.items():
            w = np.array(arr[t + 1 - rows:t + 1, :n])
            w.setflags(write=False)
            setattr(data, f, w)
        u = np.array(univ[t + 1 - rows:t + 1, :n])
        u.setflags(write=False)
        data.universe = u
        try:
            for s, (dims, ext) in typed.items():
                setattr(store, s, _extend(getattr(store, s), dims, ext, n, s))
            t0 = time.perf_counter()
            out = fn(data, store)
            times.append(time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)
            fr = next((fr for fr in reversed(tb) if fr.filename == "<alpha>"), None)
            where = ""
            if fr is not None:
                txt = src_lines[fr.lineno - 1].strip() if 0 < fr.lineno <= len(src_lines) else ""
                where = f"line {fr.lineno} in {fr.name}: {txt}"
            return DryRunResult(False, name, t, f"{type(e).__name__}: {e}  [{where}]", t)

        if not isinstance(out, np.ndarray):
            return DryRunResult(False, name, t, f"returned {type(out).__name__}, not ndarray", t)
        if out.dtype != np.float32:
            return DryRunResult(False, name, t, f"Alpha vector is not float32 ({out.dtype})", t)
        if out.shape != (n,):
            return DryRunResult(False, name, t, f"shape {out.shape}, expected ({n},)", t)
        if np.isinf(out).any() and "inf in output" not in warns:
            warns.append("inf in output")
        inu = u[-1].astype(bool)
        if t >= lookback and inu.any():
            cover.append(float(np.isfinite(out[inu]).mean()))
        if return_matrix:
            row = np.full(N, np.nan, dtype=np.float32)
            row[:n] = out
            out_rows.append(row)

    ms = 1000 * float(np.mean(times)) if times else 0.0
    est = ms * (3000 / N) * 2520 / 60000
    cov = float(np.mean(cover)) if cover else 0.0
    if cov < 0.3:
        warns.append("low coverage: check warm-up, lookback, or field sparsity")
    res = DryRunResult(True, name, T, coverage=cov, ms_per_call=ms,
                       est_minutes_full_sim=est, warnings=warns)
    return (res, np.vstack(out_rows)) if return_matrix else res


def dry_run_many(specs, **kw):
    """Dry-run a list of AlphaSpec; returns (passed_specs, results DataFrame)."""
    import pandas as pd
    rows, passed = [], []
    for sp in specs:
        r = dry_run(sp.code, lookback=sp.lookback, **kw)
        rows.append({"name": sp.name, "ok": r.ok, "coverage": round(r.coverage, 3),
                     "ms_per_day": round(r.ms_per_call, 2),
                     "est_min_full": round(r.est_minutes_full_sim, 1),
                     "error": r.error, "warnings": "; ".join(r.warnings),
                     "spec": sp.describe()})
        if r.ok:
            passed.append(sp)
    return passed, pd.DataFrame(rows)
