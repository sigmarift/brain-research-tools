"""
Offline tests: no BRAIN account and no ACE files needed.
Run from the tool folder:  pytest -q
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import local_runner as lr  # noqa: E402
from compiler import CompileError, Compiler  # noqa: E402
from oplib import STATEFUL, OperatorLibrary  # noqa: E402

LIB = OperatorLibrary()
COMP = Compiler(LIB)

SPECIAL = {
    "ts_pca_neutralize": "ts_pca_neutralize(close, returns, 30, 3)",
    "bucket": "group_neutralize(rank(close), bucket(rank(cap), 5))",
    "ts_percentile": "ts_percentile(close, 20, 0.8)",
    "coverage_mask": "rank(coverage_mask(close, 20, 0.5))",
    "ffill": "rank(ffill(close, 10))",
    "to_nan": "rank(to_nan(volume, 0))",
    "tail": "tail(rank(close), 0.4, 0.6, 0)",
    "clip": "clip(zscore(close), -2, 2)",
    "power": "power(rank(close), 2)",
    "signed_power": "signed_power(zscore(close), 2)",
    "if_else": "if_else(close > vwap, rank(volume), -rank(volume))",
    "fill_value": "rank(fill_value(close, 0))",
}


def _expr_for(sp):
    if sp.fe_name in SPECIAL:
        return SPECIAL[sp.fe_name]
    req = [p for p in sp.pos_params if p not in sp.defaults]
    if sp.kind == "group":
        return f"{sp.fe_name}(rank(close), bucket(rank(cap), 5))"
    if sp.kind == "ts":
        ins = ["close", "volume"][:len([p for p in req if p != "d"])]
        return f"{sp.fe_name}({', '.join(ins)}, 10)"
    if sp.vararg:
        return f"{sp.fe_name}(rank(close), rank(volume))"
    return f"{sp.fe_name}({', '.join(['close', 'volume', 'returns'][:len(req)])})"


OP_EXPRS = [_expr_for(sp) for sp in LIB.user_ops()] + [
    "hump(rank(close), 0.01)", "ema(rank(close), 0.2)",
    "trade_when(volume > ts_mean(volume, 20), rank(close), -1)",
    "cluster_neutralize(rank(close), returns, 30, 5, 10)",
]


@pytest.mark.parametrize("expr", OP_EXPRS)
def test_every_operator_compiles_and_runs(expr):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = COMP.compile(expr)
        r = lr.dry_run(a.code, lookback=a.lookback, n_days=a.lookback + 60, n_instruments=100)
    assert r.ok, r


def _csrank(df):
    from scipy.stats import rankdata
    return df.apply(lambda r: pd.Series((rankdata(r) - 1) / (len(r) - 1), index=r.index), axis=1)


@pytest.mark.parametrize("expr,ref", [
    ("ts_mean(ts_delta(close, 1), 5)", lambda c, v: c.diff(1).rolling(5).mean()),
    ("ts_corr(rank(close), rank(volume), 10)", lambda c, v: _csrank(c).rolling(10).corr(_csrank(v))),
    ("ts_std(ts_mean(close, 3) / ts_delay(close, 2), 7)",
     lambda c, v: (c.rolling(3).mean() / c.shift(2)).rolling(7).std(ddof=0)),
    ("a = ts_mean(close, 5); ts_zscore(a, 10)",
     lambda c, v: (lambda a: (a - a.rolling(10).mean()) / a.rolling(10).std(ddof=0))(c.rolling(5).mean())),
])
def test_nested_ops_match_pandas(expr, ref):
    pytest.importorskip("scipy")
    rng = np.random.default_rng(3)
    T, N = 120, 30
    close = np.exp(np.cumsum(rng.normal(0, .02, (T, N)), 0)) * 20
    vol = np.exp(rng.normal(12, 1, (T, N)))
    a = COMP.compile(expr)
    r, M = lr.dry_run(a.code, lookback=a.lookback, n_days=T, n_instruments=N, return_matrix=True,
                      data_override={"close": close, "volume": vol}, full_universe=True)
    assert r.ok
    want = ref(pd.DataFrame(close), pd.DataFrame(vol)).values
    s = a.lookback + 2
    ok = np.isfinite(want[s:])
    assert np.nanmax(np.abs(M[s:][ok] - want[s:][ok])) < 1e-4


@pytest.mark.parametrize("bad", ["ts_rankk(close, 5)", "vec_avg(x)", "rank(X)",
                                 "ts_mean(hump(close), 5)", "ts_mean(close, 500)", "1 + 2",
                                 "winsorize(close, sd=3)"])
def test_bad_expressions_give_clear_errors(bad):
    with pytest.raises(CompileError):
        COMP.compile(bad)


def test_generate_hits_exact_target_offline(tmp_path):
    from research import Research
    rs = Research(dict(offline=True, runs_dir=str(tmp_path),
                       sources=[{"fields": ["close", "volume", "returns", "vwap", "cap"]},
                                {"name": "fund", "fields": ["fnd6_a", "fnd6_b", "fnd6_c"]}],
                       directions=["reversion", "value"], max_alphas=30,
                       settings={"neutralization": ["MARKET", "SECTOR"]}))
    rs.load_fields()
    assert len(rs.generate()) == 30


def test_blocked_operators_are_never_used(tmp_path):
    from research import Research
    rs = Research(dict(offline=True, runs_dir=str(tmp_path), sources=[{"fields": ["close", "volume", "returns", "cap"]}],
                       directions=["reversion", "ml"], blocked_ops=["group", "stateful", "ml"], max_alphas=15))
    rs.load_fields()
    rs.generate()
    used = set(sum([c["alpha"].ops for c in rs.candidates], []))
    assert not used & {"group_neutralize", "cluster_neutralize", "hump", "ts_pca_neutralize"}


def test_kit_works_without_ace_files():
    import ace_bridge
    if ace_bridge.available():
        pytest.skip("ACE files present on this machine")
    with pytest.raises(ImportError, match="ACE library files were not found"):
        ace_bridge.load()
