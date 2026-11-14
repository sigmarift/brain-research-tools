"""
oplib.py  -  loads python_ops.py + my_ops.py: sources, signatures, kinds,
categories, FE names; resolves what to inline into a generated Alpha.
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

HERE = Path(__file__).resolve().parent
DEFAULT_FILES = [HERE / "python_ops.py", HERE / "my_ops.py"]

# FE name -> kit function (kit names avoid shadowing Python builtins)
FE_ALIASES = {
    "abs": "op_abs", "max": "op_max", "min": "op_min", "log": "op_log",
    "reverse": "neg", "ts_std_dev": "ts_std", "ts_decay_exp_window": "ts_decay_exp",
    "group_normalize": "group_zscore", "and": "and_", "or": "or_", "not": "not_",
    "greater": "gt", "less": "lt", "greater_equal": "ge", "less_equal": "le",
    "equal": "eq", "not_equal": "ne", "rank_gauss": "quantile",
    "ts_sharpe": "ts_ir", "ts_slope": "ts_trend", "ts_covar": "ts_covariance",
}
KIT_TO_FE = {"op_abs": "abs", "op_max": "max", "op_min": "min", "op_log": "log",
             "and_": "and", "or_": "or", "not_": "not"}
IDENTITY_OPS = {"pasteurize", "densify"}   # engine/settings already do these

# Stateful ops are wired by the compiler (store entries generated for you)
STATEFUL = {
    "hump": {"params": ["x", "hump"], "defaults": {"hump": 0.01},
             "doc": "Hold yesterday's value unless the change exceeds hump * sum(|x|) (cuts turnover)."},
    "ema": {"params": ["x", "a"], "defaults": {"a": 0.1},
            "doc": "Exponential moving average kept in store; can run far beyond lookback."},
    "trade_when": {"params": ["trigger", "x", "exit"], "defaults": {},
                   "doc": "Take x when trigger > 0, NaN when exit > 0, else keep previous (FE trade_when)."},
    "cluster_neutralize": {"params": ["x", "r", "d", "k", "refresh"],
                           "defaults": {"d": 60, "k": 20, "refresh": 20},
                           "doc": "Neutralize x within k-means clusters of r's last d days, re-clustered every `refresh` days."},
}

CATEGORY_ORDER = ["time-series", "cross-sectional", "group", "arithmetic", "logic",
                  "prep", "ml", "stateful", "custom"]


@dataclass
class OpSpec:
    name: str
    kind: str                  # ts | row | group | series | internal
    category: str
    params: list
    defaults: dict
    pos_params: list
    vararg: bool
    varkw: bool
    doc: str
    origin: str
    window: Optional[str] = None
    extra: Union[int, str] = 0
    unbounded: int = 250
    error: Optional[str] = None

    @property
    def fe_name(self) -> str:
        return KIT_TO_FE.get(self.name, self.name)

    def signature(self) -> str:
        parts = []
        for p in self.params:
            parts.append(f"{p}={self.defaults[p]!r}" if p in self.defaults else p)
        if self.vararg:
            parts.append("...")
        return f"{self.fe_name}({', '.join(parts)})"


def _literal_default(node):
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        return None


class OperatorLibrary:
    def __init__(self, files: Iterable[Union[str, Path]] = DEFAULT_FILES):
        self.sources: dict[str, str] = {}
        self.specs: dict[str, OpSpec] = {}
        self.order: list[str] = []
        self._deps: dict[str, set] = {}
        self.meta: dict[str, dict] = {}
        self.files = [Path(f) for f in files]
        for f in self.files:
            if f.exists():
                self._load(f.read_text(encoding="utf-8"), f.stem)
        self._finalize_specs()

    # ---------------------------------------------------------------- load
    def _load(self, text: str, origin: str) -> None:
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "OP_META" for t in node.targets):
                try:
                    self.meta.update(ast.literal_eval(node.value))
                except ValueError as e:
                    raise ValueError(f"OP_META in {origin} must be a plain dict literal: {e}")
            if not isinstance(node, ast.FunctionDef):
                continue
            a = node.args
            params = [p.arg for p in a.posonlyargs + a.args] + [p.arg for p in a.kwonlyargs]
            pos = [p.arg for p in a.posonlyargs + a.args]
            defaults = {}
            for p, dnode in zip(pos[len(pos) - len(a.defaults):], a.defaults):
                defaults[p] = _literal_default(dnode)
            for p, dnode in zip(a.kwonlyargs, a.kw_defaults):
                if dnode is not None:
                    defaults[p.arg] = _literal_default(dnode)
            if node.name not in self.sources:
                self.order.append(node.name)
            self.sources[node.name] = ast.get_source_segment(text, node)
            self._deps[node.name] = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            self.specs[node.name] = OpSpec(
                name=node.name, kind="", category="", params=params, defaults=defaults, pos_params=pos,
                vararg=a.vararg is not None, varkw=a.kwarg is not None,
                doc=(ast.get_docstring(node) or "").split("\n")[0], origin=origin)

    def _finalize_specs(self) -> None:
        for name, sp in self.specs.items():
            m = self.meta.get(name, {})
            if name.startswith("_"):
                kind = "private"
            elif "kind" in m:
                kind = m["kind"]
            elif name.startswith("ts_") or name in ("days_from_last_change", "last_diff_value"):
                kind = "ts"
            elif name.startswith("group_"):
                kind = "group"
            else:
                kind = "row"
            sp.kind = kind
            sp.extra = m.get("extra", 0)
            sp.unbounded = m.get("unbounded", 250)
            if kind == "ts":
                sp.window = m.get("window", "d")
                if sp.window not in sp.params:
                    sp.error = "time-series op needs a window parameter named 'd'"
            elif kind == "series":
                sp.window = m.get("window", "d")
            default_cat = {"ts": "time-series", "group": "group", "series": "prep"}.get(kind, "cross-sectional")
            sp.category = m.get("category", default_cat)
            if sp.origin == "my_ops" and "category" not in m:
                sp.category = "custom"

    # ---------------------------------------------------------------- query
    def canonical(self, name: str) -> str:
        return FE_ALIASES.get(name, name)

    def spec(self, name: str) -> Optional[OpSpec]:
        sp = self.specs.get(self.canonical(name))
        return sp if sp and sp.kind not in ("private", "internal") else None

    def user_ops(self) -> list[OpSpec]:
        return [self.specs[n] for n in self.order
                if self.specs[n].kind not in ("private", "internal")]

    def suggest(self, name: str, n: int = 3) -> list[str]:
        names = [s.fe_name for s in self.user_ops()] + list(STATEFUL) + list(FE_ALIASES)
        return difflib.get_close_matches(name, names, n=n, cutoff=0.5)

    def resolve(self, names: set) -> list[str]:
        need, stack = set(), [n for n in names if n in self.sources]
        while stack:
            n = stack.pop()
            if n in need:
                continue
            need.add(n)
            stack.extend(d for d in self._deps[n] if d in self.sources and d not in need)
        return [n for n in self.order if n in need]

    def inline(self, names: list[str]) -> str:
        return "\n\n\n".join(self.sources[n] for n in names)

    def catalog(self, search: str = "", category: Optional[str] = None):
        import pandas as pd
        rows = [{"operator": s.fe_name, "category": s.category, "kind": s.kind,
                 "usage": s.signature(), "description": s.doc, "source": s.origin,
                 "problem": s.error or ""} for s in self.user_ops()]
        for k, v in STATEFUL.items():
            usage = f"{k}(" + ", ".join(
                f"{p}={v['defaults'][p]!r}" if p in v["defaults"] else p for p in v["params"]) + ")"
            rows.append({"operator": k, "category": "stateful", "kind": "stateful", "usage": usage,
                         "description": v["doc"], "source": "compiler", "problem": ""})
        df = pd.DataFrame(rows)
        df["_o"] = df["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)}).fillna(99)
        df = df.sort_values(["_o", "operator"]).drop(columns="_o").reset_index(drop=True)
        if category:
            df = df[df["category"] == category]
        if search:
            s = search.lower()
            df = df[df["operator"].str.contains(s) | df["description"].str.lower().str.contains(s)]
        return df.reset_index(drop=True)

    def categories(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in self.user_ops():
            out.setdefault(s.category, []).append(s.fe_name)
        out["stateful"] = list(STATEFUL)
        return out
