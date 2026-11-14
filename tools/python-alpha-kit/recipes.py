"""
recipes.py  -  research directions, role binding, and the candidate stream
that keeps producing new, distinct ideas until the target count is reached.

An idea:  {"expr": "-rank(ts_delta(X, 5))", "roles": {"X": "price"}, "why": "..."}

Roles (what may fill a placeholder X, Y, Z ...):
    "any"                    every pooled field
    "price" / "fundamental"  a kind (see fields.KINDS); "fundamental|analyst" = either
    "dense" / "slow"         groups: dense = price|volume|returns|model|other,
                             slow = fundamental|analyst|sparse
    "@fund"                  any field from the source named "fund"
    "fundamental@fund"       that kind, from that source
    ["close", "vwap"]        an explicit list
"""

from __future__ import annotations

import ast
import copy
import itertools
import random
from typing import Iterator, Optional

import pandas as pd

ROLE_GROUPS = {
    "dense": {"price", "volume", "returns", "model", "other"},
    "slow": {"fundamental", "analyst", "sparse"},
    "any": None,
}

# --------------------------------------------------------------------------
# Directions
# --------------------------------------------------------------------------

DIRECTIONS: dict[str, dict] = {
    "reversion": {
        "about": "Short-term overreaction that tends to reverse.",
        "settings": {"decay": 3, "neutralization": "MARKET"},
        "ideas": [
            ("-rank(ts_delta(X, 5))", {"X": "price"}, "5-day move reverses"),
            ("-ts_zscore(X, 20)", {"X": "dense"}, "stretched vs its own month"),
            ("-rank(X / ts_mean(X, 20) - 1)", {"X": "price"}, "distance from 20-day mean"),
            ("-ts_rank(rank(returns), 5)", {}, "recent cross-sectional winners give back"),
            ("-rank(ts_av_diff(X, 10)) * rank(volume / ts_mean(volume, 20))", {"X": "price"},
             "reversal stronger on heavy volume"),
            ("-group_neutralize(ts_zscore(X, 10), bucket(rank(cap), 10))", {"X": "dense"},
             "reversion within size buckets"),
        ],
    },
    "momentum": {
        "about": "Medium-term trends that persist.",
        "settings": {"decay": 2, "neutralization": "MARKET"},
        "ideas": [
            ("rank(ts_sum(returns, 120) - ts_sum(returns, 20))", {}, "6m momentum skipping last month"),
            ("ts_rank(X, 120)", {"X": "dense"}, "near its 6-month high"),
            ("rank(ts_returns(X, 60))", {"X": "price"}, "3-month return"),
            ("rank(ts_trend(X, 60) / ts_std(X, 60))", {"X": "dense"}, "trend strength vs noise"),
            ("rank(ts_ir(returns, 60))", {}, "steady gains"),
            ("rank(ts_efficiency_ratio(X, 40)) * sign(ts_delta(X, 40))", {"X": "price"},
             "clean trends only"),
        ],
    },
    "volume": {
        "about": "Price/volume interaction and liquidity effects.",
        "settings": {"decay": 4, "neutralization": "MARKET"},
        "ideas": [
            ("-ts_corr(rank(X), rank(Y), 10)", {"X": "price", "Y": "volume"}, "price-volume divergence"),
            ("-rank(ts_delta(X, 3)) * rank(Y / ts_mean(Y, 20))", {"X": "price", "Y": "volume"},
             "volume-confirmed reversal"),
            ("-ts_corr(X, Y, 20)", {"X": "price", "Y": "volume"}, "raw price-volume correlation"),
            ("-rank(ts_zscore(Y, 20))", {"Y": "volume"}, "abnormal attention fades"),
        ],
    },
    "volatility": {
        "about": "Low-risk anomaly and volatility regimes.",
        "settings": {"decay": 4, "neutralization": "MARKET"},
        "ideas": [
            ("-rank(ts_std(returns, 20))", {}, "low volatility outperforms"),
            ("-rank(ts_std(returns, 5) / ts_std(returns, 60))", {}, "volatility spikes fade"),
            ("-rank(ts_std(X, 20) / ts_mean(abs(X), 20))", {"X": "dense"}, "unstable series"),
            ("-rank(ts_skew(returns, 60))", {}, "lottery-like stocks underperform"),
        ],
    },
    "value": {
        "about": "Cheap vs expensive using fundamentals scaled by size.",
        "settings": {"decay": 0, "neutralization": "SECTOR"},
        "ideas": [
            ("rank(X / cap)", {"X": "fundamental"}, "fundamental yield"),
            ("group_neutralize(rank(X / cap), bucket(rank(cap), 10))", {"X": "fundamental"},
             "yield within size buckets"),
            ("ts_rank(X / cap, 250)", {"X": "fundamental"}, "cheap vs own history"),
        ],
    },
    "quality": {
        "about": "Profitability, efficiency, balance-sheet strength.",
        "settings": {"decay": 0, "neutralization": "SECTOR"},
        "ideas": [
            ("rank(X / Y)", {"X": "fundamental", "Y": "fundamental"}, "fundamental ratio"),
            ("ts_zscore(X / Y, 250)", {"X": "fundamental", "Y": "fundamental"}, "ratio improving"),
            ("-rank(ts_std(X, 250) / abs(ts_mean(X, 250)))", {"X": "fundamental"}, "stable fundamentals"),
        ],
    },
    "growth": {
        "about": "Improving fundamentals.",
        "settings": {"decay": 0, "neutralization": "SECTOR"},
        "ideas": [
            ("rank(ts_delta(X, 63) / abs(ts_delay(X, 63)))", {"X": "fundamental"}, "quarterly growth"),
            ("ts_rank(X, 250)", {"X": "fundamental"}, "at a yearly high"),
            ("rank(ts_trend(X, 120) / ts_mean(abs(X), 120))", {"X": "fundamental"}, "steady improvement"),
        ],
    },
    "analyst": {
        "about": "Estimate revisions and analyst sentiment.",
        "settings": {"decay": 2, "neutralization": "SECTOR"},
        "ideas": [
            ("rank(ts_delta(X, 22) / abs(ts_delay(X, 22)))", {"X": "analyst"}, "monthly revision"),
            ("ts_zscore(X, 66)", {"X": "analyst"}, "estimate unusually high"),
            ("rank(ts_delta(X, 5))", {"X": "analyst"}, "fresh revisions"),
            ("group_neutralize(rank(ts_delta(X, 22)), bucket(rank(cap), 10))", {"X": "analyst"},
             "revisions within size buckets"),
        ],
    },
    "sentiment": {
        "about": "News, social and event data.",
        "settings": {"decay": 4, "neutralization": "MARKET"},
        "ideas": [
            ("rank(ts_mean(X, 5))", {"X": "event"}, "recent tone"),
            ("ts_decay_linear(rank(X), 10)", {"X": "event"}, "decaying sentiment"),
            ("rank(ts_mean(X, 5) - ts_mean(X, 60))", {"X": "event"}, "sentiment shift"),
            ("-rank(ts_mean(X, 3)) * rank(ts_std(returns, 20))", {"X": "event"},
             "overreaction in volatile names"),
        ],
    },
    "model": {
        "about": "Vendor model scores and risk factors.",
        "settings": {"decay": 2, "neutralization": "MARKET"},
        "ideas": [
            ("rank(X)", {"X": "model"}, "score as is"),
            ("ts_rank(X, 60)", {"X": "model"}, "score vs own history"),
            ("ts_decay_linear(zscore(X), 10)", {"X": "model"}, "smoothed score"),
        ],
    },
    "ml": {
        "about": "Clustering, PCA and Fourier denoising (Python-only ideas).",
        "settings": {"decay": 3, "neutralization": "MARKET"},
        "ideas": [
            ("cluster_neutralize(-ts_mean(returns, 5), returns, d=60, k=20, refresh=20)", {},
             "reversal vs correlation peers"),
            ("ts_pca_neutralize(-ts_delta(X, 5), returns, 60, 3)", {"X": "price"},
             "reversal net of statistical factors"),
            ("-rank(X - ts_fft_smooth(X, 60, 4))", {"X": "dense"}, "deviation from smooth component"),
            ("cluster_neutralize(rank(X), returns, d=60, k=20, refresh=20)", {"X": "any"},
             "field relative to correlation peers"),
        ],
    },
    "explore": {
        "about": "Generic first look at any dataset.",
        "settings": {"decay": 2, "neutralization": "MARKET"},
        "ideas": [
            ("rank(X)", {"X": "any"}, "level"),
            ("ts_rank(X, 60)", {"X": "any"}, "vs own history"),
            ("-rank(ts_delta(X, 5))", {"X": "any"}, "recent change reverses"),
            ("ts_zscore(X, 120)", {"X": "any"}, "unusual vs own history"),
            ("group_neutralize(rank(X), bucket(rank(cap), 10))", {"X": "any"}, "level within size buckets"),
            ("rank(ts_delta(X, 22))", {"X": "any"}, "monthly change continues"),
        ],
    },
}

WRAPPERS = [
    ("rank(E)", "ranked"),
    ("-(E)", "sign flipped"),
    ("ts_decay_linear(E, 5)", "smoothed (lower turnover)"),
    ("group_neutralize(E, bucket(rank(cap), 10))", "size-bucket neutral"),
    ("zscore(winsorize(E, std=3))", "winsorized z-score"),
    ("hump(E, 0.01)", "hump (lower turnover)"),
    ("ts_mean(E, 3)", "3-day average"),
]
WINDOW_FACTORS = [0.5, 2.0, 1.5, 0.75, 3.0]


def list_directions() -> pd.DataFrame:
    return pd.DataFrame([{"direction": k, "ideas": len(v["ideas"]), "about": v["about"],
                          "default settings": v["settings"]} for k, v in DIRECTIONS.items()])


def direction_ideas(name: str) -> list[dict]:
    if name not in DIRECTIONS:
        raise KeyError(f"unknown direction '{name}'. Choose from: {', '.join(DIRECTIONS)}")
    d = DIRECTIONS[name]
    return [{"expr": e, "roles": r, "why": w, "origin": f"direction:{name}",
             "settings": d["settings"]} for e, r, w in d["ideas"]]


def normalize_idea(item, default_roles: Optional[dict] = None, origin="expression") -> dict:
    if isinstance(item, str):
        item = {"expr": item}
    item = dict(item)
    item.setdefault("roles", {})
    for k, v in (default_roles or {}).items():
        item["roles"].setdefault(k, v)
    item.setdefault("why", "")
    item.setdefault("origin", origin)
    item.setdefault("settings", {})
    return item


def compose_ideas(spec: dict) -> list[dict]:
    """{"start": ["X", "X / Y"], "layers": [["rank(_)", "zscore(_)"],
        ["ts_rank(_, 20)", "ts_decay_linear(_, 10)"], ["_", "-_"]], "roles": {...}}"""
    import re
    starts = spec.get("start", ["X"])
    layers = spec.get("layers", [])
    out = []
    for start in starts:
        for combo in itertools.product(*layers) if layers else [()]:
            e = start
            for pat in combo:
                e = re.sub(r"(?<![\w.])_(?![\w])", f"({e})", pat)
            out.append({"expr": e, "roles": dict(spec.get("roles", {})), "why": "composed",
                        "origin": "compose", "settings": {}})
    return out


# --------------------------------------------------------------------------
# Binding placeholders to fields
# --------------------------------------------------------------------------

def placeholders(expr: str) -> list[str]:
    from compiler import PLACEHOLDER_RE, fe_to_python
    try:
        tree = ast.parse(fe_to_python(expr))
    except SyntaxError:
        return []
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and PLACEHOLDER_RE.match(n.id) and n.id not in names:
            names.append(n.id)
    return sorted(names)


def fields_for_role(role, pool: pd.DataFrame) -> list[str]:
    if isinstance(role, (list, tuple)):
        return list(role)
    role = role or "any"
    kinds_part, _, src = str(role).partition("@")
    df = pool
    if src:
        df = df[df["source"] == src]
    if kinds_part and kinds_part != "any":
        kinds = set()
        for k in kinds_part.split("|"):
            g = ROLE_GROUPS.get(k)
            kinds |= g if g else {k}
        df = df[df["kind"].isin(kinds)]
    return df["id"].tolist()


def bindings_for(idea: dict, pool: pd.DataFrame, rng: random.Random, offset: int = 0) -> Iterator[dict]:
    phs = placeholders(idea["expr"])
    if not phs:
        yield {}
        return
    lists = []
    for ph in phs:
        lst = fields_for_role(idea["roles"].get(ph, "any"), pool)
        if not lst:
            return
        k = offset % len(lst)
        lists.append(lst[k:] + lst[:k])      # each idea starts on a different field
    # interleave so early candidates use many different fields, not one field N times
    seen = set()
    for combo in _diagonal_product(lists):
        if len(set(combo)) < len(combo):
            continue
        key = tuple(combo)
        if key in seen:
            continue
        seen.add(key)
        yield dict(zip(phs, combo))


def _diagonal_product(lists):
    if len(lists) == 1:
        for x in lists[0]:
            yield (x,)
        return
    lens = [len(l) for l in lists]
    total = 1
    for n in lens:
        total *= n
    for k in range(total):
        idx, rem = [], k
        for n in lens:
            idx.append(rem % n)
            rem //= n
        # shift so first pass walks the diagonal
        yield tuple(lists[i][(idx[i] + idx[0] * (i > 0)) % lens[i]] for i in range(len(lists)))


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------

def window_variants(expr: str, lib, factors=WINDOW_FACTORS, max_window=250) -> list[tuple[str, str]]:
    from compiler import fe_to_python
    try:
        tree = ast.parse(fe_to_python(expr))
    except SyntaxError:
        return []
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            sp = lib.spec(node.func.id)
            if sp is None or sp.kind != "ts":
                continue
            i = sp.pos_params.index(sp.window) if sp.window in sp.pos_params else None
            if i is not None and i < len(node.args) and isinstance(node.args[i], ast.Constant):
                sites.append((node, "pos", i))
            for kw in node.keywords:
                if kw.arg == sp.window and isinstance(kw.value, ast.Constant):
                    sites.append((node, "kw", kw))
    out = []
    for f in factors:
        t2 = copy.deepcopy(tree)
        changed = False
        nodes = [n for n in ast.walk(t2) if isinstance(n, ast.Call)]
        orig = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        for node, how, where in sites:
            n2 = nodes[orig.index(node)]
            if how == "pos":
                c = n2.args[where]
            else:
                c = next(k for k in n2.keywords if k.arg == where.arg).value
            new = max(2, min(max_window, int(round(c.value * f))))
            if new != c.value:
                c.value = new
                changed = True
        if changed:
            out.append((ast.unparse(t2).replace("\n", "; "), f"windows x{f}"))
    return out


def wrapper_variants(expr: str) -> list[tuple[str, str]]:
    return [(w.replace("E", f"({expr})"), why) for w, why in WRAPPERS]


# --------------------------------------------------------------------------
# Random expressions (last-resort filler, respects allowed operators)
# --------------------------------------------------------------------------

class RandomExpr:
    WINDOWS = [5, 10, 20, 40, 60, 120]
    SKIP = {"ffill", "coverage_mask", "fill_value", "to_nan", "tail", "bucket", "clip",
            "power", "signed_power", "exp", "is_nan", "if_else", "ts_pca_neutralize",
            "ts_fft_smooth", "days_from_last_change", "last_diff_value", "ts_count",
            "ts_count_nans", "ts_product", "scale", "truncate", "quantile"}

    def __init__(self, lib, allowed, rng: random.Random):
        self.rng = rng
        ok = lambda s: s.fe_name not in self.SKIP and allowed(s.fe_name, s.category) and not s.error
        def req(s):
            return [p for p in s.pos_params if p not in s.defaults]
        self.ts1 = [s.fe_name for s in lib.user_ops() if s.kind == "ts" and ok(s) and len(req(s)) == 1]
        self.ts2 = [s.fe_name for s in lib.user_ops() if s.kind == "ts" and ok(s) and len(req(s)) == 2]
        self.cs1 = [s.fe_name for s in lib.user_ops() if s.kind == "row" and ok(s) and len(req(s)) == 1
                    and s.category in ("cross-sectional", "arithmetic") and not s.vararg]
        self.bin = [o for o in ("+", "-", "*", "/") if allowed({"+": "add", "-": "subtract",
                    "*": "multiply", "/": "divide"}[o], "arithmetic")]

    def make(self, fields: list[str]) -> Optional[str]:
        if not fields or not (self.ts1 or self.cs1):
            return None
        r = self.rng
        f = r.choice(fields)
        e = f
        if self.cs1 and r.random() < 0.5:
            e = f"{r.choice(self.cs1)}({e})"
        if self.ts1:
            e = f"{r.choice(self.ts1)}({e}, {r.choice(self.WINDOWS)})"
        if len(fields) > 1 and r.random() < 0.3:
            g = r.choice([x for x in fields if x != f])
            if self.ts2 and r.random() < 0.5:
                e = f"{r.choice(self.ts2)}({e}, {g}, {r.choice(self.WINDOWS)})"
            elif self.bin:
                e = f"({e}) {r.choice(self.bin)} rank({g})"
        if self.cs1:
            e = f"{r.choice(self.cs1)}({e})"
        if r.random() < 0.5:
            e = f"-{e}"
        return e


# --------------------------------------------------------------------------
# Candidate stream: escalates until it has produced enough distinct ideas
# --------------------------------------------------------------------------

def candidate_stream(ideas: list[dict], pool: pd.DataFrame, lib, allowed, seed=0,
                     variants=True, random_fill=True) -> Iterator[tuple[str, dict, dict]]:
    """Yields (expr, bindings, info). Level 0: ideas x field bindings, round-robin.
    Level 1: window variants. Level 2: wrapper variants. Level 3: random expressions."""
    rng = random.Random(seed)

    def roundrobin(gens):
        gens = list(gens)
        while gens:
            nxt = []
            for g in gens:
                try:
                    yield next(g)
                    nxt.append(g)
                except StopIteration:
                    pass
            gens = nxt

    def per_idea(idea, expr, level, why_extra="", offset=0):
        for b in bindings_for({**idea, "expr": expr}, pool, rng, offset):
            yield expr, b, {"idea": idea["expr"], "why": (idea.get("why", "") + why_extra).strip(),
                            "origin": idea.get("origin", ""), "level": level,
                            "idea_settings": idea.get("settings", {}),
                            "fe_alpha_id": idea.get("fe_alpha_id")}

    yield from roundrobin(per_idea(i, i["expr"], 0, "", k) for k, i in enumerate(ideas))
    if variants:
        yield from roundrobin(per_idea(i, e, 1, f" [{w}]", k + j) for k, i in enumerate(ideas)
                              for j, (e, w) in enumerate(window_variants(i["expr"], lib)))
        yield from roundrobin(per_idea(i, e, 2, f" [{w}]", k + j) for k, i in enumerate(ideas)
                              for j, (e, w) in enumerate(wrapper_variants(i["expr"])))
    if random_fill and len(pool):
        gen = RandomExpr(lib, allowed, rng)
        fields = pool["id"].tolist()
        for _ in range(100000):
            e = gen.make(fields)
            if e is None:
                return
            yield e, {}, {"idea": e, "why": "random combination of allowed operators",
                          "origin": "random", "level": 3, "idea_settings": {}}
