"""
fields.py  -  where fields come from, what kind they are, how they get cleaned.

SOURCES  (any number, fields from all of them are pooled)
    {"dataset": "fundamental6", "prefix": "fnd6_", "select": "random", "n": 30}
    {"dataset": "analyst4", "contains": ["eps", "rev"], "select": "least_used", "n": 20}
    {"dataset": "pv1", "fields": ["close", "volume", "returns", "cap"]}
    {"fields": ["close", "vwap"], "name": "prices"}               # no API needed

  keys: dataset, name (label used in roles, default = dataset id), fields,
        prefix / contains / regex / exclude (str or list), search (API keyword),
        min_coverage, select = all | random | least_used | most_used | top_coverage,
        n (how many; omit for all), seed, kind (force a kind for this source)

KINDS  price, volume, returns, fundamental, analyst, event, model, sparse, other

PREP  "minimal" | "auto" | "robust" | {key: [steps]} where key is a field id,
      a kind, "@source" or "*", and "auto" inside the list expands to the
      default steps, e.g. {"fundamental": ["auto", "winsorize(std=3)"]}
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd

KINDS = ["price", "volume", "returns", "fundamental", "analyst", "event", "model", "sparse", "other"]

DEFAULT_PREP = {
    "price": [],              # + adjust_split for per-share fields
    "volume": [],             # + adjust_split_inverse for share counts
    "returns": [],
    "fundamental": ["ffill(limit=250)"],
    "sparse": ["ffill(limit=66)"],
    "analyst": ["ffill(limit=66)"],
    "event": ["ffill(limit=5)"],
    "model": ["ffill(limit=22)"],
    "other": [],
}
ROBUST_EXTRA = ["winsorize(std=4)"]

# synthetic generator to use in the local dry-run, per kind
DRYRUN_KIND = {"price": "price", "volume": "volume", "returns": "returns",
               "fundamental": "sparse", "sparse": "sparse", "analyst": "sparse",
               "event": "sparse", "model": "dense", "other": "dense"}

STANDARD_FIELDS = {
    "close": "price", "open": "price", "high": "price", "low": "price", "vwap": "price",
    "returns": "returns", "volume": "volume", "adv20": "volume", "cap": "volume",
    "sharesout": "volume", "adjfactor": "other", "dividend": "fundamental",
    "split": "other",
}
PER_SHARE = {"close", "open", "high", "low", "vwap"}
SHARE_COUNT = {"volume", "sharesout"}

CATEGORY_KIND = {
    "pv": None, "price volume": None, "fundamental": "fundamental", "analyst": "analyst",
    "earnings": "analyst", "news": "event", "sentiment": "event", "socialmedia": "event",
    "social media": "event", "model": "model", "risk": "model", "option": "model",
    "macro": "model", "insiders": "sparse", "institutions": "sparse",
    "shortinterest": "sparse", "short interest": "sparse", "imbalance": "event",
}
PREFIX_KIND = [("fnd", "fundamental"), ("fn_", "fundamental"), ("fundamental", "fundamental"),
               ("anl", "analyst"), ("est", "analyst"), ("nws", "event"), ("news", "event"),
               ("snt", "event"), ("scl", "event"), ("rsk", "model"), ("mdl", "model"),
               ("opt", "model"), ("oth", "other"), ("pv", "price")]


def _as_list(v) -> list:
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def infer_kind(fid: str, meta: Optional[dict] = None) -> tuple[str, bool, bool, str]:
    """(kind, per_share, share_count, reason)"""
    meta = meta or {}
    low = fid.lower()
    if low in STANDARD_FIELDS:
        return STANDARD_FIELDS[low], low in PER_SHARE, low in SHARE_COUNT, "standard field"
    per_share = bool(re.search(r"(^|_)(eps|bps|bvps|dps|cfps|sps)(_|$)|per_share|_ps(_|$)", low))
    cat = str(meta.get("category_id") or meta.get("category_name") or meta.get("category") or "").lower()
    if cat in CATEGORY_KIND and CATEGORY_KIND[cat]:
        return CATEGORY_KIND[cat], per_share, False, f"category {cat}"
    if cat in ("pv", "price volume"):
        if "volume" in low or "adv" in low:
            return "volume", False, False, "pv category"
        if "ret" in low:
            return "returns", False, False, "pv category"
        return "price", per_share or any(k in low for k in PER_SHARE), False, "pv category"
    ds = str(meta.get("dataset_id") or "").lower()
    for pre, kind in PREFIX_KIND:
        if low.startswith(pre) or ds.startswith(pre):
            return kind, per_share, False, f"prefix {pre}"
    if "volume" in low or low.startswith("adv"):
        return "volume", False, False, "name"
    if "return" in low:
        return "returns", False, False, "name"
    if any(k in low for k in ("price", "close")):
        return "price", True, False, "name"
    return "other", per_share, False, "default"


# --------------------------------------------------------------------------
# Selecting fields from sources
# --------------------------------------------------------------------------

def _cache_path(cache_dir: Path, ds: str, region, delay, universe, search) -> Path:
    tag = re.sub(r"\W+", "_", f"{region}_{delay}_{universe}_{ds}_{search}")
    return cache_dir / f"fields_{tag}.csv"


def fetch_dataset_fields(s, dataset: str, region: str, delay: int, universe: str,
                         search: str = "", cache_dir: Optional[Path] = None,
                         refresh: bool = False) -> pd.DataFrame:
    import ace_python as acp
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = _cache_path(cache_dir, dataset, region, delay, universe, search)
        if cp.exists() and not refresh:
            return pd.read_csv(cp)
    df = acp.get_matrix_fields(s, dataset_id=dataset, search=search, region=region,
                               delay=delay, universe=universe)
    if cache_dir is not None and not df.empty:
        df.to_csv(cp, index=False)
    return df


def _filter(df: pd.DataFrame, src: dict) -> pd.DataFrame:
    ids = df["id"].astype(str)
    keep = pd.Series(True, index=df.index)
    pre = _as_list(src.get("prefix"))
    if pre:
        keep &= ids.apply(lambda x: any(x.startswith(p) for p in pre))
    con = _as_list(src.get("contains"))
    if con:
        keep &= ids.apply(lambda x: any(c.lower() in x.lower() for c in con))
    if src.get("regex"):
        keep &= ids.str.contains(src["regex"], regex=True)
    exc = _as_list(src.get("exclude"))
    if exc:
        keep &= ~ids.apply(lambda x: any(x == e or x.startswith(e) for e in exc))
    mc = src.get("min_coverage")
    if mc and "coverage" in df.columns:
        keep &= df["coverage"].fillna(0) >= mc
    return df[keep]


def _select(df: pd.DataFrame, src: dict) -> pd.DataFrame:
    how = src.get("select", "all")
    n = src.get("n")
    if how == "random":
        rnd = random.Random(src.get("seed", 0))
        idx = list(df.index)
        rnd.shuffle(idx)
        df = df.loc[idx]
    elif how in ("least_used", "most_used"):
        col = next((c for c in ("userCount", "alphaCount") if c in df.columns), None)
        if col:
            df = df.sort_values(col, ascending=(how == "least_used"))
    elif how == "top_coverage" and "coverage" in df.columns:
        df = df.sort_values("coverage", ascending=False)
    return df.head(int(n)) if n else df


def build_pool(sources: list[dict], s=None, region="USA", delay=1, universe="TOP3000",
               offline=False, cache_dir: Optional[Path] = None, refresh=False,
               kind_overrides: Optional[dict] = None, log=print) -> pd.DataFrame:
    """Profile table of every selected field from every source."""
    kind_overrides = kind_overrides or {}
    frames = []
    for i, src in enumerate(sources):
        src = dict(src)
        label = src.get("name") or src.get("dataset") or f"source{i + 1}"
        if src.get("fields") and not src.get("dataset"):
            df = pd.DataFrame({"id": _as_list(src["fields"])})
        elif src.get("dataset"):
            if offline or s is None:
                if not src.get("fields"):
                    log(f"[{label}] offline: no field list (add 'fields': [...] to use it offline)")
                    continue
                df = pd.DataFrame({"id": _as_list(src["fields"])})
            else:
                df = fetch_dataset_fields(s, src["dataset"], src.get("region", region),
                                          src.get("delay", delay), src.get("universe", universe),
                                          src.get("search", ""), cache_dir, refresh)
                if df.empty:
                    log(f"[{label}] no MATRIX fields found for dataset '{src['dataset']}'")
                    continue
                if src.get("fields"):
                    df = df[df["id"].isin(_as_list(src["fields"]))]
        else:
            log(f"[{label}] needs 'dataset' or 'fields'")
            continue
        total = len(df)
        df = _select(_filter(df, src), src)
        log(f"[{label}] {len(df)} fields selected (of {total}) "
            f"select={src.get('select', 'all')} n={src.get('n', 'all')}")
        df = df.copy()
        df["source"] = label
        df["forced_kind"] = src.get("kind")
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["id", "source", "kind", "per_share", "share_count", "reason"])
    pool = pd.concat(frames, ignore_index=True).drop_duplicates("id", keep="first")
    rows = []
    for _, r in pool.iterrows():
        kind, ps, sc, why = infer_kind(str(r["id"]), r.to_dict())
        if isinstance(r.get("forced_kind"), str) and r.get("forced_kind"):
            kind, why = r["forced_kind"], "source setting"
        if r["id"] in kind_overrides:
            kind, why = kind_overrides[r["id"]], "your override"
        rows.append((kind, ps, sc, why))
    pool["kind"], pool["per_share"], pool["share_count"], pool["reason"] = zip(*rows)
    keep = ["id", "source", "kind", "per_share", "share_count", "reason"] + [
        c for c in ("coverage", "userCount", "alphaCount", "description", "dataset_id") if c in pool.columns]
    return pool[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Prep plan
# --------------------------------------------------------------------------

class PrepPlan:
    """Callable: field id -> list of prep steps (strings)."""

    def __init__(self, pool: pd.DataFrame, mode="auto", auto_split=True,
                 kind_overrides: Optional[dict] = None):
        self.mode = mode
        self.auto_split = auto_split
        self.kind_overrides = kind_overrides or {}
        self.info = {r["id"]: r for r in pool.to_dict("records")} if len(pool) else {}

    def profile(self, fid: str) -> dict:
        if fid in self.info:
            return self.info[fid]
        kind, ps, sc, why = infer_kind(fid)
        if fid in self.kind_overrides:
            kind = self.kind_overrides[fid]
        return {"id": fid, "kind": kind, "per_share": ps, "share_count": sc, "source": "-", "reason": why}

    def defaults(self, fid: str) -> list[str]:
        p = self.profile(fid)
        steps = []
        if self.auto_split and p.get("per_share"):
            steps.append("adjust_split")
        if self.auto_split and p.get("share_count"):
            steps.append("adjust_split_inverse")
        steps += DEFAULT_PREP.get(p["kind"], [])
        return steps

    def __call__(self, fid: str) -> list[str]:
        if fid == "adjfactor":
            return []
        mode = self.mode
        if mode == "minimal":
            return []
        base = self.defaults(fid)
        if mode == "robust":
            return base + ROBUST_EXTRA
        if isinstance(mode, dict):
            p = self.profile(fid)
            for key in (fid, f"@{p.get('source')}", p["kind"], "*"):
                if key in mode:
                    out = []
                    for st in mode[key]:
                        out += base if st == "auto" else [st]
                    return out
            return base
        return base

    def table(self, fields: list[str]) -> pd.DataFrame:
        return pd.DataFrame([{"field": f, "kind": self.profile(f)["kind"],
                              "prep": " > ".join(["clean"] + self(f))} for f in fields])
