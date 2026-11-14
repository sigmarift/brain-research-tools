"""
ace_python.py  -  Python Alpha support for the ACE API library (ace_lib.py).

ace_lib.generate_alpha() hard-codes "language": "FASTEXPR", has no lookback
or maxPosition, and always sends Fast Expression-only keys (nanHandling,
unitHandling). This module builds Python payloads correctly and keeps
Python Alphas out of multi-simulation (not supported for Python).
Everything else in ace_lib (polling, PnL, correlations, checks,
prettify_result, set_alpha_properties) is reused unchanged.

    import ace_lib as ace, ace_python as acp
    schema = acp.get_settings_schema(s)              # live list of valid keys
    p = acp.generate_python_alpha(code, lookback=20, schema=schema)
    res = acp.simulate_python_alpha_list(s, [p])
"""

from __future__ import annotations

import ast
from typing import Iterable, Literal, Optional

import pandas as pd

from ace_bridge import ace

FASTEXPR_ONLY_KEYS = ("nanHandling", "unitHandling")
_FORBIDDEN_BRAIN_IMPORTS = ("brain.models", "BrainCache", "from brain import")
_ALLOWED_IMPORT_ROOTS = {"brain", "numpy", "math"}


# --------------------------------------------------------------------------
# Code checks
# --------------------------------------------------------------------------

def read_alpha_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def validate_python_alpha(code: str) -> list[str]:
    """Local checks for the most common platform rejections. [] means OK."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    problems: list[str] = []
    for bad in _FORBIDDEN_BRAIN_IMPORTS:
        if bad in code:
            problems.append(f"Remove '{bad}': only 'from brain.alphas import alpha' is allowed.")
    decorated = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and any((isinstance(d, ast.Call) and getattr(d.func, "id", None) == "alpha")
                or getattr(d, "id", None) == "alpha" for d in n.decorator_list)
    ]
    if len(decorated) != 1:
        problems.append(f"Expected exactly one @alpha function, found {len(decorated)}.")
    else:
        fn = decorated[0]
        if len(fn.args.args) != 2:
            problems.append(f"@alpha function '{fn.name}' must take exactly (data, store).")
        dec = next((d for d in fn.decorator_list if isinstance(d, ast.Call)), None)
        for kw in getattr(dec, "keywords", []):
            if kw.arg == "data" and isinstance(kw.value, ast.List):
                if "universe" in [getattr(e, "value", None) for e in kw.value.elts]:
                    problems.append("Do not list 'universe' in data=[...]; it is always provided.")
    if "float32" not in code:
        problems.append("No float32 cast found; the return must be float32.")
    return problems


def non_numpy_imports(code: str) -> list[str]:
    """Imports outside numpy/brain/math. Not an error, but confirm the platform has them."""
    out = []
    for n in ast.walk(ast.parse(code)):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names if a.name.split(".")[0] not in _ALLOWED_IMPORT_ROOTS]
        elif isinstance(n, ast.ImportFrom) and n.module:
            if n.module.split(".")[0] not in _ALLOWED_IMPORT_ROOTS:
                out.append(n.module)
    return out


# --------------------------------------------------------------------------
# Settings schema (OPTIONS /simulations) -> which keys/choices are valid
# --------------------------------------------------------------------------

def _choice_values(obj) -> set:
    vals: set = set()
    if isinstance(obj, list):
        for c in obj:
            if isinstance(c, dict) and "value" in c:
                vals.add(c["value"])
            else:
                vals |= _choice_values(c)
    elif isinstance(obj, dict):
        for v in obj.values():
            vals |= _choice_values(v)
    return vals


def get_settings_schema(s) -> dict:
    """{api_key: {"label":..., "type":..., "choices": set()}} straight from the
    platform. Use it to see exactly which settings Python simulations accept."""
    r = s.options(ace.brain_api_url + "/simulations")
    children = r.json()["actions"]["POST"]["settings"]["children"]
    return {
        k: {"label": v.get("label"), "type": v.get("type"),
            "choices": _choice_values(v.get("choices", []))}
        for k, v in children.items()
    }


def show_settings_schema(schema: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"key": k, "label": v["label"], "type": v["type"],
         "choices": sorted(map(str, v["choices"]))[:20]}
        for k, v in schema.items()
    ])


def check_payload(payload: dict, schema: dict) -> list[str]:
    """Keys the platform does not know, and values outside listed choices."""
    issues = []
    for k, v in payload["settings"].items():
        if k not in schema:
            issues.append(f"unknown setting '{k}'")
        elif schema[k]["choices"] and v not in schema[k]["choices"]:
            issues.append(f"'{k}'={v!r} not in platform choices")
    return issues


# --------------------------------------------------------------------------
# Payload
# --------------------------------------------------------------------------

def generate_python_alpha(
    code: str,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: Literal[0, 1] = 1,
    lookback: int = 5,
    decay: int = 0,
    neutralization: str = "MARKET",
    truncation: float = 0.08,
    pasteurization: Literal["ON", "OFF"] = "ON",
    max_trade: Literal["ON", "OFF"] = "OFF",
    max_position: Literal["ON", "OFF"] = "OFF",
    test_period: Optional[str] = "P0Y0M0D",
    visualization: bool = False,
    drop_keys: Iterable[str] = (),
    extra_settings: Optional[dict] = None,
    schema: Optional[dict] = None,
    validate: bool = True,
) -> dict:
    """Python counterpart of ace.generate_alpha(). `code` = full module text.
    With `schema` (from get_settings_schema) unknown keys are dropped
    automatically, so payloads always match what the platform accepts."""
    if region == "GLB":
        raise ValueError("Python Alphas do not support the GLB region.")
    if validate:
        problems = validate_python_alpha(code)
        if problems:
            raise ValueError("Python Alpha failed local checks:\n- " + "\n- ".join(problems))
    settings = {
        "instrumentType": "EQUITY", "region": region, "universe": universe,
        "delay": delay, "lookback": int(lookback), "decay": decay,
        "neutralization": neutralization, "truncation": truncation,
        "pasteurization": pasteurization, "maxTrade": max_trade,
        "maxPosition": max_position, "language": "PYTHON",
        "visualization": visualization,
    }
    if test_period is not None:
        settings["testPeriod"] = test_period
    if extra_settings:
        settings.update(extra_settings)
    for k in tuple(FASTEXPR_ONLY_KEYS) + tuple(drop_keys):
        settings.pop(k, None)
    payload = {"type": "REGULAR", "settings": settings, "regular": code}
    if schema:
        for k in [k for k in settings if k not in schema]:
            ace.logger.warning(f"dropping setting '{k}': platform schema does not list it")
            settings.pop(k)
        for issue in check_payload(payload, schema):
            ace.logger.warning(f"settings check: {issue}")
    return payload


# --------------------------------------------------------------------------
# Fields
# --------------------------------------------------------------------------

def get_matrix_fields(
    s,
    dataset_id: Optional[str] = None,
    search: str = "",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    instrument_type: str = "EQUITY",
    max_fields: int = 5000,
) -> pd.DataFrame:
    """All MATRIX fields (the only type Python Alphas can load), paginated.
    ace.get_datafields returns only the first page and has no dataset filter."""
    base = (f"{ace.brain_api_url}/data-fields?instrumentType={instrument_type}"
            f"&region={region}&delay={delay}&universe={universe}&type=MATRIX&limit=50")
    if dataset_id:
        base += f"&dataset.id={dataset_id}"
    if search:
        base += f"&search={search}"
    rows, offset = [], 0
    while offset < max_fields:
        r = s.get(base + f"&offset={offset}")
        ace._check_rate_limit(r)
        js = r.json()
        batch = js.get("results", [])
        rows += batch
        offset += 50
        if not batch or offset >= js.get("count", 0):
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    try:
        df = ace.expand_dict_columns(df)
    except ValueError:      # ace helper fails when no column holds dicts
        pass
    if "type" in df.columns:
        df = df[df["type"] == "MATRIX"]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def _is_python(payload: dict) -> bool:
    return payload.get("settings", {}).get("language") == "PYTHON"


def simulate_python_alpha_list(s, alpha_list: list[dict], limit_of_concurrent_simulations: int = 3,
                               simulation_config: Optional[dict] = None) -> list:
    """Python Alphas as single concurrent simulations (no multi-sim)."""
    simulation_config = simulation_config or ace.DEFAULT_CONFIG
    return ace.simulate_alpha_list(s, alpha_list,
                                   limit_of_concurrent_simulations=limit_of_concurrent_simulations,
                                   simulation_config=simulation_config)


def simulate_alpha_list_multi_safe(s, alpha_list: list[dict], **kwargs) -> list:
    """Drop-in for ace.simulate_alpha_list_multi: FE via multi-sim, Python single."""
    py = [a for a in alpha_list if _is_python(a)]
    fx = [a for a in alpha_list if not _is_python(a)]
    out: list = []
    if fx:
        out += ace.simulate_alpha_list_multi(s, fx, **kwargs)
    if py:
        out += simulate_python_alpha_list(
            s, py, kwargs.get("limit_of_concurrent_simulations", 3),
            kwargs.get("simulation_config"))
    return out


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

def results_table(result: list, specs: Optional[list] = None) -> pd.DataFrame:
    """One row per simulated alpha with IS stats, joined to template metadata.
    Failed simulations are listed with alpha_id None."""
    by_code = {sp.code: sp for sp in (specs or [])}
    rows = []
    for r in result:
        code = r.get("simulate_data", {}).get("regular")
        sp = by_code.get(code)
        row = {"alpha_id": r.get("alpha_id"),
               "name": sp.name if sp else None,
               "template": sp.template if sp else None,
               "fields": sp.fields if sp else None,
               "params": sp.params if sp else None,
               "lookback": r.get("simulate_data", {}).get("settings", {}).get("lookback")}
        is_ = (r.get("is_stats") if isinstance(r.get("is_stats"), pd.DataFrame) else None)
        if is_ is not None and not is_.empty:
            for c in ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin",
                      "longCount", "shortCount"):
                if c in is_.columns:
                    row[c] = is_[c].iloc[0]
        tests = r.get("is_tests")
        if isinstance(tests, pd.DataFrame) and not tests.empty and "result" in tests.columns:
            row["fails"] = ",".join(tests.loc[tests["result"] == "FAIL", "name"].astype(str))
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_col = "fitness" if "fitness" in df.columns else None
    return df.sort_values(sort_col, ascending=False, na_position="last") if sort_col else df


def failed_payloads(result: list) -> list[dict]:
    return [r["simulate_data"] for r in result if r.get("alpha_id") is None]


def tag_alpha(s, alpha_id: str, spec=None, fe_alpha_id: Optional[str] = None,
              extra_tags: Iterable[str] = ()) -> int:
    """Tag a Python Alpha with its template and, for translations, the
    original Fast Expression Alpha id (as the BRAIN docs ask)."""
    tags = list(extra_tags)
    desc = None
    if spec is not None:
        tags.append(f"tpl_{spec.template}"[:30])
        desc = f"Python template {spec.template}; fields {spec.fields}; params {spec.params}."
    if fe_alpha_id:
        tags.append(f"FE_{fe_alpha_id}")
        desc = (desc or "") + f" Translated from Fast Expression alpha {fe_alpha_id}."
    kwargs = {"tags": tags}
    if desc:
        kwargs["regular_desc"] = desc.strip()
    return ace.set_alpha_properties(s, alpha_id, **kwargs).status_code
