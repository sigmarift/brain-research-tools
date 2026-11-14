"""
llm.py  -  optional AI assistance. Works with your own API key, or without
one (it prints the prompt; paste the reply back).

Key lookup, first match wins:
  1. CONFIG["llm"] = {"provider": "anthropic", "api_key": "...", "model": "..."}
  2. ~/secrets/llm.json with the same keys
  3. env ANTHROPIC_API_KEY, then OPENAI_API_KEY

providers: "anthropic", "openai", "openai_compatible" (set base_url, e.g. a
local server, OpenRouter, DeepSeek). Everything AI returns is validated
before use: expressions are compiled and dry-run, operators are checked
against a strict whitelist and tested on synthetic windows.
"""

from __future__ import annotations

import ast
import json
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests

SECRETS = Path(os.path.expanduser("~")) / "secrets" / "llm.json"
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o", "openai_compatible": ""}


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------

def load_llm_config(cfg: Optional[dict] = None) -> Optional[dict]:
    cfg = dict(cfg or {})
    if not cfg.get("api_key") and SECRETS.exists():
        try:
            cfg = {**json.loads(SECRETS.read_text()), **{k: v for k, v in cfg.items() if v}}
        except Exception:  # noqa: BLE001
            pass
    if not cfg.get("api_key"):
        if os.environ.get("ANTHROPIC_API_KEY"):
            cfg.setdefault("provider", "anthropic")
            cfg["api_key"] = os.environ["ANTHROPIC_API_KEY"]
        elif os.environ.get("OPENAI_API_KEY"):
            cfg.setdefault("provider", "openai")
            cfg["api_key"] = os.environ["OPENAI_API_KEY"]
    if not cfg.get("api_key"):
        return None
    cfg.setdefault("provider", "anthropic")
    cfg.setdefault("model", DEFAULT_MODELS.get(cfg["provider"], ""))
    if not cfg["model"]:
        raise ValueError("LLM config: set 'model' for this provider")
    return cfg


def save_llm_config(provider: str, api_key: str, model: str = "", base_url: str = "") -> Path:
    SECRETS.parent.mkdir(parents=True, exist_ok=True)
    data = {"provider": provider, "api_key": api_key, "model": model or DEFAULT_MODELS.get(provider, "")}
    if base_url:
        data["base_url"] = base_url
    SECRETS.write_text(json.dumps(data, indent=1))
    try:
        os.chmod(SECRETS, 0o600)
    except OSError:
        pass
    return SECRETS


def ask(prompt: str, cfg: Optional[dict], system: str = "", max_tokens: int = 4000,
        retries: int = 2) -> Optional[str]:
    """Text reply, or None when no key is configured."""
    if not cfg:
        return None
    prov = cfg["provider"]
    for attempt in range(retries + 1):
        try:
            if prov == "anthropic":
                r = requests.post(
                    cfg.get("base_url", "https://api.anthropic.com") + "/v1/messages",
                    headers={"x-api-key": cfg["api_key"], "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                    json={"model": cfg["model"], "max_tokens": max_tokens,
                          **({"system": system} if system else {}),
                          "messages": [{"role": "user", "content": prompt}]},
                    timeout=180)
                r.raise_for_status()
                return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
            base = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
            r = requests.post(base + "/chat/completions",
                              headers={"Authorization": f"Bearer {cfg['api_key']}"},
                              json={"model": cfg["model"], "messages": msgs, "max_tokens": max_tokens},
                              timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code in (429, 500, 502, 503, 529) and attempt < retries:
                time.sleep(5 * (attempt + 1))
                continue
            body = e.response.text[:300] if e.response is not None else ""
            raise RuntimeError(f"LLM request failed ({code}): {body}")
    return None


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

SYSTEM = ("You are a senior quantitative researcher writing WorldQuant BRAIN alphas. "
          "You value simple, economically motivated signals and avoid overfitting. "
          "Reply only in the exact format requested.")

SYNTAX = """Expression syntax (Fast Expression style):
- Call operators exactly as listed, e.g. ts_rank(rank(X), 20), group_neutralize(rank(X), bucket(rank(cap), 10))
- Arithmetic + - * /, comparisons > < >= <= == !=, cond ? a : b, && ||, unary minus
- Multi-step: a = rank(X); b = ts_mean(a, 5); b - ts_mean(a, 20)
- Placeholders X, Y, Z stand for fields chosen by the system; give each a role
- You may also use these fields directly: close, open, high, low, vwap, returns, volume, cap, adv20
- Windows are trading days; keep every window <= 250
- Stateful ops (hump, ema, trade_when, cluster_neutralize) must be outermost, never inside ts_ operators
- No vec_ operators, no GROUP fields (sector/industry); build groups with bucket(rank(cap), 10)"""

ROLES_HELP = ('Roles: "any", a kind (price, volume, returns, fundamental, analyst, event, model, sparse, other), '
              '"dense" (price|volume|returns|model|other), "slow" (fundamental|analyst|sparse), '
              'or "kind@source" to pick from one source.')


def _ops_block(catalog_df, limit=140) -> str:
    rows = catalog_df.head(limit)
    return "\n".join(f"- {r.usage}: {r.description}" for r in rows.itertuples())


def _fields_block(pool_df, limit=60) -> str:
    if pool_df is None or not len(pool_df):
        return "(no dataset fields loaded; use placeholders with roles)"
    lines = []
    for r in pool_df.head(limit).to_dict("records"):
        desc = str(r.get("description") or "")[:90]
        lines.append(f"- {r['id']} [{r['kind']}, source {r['source']}] {desc}")
    kinds = pool_df.groupby(["source", "kind"]).size().to_dict()
    summary = ", ".join(f"{s}/{k}: {n}" for (s, k), n in kinds.items())
    return f"Field counts by source/kind: {summary}\nExamples:\n" + "\n".join(lines)


def ideas_prompt(idea_text: str, pool_df, catalog_df, n: int = 12) -> str:
    return textwrap.dedent(f"""
    Research direction from the user:
    \"\"\"{idea_text}\"\"\"

    Write {n} distinct alpha expressions that test this direction from different angles
    (different operators, windows, normalizations). Each must have a clear economic reason.

    {SYNTAX}

    {ROLES_HELP}

    Available fields:
    {_fields_block(pool_df)}

    Allowed operators (use ONLY these):
    {_ops_block(catalog_df)}

    Reply with a JSON array only, no prose:
    [{{"expr": "...", "roles": {{"X": "fundamental"}}, "why": "one short sentence"}}, ...]
    """).strip()


def repair_prompt(failures: list[dict], catalog_df) -> str:
    items = "\n".join(f"- {f['expr']}  -> error: {f['error']}" for f in failures)
    return textwrap.dedent(f"""
    These alpha expressions failed to compile. Fix each one keeping its idea.
    {items}

    {SYNTAX}

    Allowed operators:
    {_ops_block(catalog_df)}

    Reply with a JSON array only: [{{"expr": "...", "roles": {{...}}, "why": "..."}}, ...]
    """).strip()


def results_prompt(results_df, catalog_df, n: int = 10) -> str:
    cols = [c for c in ("resolved", "settings", "sharpe", "fitness", "turnover", "returns", "drawdown")
            if c in results_df.columns]
    table = results_df[cols].head(25).to_string(index=False)
    return textwrap.dedent(f"""
    Simulation results so far (in-sample):
    {table}

    Suggest {n} new expressions that build on what worked and fix what did not
    (e.g. lower turnover, better normalization, related fields). Avoid tiny parameter tweaks.

    {SYNTAX}

    Allowed operators:
    {_ops_block(catalog_df)}

    Reply with a JSON array only: [{{"expr": "...", "roles": {{...}}, "why": "..."}}, ...]
    """).strip()


def operator_prompt(description: str, existing: list[str]) -> str:
    return textwrap.dedent(f"""
    Write ONE new numpy operator for a BRAIN Python Alpha operator library.
    What it should do: {description}

    Rules:
    - A single top-level Python function, numpy only (already imported as np). No imports.
    - Name: ts_<name>(x, ..., d=20) for time-series ops. x is a window of shape
      [rows, instruments] (oldest first, today last); return a 1-D array [instruments]
      for today. The window parameter MUST be named d.
      Or <name>(x, ...) for per-day ops that work row-wise on [rows, instruments].
    - NaN-safe: missing values are NaN; never let one NaN wipe out a whole instrument.
      Never modify inputs in place. No Python loops over instruments.
    - You may call these helpers: _tail(x, d) (last d rows as 2-D), _safe_div(a, b),
      _nanmean0(w), _nanstd0(w) (column-wise over time), _cs_mean(x), _cs_std(x), rank(x), today(x)
    - One-line docstring describing it.
    - Must not clash with existing names: {', '.join(existing[:200])}

    Reply with JSON only: {{"code": "def ...", "extra_rows": 0}}
    extra_rows = how many rows beyond d the op needs (1 if it compares to d days ago).
    """).strip()


# --------------------------------------------------------------------------
# Parsing + validation
# --------------------------------------------------------------------------

def _extract_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    pairs = sorted((("[", "]"), ("{", "}")), key=lambda p: (text.find(p[0]) if text.find(p[0]) >= 0 else 1e9))
    for opener, closer in pairs:
        i, j = text.find(opener), text.rfind(closer)
        if i >= 0 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("could not find JSON in the reply")


def parse_ideas(text: str) -> list[dict]:
    try:
        data = _extract_json(text)
    except ValueError:
        # fall back: one expression per line
        lines = [l.strip(" -*\t`") for l in text.splitlines()]
        return [{"expr": l, "roles": {}, "why": "from AI reply"} for l in lines if "(" in l]
    if isinstance(data, dict):
        data = [data]
    out = []
    for d in data:
        if isinstance(d, str):
            out.append({"expr": d, "roles": {}, "why": ""})
        elif isinstance(d, dict) and d.get("expr"):
            out.append({"expr": str(d["expr"]), "roles": d.get("roles") or {}, "why": d.get("why", "")})
    return out


_BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                 "getattr", "setattr", "delattr", "input", "breakpoint", "os", "sys",
                 "subprocess", "shutil", "pathlib", "socket", "requests", "importlib", "builtins"}


def check_operator_code(code: str) -> tuple[Optional[str], list[str]]:
    """(function name, problems). Strict whitelist: one function, numpy only."""
    problems = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return None, [f"syntax error: {e}"]
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    others = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.Expr))]
    if len(fns) != 1 or others:
        problems.append("must be exactly one top-level function and nothing else")
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            problems.append("imports are not allowed")
        if isinstance(n, ast.Name) and n.id in _BANNED_NAMES:
            problems.append(f"'{n.id}' is not allowed")
        if isinstance(n, ast.Attribute):
            if n.attr.startswith("__"):
                problems.append("dunder attributes are not allowed")
            if isinstance(n.value, ast.Name) and n.value.id not in ("np",) and n.attr in _BANNED_NAMES:
                problems.append(f"'{n.attr}' is not allowed")
        if isinstance(n, (ast.Global, ast.Nonlocal, ast.With, ast.AsyncFunctionDef, ast.Lambda)):
            problems.append(f"{type(n).__name__} is not allowed")
    name = fns[0].name if fns else None
    if name:
        params = [a.arg for a in fns[0].args.args]
        if name.startswith("ts_") and "d" not in params:
            problems.append("time-series operators need a window parameter named d")
    return name, sorted(set(problems))


def test_operator(code: str, name: str, ops_namespace: dict) -> list[str]:
    """Run the op on synthetic windows with NaNs; check shape and NaN safety."""
    ns = dict(ops_namespace)      # library helpers are globals for the new op
    ns["np"] = np
    safe_builtins = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                     for k in ("abs", "min", "max", "int", "float", "len", "range", "sum", "round",
                               "bool", "enumerate", "zip", "isinstance", "list", "tuple")}
    ns["__builtins__"] = safe_builtins
    try:
        exec(compile(code, "<ai_op>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        return [f"does not run: {type(e).__name__}: {e}"]
    fn = ns[name]
    rng = np.random.default_rng(0)
    problems = []
    for rows in (60, 3, 1):
        w = rng.normal(size=(rows, 50))
        w[rng.random(w.shape) < 0.1] = np.nan
        w[:, 0] = np.nan
        w.setflags(write=False)
        try:
            out = fn(w, d=20) if name.startswith("ts_") else fn(w)
        except Exception as e:  # noqa: BLE001
            problems.append(f"fails on a {rows}-row window: {type(e).__name__}: {e}")
            continue
        out = np.asarray(out)
        expect = (50,) if name.startswith("ts_") else w.shape
        if out.shape != expect and not (not name.startswith("ts_") and out.shape == (50,) and rows == 1):
            problems.append(f"{rows}-row window: output shape {out.shape}, expected {expect}")
        elif np.isfinite(out).mean() < 0.5 and rows >= 20:
            problems.append("output is mostly NaN on data with 10% gaps")
    return problems


def install_operator(code: str, name: str, extra_rows: int, description: str,
                     my_ops_path: Path) -> None:
    text = my_ops_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    meta_node = next((n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "OP_META" for t in n.targets)), None)
    meta = ast.literal_eval(meta_node.value) if meta_node else {}
    if extra_rows:
        meta[name] = {**meta.get(name, {}), "extra": int(extra_rows)}
    stamp = time.strftime("%Y-%m-%d")
    block = f"\n\n# AI-generated {stamp}: {description.strip()[:100]}\n{code.strip()}\n"
    if meta_node:
        lines = text.splitlines(keepends=True)
        before = "".join(lines[:meta_node.lineno - 1]).rstrip() + "\n"
        after = "".join(lines[meta_node.end_lineno:])
        meta_txt = "OP_META = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in meta.items()) + "}\n"
        text = before + block + "\n\n" + meta_txt + after
    else:
        text = text.rstrip() + "\n" + block + "\n\nOP_META = " + repr(meta) + "\n"
    my_ops_path.write_text(text, encoding="utf-8")
