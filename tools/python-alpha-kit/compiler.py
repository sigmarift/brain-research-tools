"""
compiler.py  -  Fast Expression style text -> self-contained BRAIN Python Alpha.

    from compiler import Compiler
    c = Compiler()
    alpha = c.compile("ts_rank(rank(X), 20)", bindings={"X": "close"})
    print(alpha.code, alpha.lookback)

What it handles for you
  * FE syntax: ;-separated statements, a = ...; assignments, cond ? a : b,
    &&, ||, !, true/false, keyword args (std=4, driver=gaussian, range="0,1,0.1")
  * FE names (abs, max, ts_std_dev, reverse, ...) mapped to kit operators
  * nested time-series ops behave like FE: ts_mean(ts_delta(x, 1), 5) works
  * lookback computed automatically from every window in the expression
  * identical sub-expressions computed once
  * per-field cleaning/backfill (prep plan) applied before the expression
  * stateful ops (hump, ema, trade_when, cluster_neutralize) get their
    `store` entries generated
  * only the operators actually used are inlined
"""

from __future__ import annotations

import ast
import copy
import hashlib
import keyword
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from oplib import IDENTITY_OPS, STATEFUL, OperatorLibrary

GROUP_FIELD_NAMES = {"sector", "industry", "subindustry", "country", "exchange"}
PLACEHOLDER_RE = re.compile(r"^[A-Z][0-9]?$")      # X, Y, Z, X2 ...
DEFAULT_MAX_LOOKBACK = 260

_HEADER = "from brain.alphas import alpha\nimport numpy as np\nimport numpy.typing as npt\n"


class CompileError(Exception):
    pass


class _NotConst(Exception):
    pass


@dataclass
class CompiledAlpha:
    name: str
    code: str
    lookback: int
    expr: str                    # what the user/idea wrote
    resolved: str                # with fields filled in
    bindings: dict
    fields: list
    prep: dict                   # field -> list of steps applied
    ops: list                    # user-facing op names used
    warnings: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return hashlib.md5(self.code.encode()).hexdigest()[:12]

    def payload(self, **settings) -> dict:
        import ace_python as acp
        settings = dict(settings)
        settings["lookback"] = self.lookback
        return acp.generate_python_alpha(self.code, **settings)

    def save(self, folder="python_alphas"):
        from pathlib import Path
        d = Path(folder)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{self.name}.py"
        p.write_text(self.code, encoding="utf-8")
        return p


# --------------------------------------------------------------------------
# FE text -> Python-parsable text
# --------------------------------------------------------------------------

def _match_close(s: str, i: int) -> int:
    pairs = {"(": ")", "[": "]"}
    depth, j, open_c = 0, i, s[i]
    while j < len(s):
        c = s[j]
        if c in "\"'":
            j = s.index(c, j + 1)
        elif c == open_c:
            depth += 1
        elif c == pairs[open_c]:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise CompileError(f"unbalanced '{open_c}' in: {s}")


def _split_top(s: str, sep: str) -> list[str]:
    parts, depth, cur, j = [], 0, "", 0
    while j < len(s):
        c = s[j]
        if c in "\"'":
            k = s.index(c, j + 1)
            cur += s[j:k + 1]
            j = k + 1
            continue
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
        j += 1
    parts.append(cur)
    return parts


def _ternary(seg: str) -> str:
    res, i = "", 0
    while i < len(seg):
        c = seg[i]
        if c in "\"'":
            k = seg.index(c, i + 1)
            res += seg[i:k + 1]
            i = k + 1
        elif c in "([":
            j = _match_close(seg, i)
            inner = ",".join(_ternary(p) for p in _split_top(seg[i + 1:j], ","))
            res += c + inner + seg[j]
            i = j + 1
        else:
            res += c
            i += 1
    q = res.find("?")
    if q < 0:
        return res
    rest, depth = res[q + 1:], 0
    for k, c in enumerate(rest):
        if c == "?":
            depth += 1
        elif c == ":":
            if depth == 0:
                return (f"if_else({res[:q].strip()}, {_ternary(rest[:k]).strip()}, "
                        f"{_ternary(rest[k + 1:]).strip()})")
            depth -= 1
    raise CompileError(f"'?' without matching ':' in: {seg}")


def fe_to_python(src: str) -> str:
    s = src.replace("&&", " & ").replace("||", " | ")
    s = re.sub(r"!(?!=)", " ~", s)
    s = re.sub(r"\band\s*\(", "and_(", s)
    s = re.sub(r"\bor\s*\(", "or_(", s)
    s = re.sub(r"\bnot\s*\(", "not_(", s)
    s = re.sub(r"\btrue\b", "True", s)
    s = re.sub(r"\bfalse\b", "False", s)
    stmts = []
    for line in s.splitlines():
        line = line.split("#")[0]
        stmts += [p.strip() for p in _split_top(line, ";") if p.strip()]
    out = []
    for p in stmts:
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*=(?!=)(.*)$", p, re.S)
        out.append(f"{m.group(1)} = {_ternary(m.group(2))}" if m else _ternary(p))
    return "\n".join(out)


# --------------------------------------------------------------------------
# Compiler
# --------------------------------------------------------------------------

class _Subst(ast.NodeTransformer):
    def __init__(self, env):
        self.env = env

    def visit_Name(self, node):
        if node.id in self.env:
            return copy.deepcopy(self.env[node.id])
        return node


class _Resolve(ast.NodeTransformer):
    def __init__(self, bindings, params):
        self.b, self.p = bindings, params

    def visit_Name(self, node):
        if node.id in self.b:
            return ast.Name(id=self.b[node.id], ctx=ast.Load())
        if node.id in self.p:
            return ast.Constant(self.p[node.id])
        return node


class Compiler:
    def __init__(
        self,
        library: Optional[OperatorLibrary] = None,
        prep_plan: Optional[Callable[[str], list]] = None,
        max_lookback: int = DEFAULT_MAX_LOOKBACK,
        allowed: Optional[Callable[[str, str], bool]] = None,
        known_fields: Optional[Iterable[str]] = None,
    ):
        self.lib = library or OperatorLibrary()
        self.prep_plan = prep_plan or (lambda f: [])
        self.max_lookback = max_lookback
        self.allowed = allowed or (lambda name, category: True)
        self.known_fields = set(known_fields) if known_fields else None

    # ------------------------------------------------------------- public
    def compile(self, expr: str, bindings: Optional[dict] = None, params: Optional[dict] = None,
                name: Optional[str] = None, meta: Optional[dict] = None) -> CompiledAlpha:
        self.b = dict(bindings or {})
        self.p = dict(params or {})
        self.lines: list[str] = []
        self.memo: dict = {}
        self.k = 0
        self.need: dict[str, int] = {}
        self.order: list[str] = []
        self.store: list[str] = []
        self.ops_used: list[str] = []
        self.warn: list[str] = []
        self.nstate = 0

        try:
            tree = ast.parse(fe_to_python(expr))
        except SyntaxError as e:
            raise CompileError(f"syntax error: {e.msg} (in: {expr})")
        env, final = {}, None
        for st in tree.body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                env[st.targets[0].id] = _Subst(env).visit(st.value)
                final = env[st.targets[0].id]
            elif isinstance(st, ast.Expr):
                final = _Subst(env).visit(st.value)
            else:
                raise CompileError(f"unsupported statement: {ast.unparse(st)}")
        if final is None:
            raise CompileError("empty expression")

        out = self._emit(final, 1)
        if not out.startswith("_t"):
            raise CompileError("expression is constant (uses no data field)")

        load_lines, data_ids, lookback, prep_used = self._loads()
        body_text = "\n".join(load_lines + self.lines)
        names = {n.id for n in ast.walk(ast.parse("def _f():\n" + (body_text or "    pass")))
                 if isinstance(n, ast.Name)}
        names |= {"clean", "finalize"}
        ops = self.lib.resolve(names)

        resolved = ast.unparse(_Resolve(self.b, self.p).visit(copy.deepcopy(final)))
        h = hashlib.md5(f"{resolved}|{prep_used}".encode()).hexdigest()[:8]
        slug = "_".join(dict.fromkeys(re.sub(r"\W", "", o) for o in self.ops_used))[:28] or "alpha"
        fn = name or f"a_{slug}_{h}"
        if not fn.isidentifier() or keyword.iskeyword(fn):
            fn = f"a_{h}"
        prep_txt = "; ".join(f"{f}: {' > '.join(['clean'] + s)}" for f, s in prep_used.items())
        head = (f"# expression: {' '.join(expr.split())}\n# resolved:   {resolved}\n"
                f"# prep:       {prep_txt}\n# lookback:   {lookback}\n")
        store_src = "[" + ", ".join(self.store) + "]"
        data_src = "[" + ", ".join(f'"{d}"' for d in data_ids) + "]"
        code = (head + _HEADER + "\n\n" + self.lib.inline(ops) + "\n\n\n"
                f"@alpha(\n    data={data_src},\n    store={store_src},\n)\n"
                f"def {fn}(data, store) -> npt.NDArray[np.float32]:\n"
                + body_text + f"\n    return finalize({out}, data.universe[-1])\n")
        ast.parse(code)
        return CompiledAlpha(fn, code, lookback, expr, resolved, dict(self.b), list(self.order),
                             prep_used, list(dict.fromkeys(self.ops_used)), self.warn, dict(meta or {}))

    # ------------------------------------------------------------- helpers
    def _const(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.p:
                return self.p[node.id]
            if node.id in ("True", "False", "None"):
                return {"True": True, "False": False, "None": None}[node.id]
            raise _NotConst
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = self._const(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = self._const(node.left), self._const(node.right)
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                   ast.Div: lambda: a / b, ast.FloorDiv: lambda: a // b, ast.Pow: lambda: a ** b}
            if type(node.op) in ops:
                return ops[type(node.op)]()
        raise _NotConst

    def _is_const(self, node) -> bool:
        try:
            self._const(node)
            return True
        except _NotConst:
            return False

    def _new(self, code: str) -> str:
        self.k += 1
        v = f"_t{self.k}"
        self.lines.append(f"    {v} = {code}")
        return v

    def _field(self, fid: str, n: int) -> str:
        if fid == "universe":
            raise CompileError("'universe' is provided automatically; it cannot be used as a field")
        if fid in GROUP_FIELD_NAMES:
            self.warn.append(f"'{fid}' is a GROUP field and may not load in Python Alphas; "
                             f"bucket(rank(cap), 20) or cluster_neutralize(...) always work")
        if self.known_fields is not None and fid not in self.known_fields:
            self.warn.append(f"field '{fid}' is not in the loaded field list (check spelling)")
        if fid not in self.need:
            self.order.append(fid)
        self.need[fid] = max(self.need.get(fid, 0), n)
        key = ("field", fid, n)
        if key not in self.memo:
            self.memo[key] = self._new(f"_tail(fv_{_ident(fid)}, {n})")
        return self.memo[key]

    # ------------------------------------------------------------- emit
    def _emit(self, node, n: int) -> str:
        if self._is_const(node):
            v = self._const(node)
            return "np.nan" if isinstance(v, float) and v != v else repr(v)
        key = (ast.dump(node), n)
        if key in self.memo:
            return self.memo[key]
        out = self._emit_inner(node, n)
        self.memo[key] = out
        return out

    def _emit_inner(self, node, n: int) -> str:
        if isinstance(node, ast.Name):
            nid = node.id
            if nid in ("nan", "NaN"):
                return "np.nan"
            if nid in self.b:
                return self._field(self.b[nid], n)
            if nid == "market":
                return "np.zeros(data.universe.shape[1])"
            if PLACEHOLDER_RE.match(nid):
                raise CompileError(f"placeholder {nid} has no field assigned")
            if self.lib.spec(nid) or nid in STATEFUL:
                raise CompileError(f"'{nid}' is an operator; call it like {nid}(...)")
            return self._field(nid, n)

        if isinstance(node, ast.UnaryOp):
            c = self._emit(node.operand, n)
            if isinstance(node.op, ast.USub):
                return self._row("neg", [c])
            if isinstance(node.op, ast.UAdd):
                return c
            return self._row("not_", [c])

        if isinstance(node, ast.BinOp):
            m = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply", ast.Div: "divide",
                 ast.Pow: "power", ast.BitAnd: "and_", ast.BitOr: "or_"}
            if type(node.op) not in m:
                raise CompileError(f"operator '{ast.unparse(node)}' is not supported")
            return self._row(m[type(node.op)], [self._emit(node.left, n), self._emit(node.right, n)])

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise CompileError("chained comparisons (a < b < c) are not supported")
            m = {ast.Gt: "gt", ast.Lt: "lt", ast.GtE: "ge", ast.LtE: "le", ast.Eq: "eq", ast.NotEq: "ne"}
            return self._row(m[type(node.ops[0])],
                             [self._emit(node.left, n), self._emit(node.comparators[0], n)])

        if isinstance(node, ast.BoolOp):
            op = "and_" if isinstance(node.op, ast.And) else "or_"
            acc = self._emit(node.values[0], n)
            for v in node.values[1:]:
                acc = self._row(op, [acc, self._emit(v, n)])
            return acc

        if isinstance(node, ast.IfExp):
            return self._row("if_else", [self._emit(node.test, n), self._emit(node.body, n),
                                         self._emit(node.orelse, n)])

        if isinstance(node, ast.Call):
            return self._call(node, n)

        raise CompileError(f"unsupported syntax: {ast.unparse(node)}")

    def _row(self, kit_name: str, args: list[str], kw: str = "") -> str:
        self._use(kit_name)
        return self._new(f"call_row({kit_name}, {', '.join(args)}{kw})")

    def _use(self, kit_name: str) -> None:
        sp = self.lib.specs.get(kit_name)
        fe = sp.fe_name if sp else kit_name
        cat = sp.category if sp else "stateful"
        if not self.allowed(fe, cat):
            raise CompileError(f"operator '{fe}' is blocked by your operator settings")
        self.ops_used.append(fe)

    def _bind(self, fname, node, params, defaults, vararg=False, varkw=False, pos_params=None):
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise CompileError("*args are not supported")
        got = {}
        pp = params if pos_params is None else pos_params
        if len(node.args) > len(pp) and not vararg:
            raise CompileError(f"{fname}() takes at most {len(pp)} positional arguments: {', '.join(params)}")
        pos = list(node.args)
        for p, a in zip(pp, pos):
            got[p] = a
        extra_pos = pos[len(pp):]
        for kw in node.keywords:
            if kw.arg is None:
                raise CompileError("**kwargs are not supported")
            if kw.arg not in params and not varkw:
                raise CompileError(f"{fname}() has no parameter '{kw.arg}'. Parameters: {', '.join(params)}")
            v = kw.value
            if isinstance(v, ast.Name) and v.id not in self.b and v.id not in self.p \
                    and v.id not in ("True", "False", "None", "nan") \
                    and not (self.known_fields and v.id in self.known_fields):
                v = ast.Constant(v.id)          # driver=gaussian -> "gaussian"
            got[kw.arg] = v
        return got, extra_pos

    def _window_value(self, fname, got, wname, defaults):
        if wname in got:
            try:
                return int(self._const(got[wname]))
            except _NotConst:
                raise CompileError(f"{fname}(): window '{wname}' must be a number")
        if wname in defaults and defaults[wname] is not None:
            return int(defaults[wname])
        raise CompileError(f"{fname}(): window '{wname}' is required")

    def _args_code(self, sp, got, extra_pos, n_child):
        """Positional params in order (until the first gap), then *args, then keywords."""
        def code(node):
            return self._emit(node, 1) if self._is_const(node) else self._emit(node, n_child)
        pos_code, kw_code, gap = [], [], False
        for p in sp.pos_params:
            if p in got and not gap:
                pos_code.append(code(got[p]))
            elif p in got:
                kw_code.append(f"{p}={code(got[p])}")
            else:
                gap = True
        if extra_pos and gap:
            raise CompileError(f"{sp.fe_name}(): give all parameters before the extra inputs")
        pos_code += [code(a) for a in extra_pos]
        for p in sp.params:
            if p not in sp.pos_params and p in got:
                kw_code.append(f"{p}={code(got[p])}")
        return pos_code, kw_code

    def _call(self, node, n: int) -> str:
        if not isinstance(node.func, ast.Name):
            raise CompileError(f"unsupported call: {ast.unparse(node)}")
        raw = node.func.id
        if raw.startswith("vec_"):
            raise CompileError(f"{raw}: VECTOR fields/operators are not supported in Python Alphas")
        if raw in IDENTITY_OPS:
            if not node.args:
                raise CompileError(f"{raw}() needs an argument")
            return self._emit(node.args[0], n)
        if raw in STATEFUL:
            return self._stateful(raw, node, n)
        sp = self.lib.spec(raw)
        if sp is None:
            sug = self.lib.suggest(raw)
            hint = f" Did you mean: {', '.join(sug)}?" if sug else ""
            raise CompileError(f"unknown operator '{raw}'.{hint}")
        if sp.error:
            raise CompileError(f"operator '{raw}' is defined incorrectly: {sp.error}")
        self._use(sp.name)
        got, extra_pos = self._bind(raw, node, sp.params, sp.defaults, sp.vararg, sp.varkw, sp.pos_params)

        if sp.kind == "ts":
            w = self._window_value(raw, got, sp.window, sp.defaults)
            if w < 1:
                raise CompileError(f"{raw}(): window must be >= 1")
            extra = sp.extra
            if isinstance(extra, str):
                extra = int(self._const(got[extra])) if extra in got else int(sp.defaults.get(extra) or 0)
            need = n + w - 1 + int(extra)
            got[sp.window] = ast.Constant(w)
            pos, kw = self._args_code(sp, got, extra_pos, need)
            return self._new(f"roll({sp.name}, {n}, {', '.join(pos + kw)})")

        if sp.kind == "series":
            w = self._window_value(raw, got, sp.window, sp.defaults) if (sp.window in got or sp.window in sp.defaults) else 0
            w = w or sp.unbounded
            pos, kw = self._args_code(sp, got, extra_pos, n + w)
            return self._new(f"_tail({sp.name}({', '.join(pos + kw)}), {n})")

        if sp.kind == "group":
            pos, kw = self._args_code(sp, got, extra_pos, n)
            return self._new(f"rows_apply({sp.name}, {', '.join(pos + kw)})")

        pos, kw = self._args_code(sp, got, extra_pos, n)
        return self._new(f"call_row({sp.name}, {', '.join(pos + kw)})")

    def _stateful(self, raw, node, n):
        if n != 1:
            raise CompileError(f"{raw}() keeps state day to day, so it cannot sit inside a "
                               f"time-series operator; put it at the outside of the expression")
        spec = STATEFUL[raw]
        self.ops_used.append(raw)
        if not self.allowed(raw, "stateful"):
            raise CompileError(f"operator '{raw}' is blocked by your operator settings")
        got, extra = self._bind(raw, node, spec["params"], spec["defaults"])
        if extra:
            raise CompileError(f"{raw}() takes: {', '.join(spec['params'])}")
        for p in spec["params"]:
            if p not in got:
                if p not in spec["defaults"]:
                    raise CompileError(f"{raw}() needs '{p}'")
                got[p] = ast.Constant(spec["defaults"][p])

        def c(p):
            try:
                return repr(self._const(got[p]))
            except _NotConst:
                raise CompileError(f"{raw}(): '{p}' must be a number")

        self.nstate += 1
        s = f"s{self.nstate}"
        decl = lambda nm: f'{{"name": "{nm}", "dims": "i", "extend": np.float64(np.nan)}}'
        if raw == "hump":
            x = self._emit(got["x"], 1)
            self.store.append(decl(s))
            v = self._new(f"hump_step({x}, store.{s}, {c('hump')})[np.newaxis, :]")
        elif raw == "ema":
            x = self._emit(got["x"], 1)
            self.store.append(decl(s))
            v = self._new(f"ema_step({x}, store.{s}, {c('a')})[np.newaxis, :]")
        elif raw == "trade_when":
            t, x, e = (self._emit(got[p], 1) for p in ("trigger", "x", "exit"))
            self.store.append(decl(s))
            v = self._new(f"trade_when_step({t}, {x}, {e}, store.{s})[np.newaxis, :]")
        else:  # cluster_neutralize
            d = int(self._const(got["d"]))
            x = self._emit(got["x"], 1)
            r = self._emit(got["r"], d)
            self.store.append(decl(s))
            self.store.append(f'"{s}_age"')
            self.lines += [
                f"    if store.{s} is None or store.{s}_age is None or store.{s}_age >= {c('refresh')}:",
                f"        store.{s} = kmeans_labels(_tail({r}, {d}).T, {c('k')})",
                f"        store.{s}_age = 0",
                f"    store.{s}_age = store.{s}_age + 1",
            ]
            return self._new(f"group_neutralize({x}, store.{s})[np.newaxis, :]")
        self.lines.append(f"    store.{s} = {v}[-1]")
        return v

    # ------------------------------------------------------------- loads + prep
    def _parse_step(self, step: str):
        if step.strip() in ("adjust_split", "adjust_split()"):
            return "adjust_split", {}
        if step.strip().startswith("adjust_split_inverse"):
            return "adjust_split", {"inverse": True}
        try:
            node = ast.parse(step.strip(), mode="eval").body
        except SyntaxError:
            raise CompileError(f"bad prep step '{step}'")
        if isinstance(node, ast.Name):
            fname, args, kws = node.id, [], {}
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            args = [ast.literal_eval(a) for a in node.args]
            kws = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        else:
            raise CompileError(f"bad prep step '{step}'")
        sp = self.lib.spec(fname)
        if sp is None or sp.kind not in ("row", "series"):
            raise CompileError(f"prep step '{fname}' must be a row or series operator")
        names = sp.params[1:]
        kws.update(dict(zip(names, args)))
        return sp.name, kws

    def _loads(self):
        lines, data_ids, lookbacks, prep_used = [], [], [], {}
        need_adj = False
        for fid in self.order:
            steps = [self._parse_step(st) for st in self.prep_plan(fid)]
            base = self.need[fid] - 1
            if base > self.max_lookback:
                raise CompileError(f"windows on '{fid}' need {base} days of history; "
                                   f"max lookback is {self.max_lookback}. Use shorter windows.")
            room = self.max_lookback - base
            used = 0
            var = f"fv_{_ident(fid)}"
            access = (f"data.{fid}" if fid.isidentifier() and not keyword.iskeyword(fid)
                      else f'getattr(data, "{fid}")')
            lines.append(f"    {var} = clean({access})")
            applied = []
            for nm, kw in steps:
                if nm == "adjust_split":
                    need_adj = True
                    inv = ", inverse=True" if kw.get("inverse") else ""
                    lines.append(f"    {var} = adjust_per_share({var}, _adj{inv})")
                    applied.append("adjust_split" + ("_inverse" if inv else ""))
                    continue
                if nm == "ffill":
                    want = int(kw.get("limit") or 250)
                    lim = min(want, room - used)
                    if lim < want:
                        self.warn.append(f"{fid}: forward-fill shortened {want}->{lim} days to fit max lookback")
                    if lim <= 0:
                        continue
                    used += lim
                    kw = {**kw, "limit": lim}
                args = "".join(f", {k}={v!r}" for k, v in kw.items())
                lines.append(f"    {var} = {nm}({var}{args})")
                applied.append(f"{nm}({', '.join(f'{k}={v!r}' for k, v in kw.items())})" if kw else nm)
            prep_used[fid] = applied
            data_ids.append(fid)
            lookbacks.append(base + used)
        if need_adj:
            if "adjfactor" not in data_ids:
                data_ids.append("adjfactor")
            lines.insert(0, "    _adj = data.adjfactor")
        return lines, data_ids, max(lookbacks or [0]), prep_used


def _ident(s: str) -> str:
    s = re.sub(r"\W+", "_", str(s))
    return s if s and not s[0].isdigit() else "f_" + s
