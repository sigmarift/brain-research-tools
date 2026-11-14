"""
research.py  -  one object that runs the whole research loop.

    rs = Research(CONFIG)
    rs.connect()        # reuse ~/secrets/session.pkl, read the platform's settings schema
    rs.load_fields()    # pool fields from all SOURCES, profile them, plan cleaning
    rs.generate()       # exactly max_alphas new, distinct, dry-run-checked alphas
    rs.simulate()       # batches of 10, 3 at a time; saved after each batch
    rs.results()        # every run so far, best first
    rs.refine()         # variants of the best, then generate/simulate again

Everything is logged in research_runs/, so nothing is simulated twice.
"""

from __future__ import annotations

import copy
import inspect
import itertools
import json
import random
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from ace_bridge import ace
import ace_python as acp
import fields as F
import llm as L
import local_runner as lr
import recipes as R
from compiler import CompileError, Compiler
from oplib import OperatorLibrary

HERE = Path(__file__).resolve().parent

DEFAULT_CONFIG = dict(
    offline=False,
    region="USA", delay=1, universe="TOP3000",
    sources=[{"dataset": "pv1", "select": "all"}],
    kinds={},
    directions=["explore"],
    expressions=[],
    roles={},
    compose=None,
    ai_idea="",
    ai_ideas_count=12,
    prep="auto",
    auto_split=True,
    allowed_ops="all",
    blocked_ops=[],
    max_alphas=20,
    variants=True,
    random_fill=True,
    settings={},
    settings_mode="spread",
    max_lookback=260,
    concurrency=3,
    batch_size=10,
    stop_if_batch_fails=True,
    dry_run=True,
    seed=0,
    llm={},
    runs_dir="research_runs",
    refresh_fields=False,
)

SETTING_PARAMS = set(inspect.signature(acp.generate_python_alpha).parameters) - {
    "code", "drop_keys", "extra_settings", "schema", "validate", "lookback"}
ALPHA_TAG = "ace_tag_py"   # tag put on every simulated alpha
STAT_COLS = ["sharpe", "fitness", "turnover", "returns", "drawdown", "margin", "longCount", "shortCount"]


def _say(msg: str) -> None:
    print(msg, flush=True)


class Research:
    def __init__(self, config: Optional[dict] = None):
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg.update(copy.deepcopy(config or {}))
        if isinstance(cfg["directions"], str):
            cfg["directions"] = [cfg["directions"]] if cfg["directions"] else []
        self.cfg = cfg
        self.runs = Path(cfg["runs_dir"])
        (self.runs / "code").mkdir(parents=True, exist_ok=True)
        self.log_path = self.runs / "log.csv"
        self.s = None
        self.schema = None
        self.lib = OperatorLibrary()
        self.pool = pd.DataFrame(columns=["id", "source", "kind"])
        self.plan = F.PrepPlan(self.pool, cfg["prep"], cfg["auto_split"], cfg["kinds"])
        self.compiler = None
        self.extra_ideas: list[dict] = []
        self.candidates: list[dict] = []
        self.round = int(self._log()["round"].max()) + 1 if len(self._log()) else 1
        self.llm_cfg = L.load_llm_config(cfg.get("llm"))
        self._rng = random.Random(cfg["seed"])

    # ================================================================ setup
    def connect(self):
        if self.cfg["offline"]:
            _say("OFFLINE: no API calls. You can load fields (from 'fields' lists), generate and dry-run.")
            return self
        import session_manager as sm
        self.s = sm.get_session()
        sm.install()
        _say(f"Session ready ({sm.session_time_left(self.s) // 60} min left).")
        try:
            self.schema = acp.get_settings_schema(self.s)
            lang = self.schema.get("language", {}).get("choices", set())
            _say(f"Platform settings schema loaded ({len(self.schema)} keys; PYTHON listed: {'PYTHON' in lang}).")
        except Exception as e:  # noqa: BLE001
            _say(f"Could not read settings schema ({e}); payloads will not be pre-checked.")
        if self.llm_cfg:
            _say(f"AI assistant: {self.llm_cfg['provider']} / {self.llm_cfg['model']}")
        return self

    def load_fields(self):
        c = self.cfg
        self.pool = F.build_pool(c["sources"], self.s, c["region"], c["delay"], c["universe"],
                                 offline=c["offline"] or self.s is None, cache_dir=self.runs / "fields",
                                 refresh=c["refresh_fields"], kind_overrides=c["kinds"], log=_say)
        self.plan = F.PrepPlan(self.pool, c["prep"], c["auto_split"], c["kinds"])
        known = set(self.pool["id"]) | set(F.STANDARD_FIELDS)
        self.compiler = Compiler(self.lib, self.plan, c["max_lookback"], self._allowed,
                                 known_fields=known if len(self.pool) else None)
        if len(self.pool):
            counts = self.pool.groupby(["source", "kind"]).size().rename("fields").reset_index()
            _say("Field pool: " + ", ".join(f"{r.source}/{r.kind}={r.fields}" for r in counts.itertuples()))
        return self.pool

    def _ensure(self):
        if self.compiler is None:
            self.load_fields()

    def _allowed(self, name: str, category: str) -> bool:
        allow = self.cfg["allowed_ops"]
        if name in self.cfg["blocked_ops"] or category in self.cfg["blocked_ops"]:
            return False
        if allow in ("all", None, "*"):
            return True
        allow = [allow] if isinstance(allow, str) else allow
        always = {"neg", "add", "subtract", "multiply", "divide"}   # plain arithmetic in expressions
        return name in allow or category in allow or name in always

    # ================================================================ browse
    def ops(self, search: str = "", category: Optional[str] = None):
        df = self.lib.catalog(search, category)
        df.insert(1, "allowed", [self._allowed(o, c) for o, c in zip(df.operator, df.category)])
        return df

    def directions(self):
        return R.list_directions()

    def fields(self):
        self._ensure()
        df = self.pool.copy()
        df["prep"] = [" > ".join(["clean"] + self.plan(f)) for f in df["id"]]
        return df

    def show_code(self, i: int = 0) -> None:
        print(self.candidates[i]["alpha"].code)

    # ================================================================ ideas
    def ideas(self) -> list[dict]:
        c = self.cfg
        out = []
        for d in c["directions"]:
            out += R.direction_ideas(d)
        for e in c["expressions"]:
            out.append(R.normalize_idea(e, c["roles"], "expression"))
        if c["compose"]:
            spec = dict(c["compose"])
            spec.setdefault("roles", c["roles"])
            out += R.compose_ideas(spec)
        out += self.extra_ideas
        return out

    def add_ideas(self, items, origin="added"):
        if isinstance(items, str):
            items = L.parse_ideas(items)
        new = [R.normalize_idea(i, self.cfg["roles"], origin) for i in items]
        self.extra_ideas += new
        _say(f"Added {len(new)} ideas.")
        return new

    # ================================================================ AI
    def _ai(self, prompt: str, what: str) -> Optional[str]:
        if not self.llm_cfg:
            _say(f"No AI key configured. Copy this prompt into any AI chat, then pass the reply to "
                 f"rs.paste_ai_reply(\"\"\"...\"\"\") ({what}).\n\n" + "-" * 70 + "\n" + prompt + "\n" + "-" * 70)
            return None
        _say(f"Asking {self.llm_cfg['provider']} ({what})...")
        return L.ask(prompt, self.llm_cfg, L.SYSTEM)

    def ai_ideas(self, text: Optional[str] = None, n: Optional[int] = None):
        """Turn a research direction in plain words into validated ideas."""
        self._ensure()
        text = text or self.cfg["ai_idea"]
        if not text:
            _say("Describe an idea first: rs.ai_ideas('...') or set CONFIG['ai_idea'].")
            return None
        prompt = L.ideas_prompt(text, self.pool, self.ops().query("allowed"), n or self.cfg["ai_ideas_count"])
        reply = self._ai(prompt, "ideas")
        return self.paste_ai_reply(reply) if reply else None

    def paste_ai_reply(self, text: str, repair: bool = True):
        ideas = L.parse_ideas(text)
        good, bad = self._validate_ideas(ideas)
        if bad and repair and self.llm_cfg:
            reply = L.ask(L.repair_prompt(bad, self.ops().query("allowed")), self.llm_cfg, L.SYSTEM)
            fixed_good, bad = self._validate_ideas(L.parse_ideas(reply or "[]"))
            good += fixed_good
        for b in bad:
            _say(f"  dropped: {b['expr']}  ({b['error']})")
        self.add_ideas(good, origin="ai")
        return pd.DataFrame(good)

    def _validate_ideas(self, ideas):
        self._ensure()
        good, bad = [], []
        for idea in ideas:
            idea = R.normalize_idea(idea, self.cfg["roles"], "ai")
            b = next(R.bindings_for(idea, self.pool, self._rng), None) if R.placeholders(idea["expr"]) else {}
            if b is None:
                bad.append({**idea, "error": "no field in the pool matches its roles"})
                continue
            try:
                self.compiler.compile(idea["expr"], b)
                good.append(idea)
            except CompileError as e:
                bad.append({**idea, "error": str(e)})
        return good, bad

    def ai_refine(self, n: int = 10):
        """Let the AI propose new ideas from the results so far."""
        res = self.results()
        if res.empty:
            _say("No simulated results yet: run rs.simulate() first.")
            return None
        reply = self._ai(L.results_prompt(res, self.ops().query("allowed"), n), "refine")
        return self.paste_ai_reply(reply) if reply else None

    def ai_operator(self, description: str, install: bool = True):
        """Have the AI write a new operator; it is checked and tested before it is saved to my_ops.py."""
        existing = [s.fe_name for s in self.lib.user_ops()]
        reply = self._ai(L.operator_prompt(description, existing), "operator")
        if not reply:
            return None
        return self.add_operator_reply(reply, description, install)

    def add_operator_reply(self, reply: str, description: str = "", install: bool = True):
        try:
            data = L._extract_json(reply)
            code, extra = data["code"], int(data.get("extra_rows", 0) or 0)
        except Exception:  # noqa: BLE001
            code, extra = reply, 0
        name, problems = L.check_operator_code(code)
        if not problems:
            ns = {"np": __import__("numpy")}
            exec(compile(self.lib.inline(self.lib.resolve(set(self.lib.sources))), "<ops>", "exec"), ns)
            problems = L.test_operator(code, name, ns)
        print(code)
        if problems:
            _say("Operator rejected:\n  - " + "\n  - ".join(problems))
            return None
        _say(f"Operator '{name}' passed checks.")
        if install:
            L.install_operator(code, name, extra, description, HERE / "my_ops.py")
            self.lib = OperatorLibrary()
            if self.compiler:
                self.compiler.lib = self.lib
            _say(f"Saved to my_ops.py; use it in expressions as {name}(...).")
        return name

    # ================================================================ settings
    def _combos(self) -> list[dict]:
        ck = json.dumps(self.cfg["settings"], sort_keys=True, default=str) + str(bool(self.schema))
        if getattr(self, "_combo_cache", (None,))[0] == ck:
            return self._combo_cache[1]
        grid = self.cfg["settings"] or {}
        keys = list(grid)
        lists = [v if isinstance(v, (list, tuple)) else [v] for v in grid.values()]
        combos = [dict(zip(keys, c)) for c in itertools.product(*lists)] if keys else [{}]
        if self.schema:
            ok = []
            for cb in combos:
                p = acp.generate_python_alpha("x", validate=False, **self._base_settings(cb))
                issues = [i for i in acp.check_payload(p, self.schema) if "not in platform choices" in i]
                if issues:
                    _say(f"  settings {cb} skipped: {'; '.join(issues)}")
                else:
                    ok.append(cb)
            combos = ok or [{}]
        _say(f"Settings grid: {len(combos)} combination(s), mode '{self.cfg['settings_mode']}'.")
        self._combo_cache = (ck, combos)
        return combos

    def _base_settings(self, combo: dict, idea_settings: Optional[dict] = None) -> dict:
        c = self.cfg
        out = {"region": c["region"], "delay": c["delay"], "universe": c["universe"]}
        out.update({k: v for k, v in (idea_settings or {}).items() if k not in (c["settings"] or {})})
        out.update(combo)
        extra = {k: out.pop(k) for k in list(out) if k not in SETTING_PARAMS}
        if extra:
            out["extra_settings"] = extra
        return out

    # ================================================================ generate
    def _log(self) -> pd.DataFrame:
        if self.log_path.exists():
            try:
                return pd.read_csv(self.log_path)
            except Exception:  # noqa: BLE001
                pass
        return pd.DataFrame(columns=["round", "status", "settings_key"])

    def generate(self, n: Optional[int] = None, ideas: Optional[list] = None, quiet=False):
        """Build exactly n (default max_alphas) new alpha+settings candidates."""
        self._ensure()
        c = self.cfg
        target = int(n or c["max_alphas"])
        ideas = ideas if ideas is not None else self.ideas()
        if not ideas and not (c["random_fill"] and len(self.pool)):
            _say("No ideas and no fields: set directions, expressions, compose or ai_idea, and at least one source.")
            return pd.DataFrame()
        log = self._log()
        done = set(log.loc[log["status"] == "ok", "settings_key"].astype(str)) if len(log) else set()
        combos = self._combos()
        stream = R.candidate_stream(ideas, self.pool, self.lib, self._allowed, c["seed"] + self.round,
                                    c["variants"], c["random_fill"])
        out, seen_code, seen_sk = [], set(), set()
        stats = {"tried": 0, "compile errors": 0, "duplicates": 0, "already simulated": 0,
                 "failed dry-run": 0}
        errors, level_count, ci = [], {}, 0
        t0 = time.time()
        for expr, b, info in stream:
            if len(out) >= target or stats["tried"] >= max(400, 60 * target):
                break
            stats["tried"] += 1
            try:
                a = self.compiler.compile(expr, b, meta=info)
            except CompileError as e:
                if "blocked by your operator settings" in str(e):
                    stats["used blocked operators"] = stats.get("used blocked operators", 0) + 1
                    continue
                stats["compile errors"] += 1
                if info["level"] < 3 and len(errors) < 15:
                    errors.append(f"{expr} {b or ''}: {e}")
                continue
            if a.key in seen_code:
                stats["duplicates"] += 1
                continue
            seen_code.add(a.key)
            if c["dry_run"]:
                kinds = {f: F.DRYRUN_KIND.get(self.plan.profile(f)["kind"], "dense") for f in a.fields}
                r = lr.dry_run(a.code, lookback=a.lookback, n_days=a.lookback + 60,
                               n_instruments=120, field_kinds=kinds, seed=len(seen_code))
                if not r.ok or r.coverage < 0.05:
                    stats["failed dry-run"] += 1
                    if len(errors) < 15:
                        errors.append(f"{a.resolved}: dry-run {r.error or 'no coverage'}")
                    continue
            if c["settings_mode"] == "all":
                mine = combos
            elif c["settings_mode"] == "random":
                mine = [self._rng.choice(combos)]
            else:
                mine = [combos[ci % len(combos)]]
                ci += 1
            for cb in mine:
                if len(out) >= target:
                    break
                st = self._base_settings(cb, info.get("idea_settings"))
                sk = a.key + "|" + json.dumps(st, sort_keys=True, default=str)
                if sk in done:
                    stats["already simulated"] += 1
                    continue
                if sk in seen_sk:
                    continue
                seen_sk.add(sk)
                out.append({"alpha": a, "settings": st, "settings_key": sk})
                level_count[info["level"]] = level_count.get(info["level"], 0) + 1
        self.candidates = out
        names = {0: "from ideas", 1: "window variants", 2: "wrapper variants", 3: "random fill"}
        made = ", ".join(f"{v} {names[k]}" for k, v in sorted(level_count.items()))
        skipped = ", ".join(f"{v} {k}" for k, v in stats.items() if k != "tried" and v)
        _say(f"Generated {len(out)}/{target} alphas in {time.time() - t0:.0f}s ({made or 'none'}). "
             f"Skipped: {skipped or 'nothing'}.")
        if len(out) < target:
            _say("  Not enough distinct candidates: add sources/fields, directions or expressions, "
                 "allow more operators, or set variants/random_fill to True.")
        if errors and not quiet:
            _say("  First problems:\n    " + "\n    ".join(errors[:8]))
        return self.table()

    def table(self) -> pd.DataFrame:
        rows = []
        for i, cnd in enumerate(self.candidates):
            a, st = cnd["alpha"], cnd["settings"]
            rows.append({"#": i, "resolved": a.resolved, "lookback": a.lookback,
                         "neut": st.get("neutralization"), "universe": st.get("universe"),
                         "decay": st.get("decay"), "origin": a.meta.get("origin"),
                         "why": a.meta.get("why"), "warnings": "; ".join(a.warnings)})
        return pd.DataFrame(rows)

    # ================================================================ simulate
    def simulate(self, candidates: Optional[list] = None, concurrency: Optional[int] = None,
                 batch_size: Optional[int] = None):
        cands = candidates if candidates is not None else self.candidates
        if not cands:
            _say("Nothing to simulate: run rs.generate() first.")
            return pd.DataFrame()
        if self.cfg["offline"] or self.s is None:
            _say("Offline: skipping simulation. Code is ready in rs.candidates.")
            return pd.DataFrame()
        items = []
        for cnd in cands:
            st = dict(cnd["settings"])
            p = cnd["alpha"].payload(schema=self.schema, **st)
            items.append((json.dumps(p, sort_keys=True), p, cnd))
        by_key = {k: cnd for k, _, cnd in items}
        payloads = [p for _, p, _ in items]
        cfg = {**ace.DEFAULT_CONFIG}
        conc = int(concurrency or self.cfg["concurrency"])
        if conc > 8:
            _say(f"concurrency {conc} is above the ACE maximum of 8; using 8.")
            conc = 8
        conc = max(1, conc)
        bs = max(1, int(batch_size or self.cfg["batch_size"]))
        batches = [payloads[i:i + bs] for i in range(0, len(payloads), bs)]
        _say(f"Simulating {len(payloads)} alphas in {len(batches)} batch(es) of up to {bs}, "
             f"{conc} at a time (a new one starts as soon as a slot frees up).")
        done_ok = 0
        for b, batch in enumerate(batches, 1):
            t0 = time.time()
            res = acp.simulate_python_alpha_list(self.s, batch, min(conc, len(batch)), cfg)
            rows = self._record(res, by_key)
            ok = [r for r in rows if r["status"] == "ok"]
            done_ok += len(ok)
            best = max((r.get("sharpe") for r in ok if r.get("sharpe") is not None), default=None)
            _say(f"Batch {b}/{len(batches)} done in {(time.time() - t0) / 60:.1f} min: "
                 f"{len(ok)}/{len(batch)} ok" + (f", best sharpe {best:.2f}" if best is not None else "")
                 + f"  | total ok {done_ok}")
            if not ok and self.cfg["stop_if_batch_fails"]:
                _say("Every alpha in this batch failed, so the remaining batches were NOT submitted. "
                     "The platform's reply is in the warnings above. Common causes: a setting the "
                     "platform rejects (see rs.schema), a field missing in this region/universe, "
                     "or a GROUP field.")
                break
        self.round += 1
        return self.results(this_round=True, round_no=self.round - 1)

    def _record(self, results: list, by_key: dict) -> list:
        rows = []
        for r in results:
            sd = r.get("simulate_data") or {}
            cnd = by_key.get(json.dumps(sd, sort_keys=True))
            if cnd is None:   # fall back to matching code + settings values
                for k, v in by_key.items():
                    if v["alpha"].code == sd.get("regular") and json.loads(k).get("settings") == sd.get("settings"):
                        cnd = v
                        break
            if cnd is None:
                continue
            a, st = cnd["alpha"], cnd["settings"]
            code_path = self.runs / "code" / f"{a.key}.py"
            code_path.write_text(a.code, encoding="utf-8")
            row = {"time": time.strftime("%Y-%m-%d %H:%M"), "round": self.round,
                   "status": "ok" if r.get("alpha_id") else "error", "alpha_id": r.get("alpha_id"),
                   "name": a.name, "key": a.key, "settings_key": cnd["settings_key"],
                   "origin": a.meta.get("origin"), "level": a.meta.get("level"),
                   "idea": a.meta.get("idea"), "resolved": a.resolved,
                   "bindings": json.dumps(a.bindings), "settings": json.dumps(st, default=str),
                   "neut": st.get("neutralization"), "universe": st.get("universe"),
                   "decay": st.get("decay"), "lookback": a.lookback, "why": a.meta.get("why"),
                   "code_path": str(code_path)}
            is_ = r.get("is_stats")
            if isinstance(is_, pd.DataFrame) and not is_.empty:
                for col in STAT_COLS:
                    if col in is_.columns:
                        row[col] = is_[col].iloc[0]
            tests = r.get("is_tests")
            if isinstance(tests, pd.DataFrame) and "result" in tests.columns:
                row["fails"] = ",".join(tests.loc[tests["result"] == "FAIL", "name"].astype(str))
            if row["alpha_id"]:          # tag every successful simulation
                fe = a.meta.get("fe_alpha_id")
                tags = [ALPHA_TAG] + ([f"FE_{fe}"] if fe else [])
                kw = {"regular_desc": f"Python translation of Fast Expression alpha {fe}."} if fe else {}
                try:
                    ace.set_alpha_properties(self.s, row["alpha_id"], tags=tags, **kw)
                except Exception as e:  # noqa: BLE001
                    _say(f"  could not tag {row['alpha_id']}: {e}")
            rows.append(row)
        if rows:
            df = pd.concat([self._log(), pd.DataFrame(rows)], ignore_index=True)
            df.to_csv(self.log_path, index=False)
        return rows

    # ================================================================ results
    def results(self, this_round: bool = False, round_no: Optional[int] = None, ok_only=True):
        df = self._log()
        if df.empty:
            return df
        if this_round:
            df = df[df["round"] == (round_no or df["round"].max())]
        if ok_only:
            df = df[df["status"] == "ok"]
        cols = [c for c in ["round", "alpha_id", "resolved", "neut", "universe", "decay", *STAT_COLS[:5],
                            "fails", "origin", "why"] if c in df.columns]
        sort = "fitness" if "fitness" in df.columns else None
        df = df.sort_values(sort, ascending=False) if sort else df
        return df[cols].reset_index(drop=True)

    # ================================================================ refine
    def refine(self, top: int = 5, n: Optional[int] = None, min_sharpe: float = 0.5):
        """Rule-based next round from the best results (or flipped losers), then generate()."""
        df = self.results()
        if df.empty:
            _say("No simulated results yet: run rs.simulate() first (needs offline=False).")
            return pd.DataFrame()
        df = df.copy()
        df["abs_sharpe"] = df["sharpe"].abs() if "sharpe" in df.columns else 0
        best = df.sort_values("abs_sharpe", ascending=False).head(top)
        ideas = []
        for r in best.to_dict("records"):
            e, sh, to = r["resolved"], r.get("sharpe", 0) or 0, r.get("turnover", 0) or 0
            base = {"roles": {}, "origin": "refine", "settings": {}}
            if sh < -min_sharpe:
                ideas.append({**base, "expr": f"-({e})", "why": f"flip: sharpe was {sh:.2f}"})
                e = f"-({e})"
            ideas.append({**base, "expr": e, "why": f"base sharpe {abs(sh):.2f}"})
            if to > 0.4:
                ideas.append({**base, "expr": f"ts_decay_linear({e}, 5)", "why": f"turnover {to:.2f} too high"})
                ideas.append({**base, "expr": f"hump({e}, 0.01)", "why": f"turnover {to:.2f} too high"})
            ideas.append({**base, "expr": f"group_neutralize({e}, bucket(rank(cap), 10))",
                          "why": "size-neutral version"})
            for v, w in R.window_variants(e, self.lib, [0.5, 2.0]):
                ideas.append({**base, "expr": v, "why": w})
        _say(f"Refining {len(best)} best alphas into {len(ideas)} ideas.")
        return self.generate(n or self.cfg["max_alphas"], ideas=ideas)

    # ================================================================ output
    def save_best(self, n: int = 5, folder: str = "python_alphas", tag: bool = True):
        df = self.results().head(n)
        if df.empty:
            _say("No simulated results to save yet.")
            return []
        log = self._log().set_index("alpha_id")
        saved = []
        for aid in df["alpha_id"]:
            src = Path(log.loc[aid, "code_path"] if not isinstance(log.loc[aid], pd.DataFrame)
                       else log.loc[aid].iloc[0]["code_path"])
            dst = Path(folder)
            dst.mkdir(exist_ok=True)
            out = dst / f"{aid}.py"
            out.write_text(src.read_text(), encoding="utf-8")
            saved.append(str(out))
            if tag and self.s is not None:
                try:
                    ace.set_alpha_properties(self.s, aid, tags=[ALPHA_TAG])
                except Exception:  # noqa: BLE001
                    pass
        _say(f"Saved {len(saved)} alphas to {folder}/ (paste into the platform UI if you like).")
        return saved

    def run(self, rounds: int = 1, refine_top: int = 5):
        """Everything in one call: generate -> simulate, then refine rounds."""
        self.connect()
        self.load_fields()
        self.generate()
        res = self.simulate()
        for _ in range(rounds - 1):
            if self.results().empty:
                break
            self.refine(refine_top)
            res = self.simulate()
        return self.results()
