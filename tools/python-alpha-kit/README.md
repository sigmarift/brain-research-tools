# Python Alpha Research Kit (ACE API)

Build, check, simulate and refine BRAIN **Python Alphas** without writing Python. You choose a direction. The system picks fields, cleans and backfills them, writes self-contained alpha code, tests it, simulates it, and ranks the results.

## Start

1. **Add the ACE files.** Copy `ace_lib.py` and `helpful_functions.py` (from the BRAIN ACE API package) into the repo's shared [`ace/`](../../ace) folder. They're not included in this repo. Offline mode works without them.
2. Install the requirements and open the notebook:

```bash
pip install -r requirements.txt
jupyter notebook python_alphas.ipynb
```

Edit the **CONFIG** cell and run all cells.

- **Login:** reuses `~/secrets/session.pkl` while it has time left. Otherwise it logs in with ACE's normal flow (`~/secrets/platform-brain.json`) and saves the new session.
- **AI (optional):** run `import llm; llm.save_llm_config("anthropic", "sk-ant-...")` once. `openai` works too, and so does `openai_compatible` with `base_url=` (OpenRouter, DeepSeek, a local server). You can also set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in the environment instead. Without a key, AI steps print a prompt you can paste into any chat.

## What you control in CONFIG

| Key | What it does |
|---|---|
| `sources` | Any number of field sources, all pooled. Each takes `dataset`, `prefix`, `contains`, `regex`, `exclude`, `min_coverage`, `search`, `select` (`all`, `random`, `least_used`, `most_used`, `top_coverage`), `n`, `seed`, `kind`, `name`, or a plain `fields` list |
| `directions` | One or more of: reversion, momentum, volume, volatility, value, quality, growth, analyst, sentiment, model, ml, explore |
| `expressions` | Your own FE-style ideas. Placeholders `X`/`Y`/`Z` are filled from the pool, and real field names work too. Add `fe_alpha_id` to translate an FE alpha and tag it |
| `compose` | An operator grid: `{"start": ["X"], "layers": [["rank(_)", "zscore(_)"], ["ts_rank(_, 20)"], ["_", "-_"]]}` |
| `ai_idea` | An idea in plain words. The AI turns it into expressions, and every one is compiled before use (broken ones are sent back once for repair) |
| `roles` | Which fields may fill a placeholder: `"fundamental"`, `"analyst@analyst4"`, `"@pv1"`, `"dense"`, `"slow"`, `"any"`, or a list of fields |
| `max_alphas` | Exactly this many **new** alphas per `generate()` (see below) |
| `settings` | A grid: any value given as a list is tried, e.g. `{"neutralization": ["MARKET", "SECTOR"], "universe": ["TOP3000", "TOP1000"], "decay": [0, 4]}`. Combinations the platform doesn't accept are dropped before submitting |
| `concurrency` / `batch_size` | Simulations at once (1 to 8, default 3) and alphas per batch (default 10) |
| `settings_mode` | `spread` rotates combinations across alphas; `all` crosses every alpha with every combination; `random` picks one at random |
| `prep` | `minimal`, `auto` (default), `robust`, or a dict per field, kind, `@source` or `*`, e.g. `{"fundamental": ["auto", "winsorize(std=3)"]}` |
| `allowed_ops` / `blocked_ops` | Operator names or categories (time-series, cross-sectional, group, arithmetic, logic, prep, ml, stateful, custom) |
| `kinds` | Override any field's detected kind |

## How the target count is guaranteed

`generate()` keeps producing candidates until it has `max_alphas` new, distinct ones that pass the dry-run. It works through these sources in order:
1. Your ideas filled with pooled fields, rotating so each idea starts on a different field.
2. Window variants (x0.5, x1.5, x2 ...).
3. Wrapper variants (rank, sign flip, decay, size-bucket neutral, winsorized z-score, hump).
4. Random combinations of your allowed operators.

Duplicates and anything already simulated (per `research_runs/log.csv`) are skipped. If even that can't reach the target, for example with one field and every operator blocked, it tells you why.

## What the compiler does for you

- **FE syntax:** `a = ...; b = ...;` statements, `cond ? a : b`, `&&` `||` `!`, keyword arguments, and FE names (`abs`, `max`, `ts_std_dev`, `reverse` ...).
- **Nested time-series operators:** they behave as in FE. Tested against pandas, they match to float32 precision.
- **Lookback:** computed automatically as the minimum the expression needs, capped at `max_lookback` (260). Backfill is shortened to fit if needed.
- **Automatic cleaning:**
  - every field: missing-value markers and inf become NaN;
  - per-share prices: split adjustment (share counts get the inverse);
  - forward-fill: fundamentals 250 days, analyst and sparse data 66, models 22, news and events 5.
- **Stateful ops:** `hump`, `ema`, `trade_when` and `cluster_neutralize` get their `store` entries written for you.
- **Lean code:** only the operators an alpha uses are inlined, so each alpha is one self-contained file.

## Operators

There are 102: 40 time-series, 12 cross-sectional, 6 group, 19 arithmetic, 11 logic, 6 prep, 2 ML (`ts_pca_neutralize`, `ts_fft_smooth`) and 4 stateful, plus your own. Browse them with `rs.ops()` or `rs.ops("volatility")`.

**Your own operators:** add a function to `my_ops.py`, or run `rs.ai_operator("what it should do")`. AI-written code is checked against a strict whitelist (numpy only; no imports, file access or dunder access) and tested on windows with gaps before it's saved.

## Research loop

```python
rs.generate(); rs.simulate(); rs.results()
rs.refine(top=5); rs.simulate()      # rules: flip negatives, cut high turnover, size-neutral, window variants
rs.ai_refine()                        # AI suggests new ideas from your results
rs.save_best(5)                       # .py files ready for the platform UI
rs.run(rounds=2)                      # all of the above in one call
```

Simulations run in batches of `batch_size` (10), with `concurrency` (3, max 8) running at once: a batch of 10 goes 3, 3, 3, 1, and a new one starts the moment a slot frees up. After each batch the results are saved to the log, tagged `ace_tag_py` and summarized, so an interrupted run loses at most one batch. If every alpha in a batch fails (usually a setting or field the platform rejects), the remaining batches are not submitted.

## Files

| File | Role |
|---|---|
| `python_alphas.ipynb` | The notebook (edit CONFIG only) |
| `research.py` | The research loop |
| `compiler.py` | FE text to Python Alpha |
| `python_ops.py`, `my_ops.py`, `oplib.py` | Operators and their metadata |
| `fields.py` | Sources, field-kind detection, prep plans |
| `recipes.py` | Directions, roles, variants, the candidate stream |
| `llm.py` | AI providers, prompts, validation |
| `local_runner.py` | Local dry-run that follows the engine's rules |
| `ace_python.py`, `session_manager.py` | Python payloads, settings schema, session reuse |
| `ace_bridge.py` | Finds the ACE files (shared `ace/` folder, this folder, or `ACE_LIB_DIR`) and explains how to add them if missing |
| `tests/` | Offline tests: `pytest -q` (no account or ACE files needed) |

## Tested here, and not yet

- **Tested:**
  - all 102 operators compile and run without warnings;
  - nested expressions match pandas;
  - multi-source pools hit exact target counts, with blocked operators respected and settings spread across combinations;
  - AI wiring (mocked providers), idea repair, and operator safety checks;
  - the full loop against a mocked BRAIN API: schema filtering, batching at the set concurrency, stop after a fully failed batch, results, refine, no re-simulation, saving;
  - the notebook end to end offline.
- **Not yet:** a run against the live BRAIN API. That's why a fully failed batch stops the run.
- **Limits:**
  - GROUP fields (sector, industry) may not load in Python Alphas; use `bucket(rank(cap), 10)`, `market` or `cluster_neutralize`.
  - Fields are fetched for the top-level universe. Other universes in the settings grid assume the same fields exist there.
