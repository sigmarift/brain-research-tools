# Contributing

Thanks for helping. Anything that makes alpha research easier for the community is welcome.

## Ways to help

- **Report a bug:** open an issue with what you ran, what you expected and what happened. Paste the error, but remove any credentials or alpha IDs you don't want public.
- **Add an operator** to `tools/python-alpha-kit/python_ops.py`:
  - numpy only, no imports;
  - never modify inputs;
  - NaN-safe;
  - time-series ops start with `ts_` and take a window parameter `d`.
  - The tests pick up new operators automatically.
- **Add a research direction:** append to `DIRECTIONS` in `tools/python-alpha-kit/recipes.py`. Give each idea an expression, roles for its placeholders and a one-line reason.
- **Improve the docs:** clearer READMEs, examples, and notes on what you found when running against the live platform.
- **Add a new tool:** create `tools/<tool-name>/` with its own README, requirements and `tests/`. Load ACE through a bridge like `tools/python-alpha-kit/ace_bridge.py`, so it finds the shared `ace/` folder and never bundles the files.

## Workflow

1. Fork the repo and create a branch: `git checkout -b my-change`
2. Make your change and run the tests for that tool:
   ```bash
   cd tools/python-alpha-kit
   pip install -r requirements.txt -r requirements-dev.txt
   pytest -q
   ```
   The tests run offline and need no BRAIN account or ACE files.
3. Open a pull request saying what changed and why. If you tested against the live platform, say so and describe the result.

## Ground rules

- **Never commit:**
  - WorldQuant's ACE files;
  - credentials, session pickles or API keys;
  - private alpha expressions you don't mean to share.
- Keep the project unofficial and neutral: don't describe it as WorldQuant's or endorsed by WorldQuant.
- Be kind and specific in reviews and issues.
