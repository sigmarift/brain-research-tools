# BRAIN Research Tools

Open-source tools that help WorldQuant BRAIN consultants research alphas faster. Built in Kenya, for anyone in the community to use, improve and share back.

> **Unofficial.** This is a community project. It is not affiliated with, endorsed by or supported by WorldQuant. You use it under your own BRAIN account and the platform's terms.

## Tools

| Tool | What it does | Status |
|---|---|---|
| [python-alpha-kit](tools/python-alpha-kit) | Turns research directions or Fast Expression style ideas into tested BRAIN Python Alphas, then simulates them in batches, ranks the results and refines the best. Supports multiple datasets, settings grids, 100+ operators and optional AI help | Beta |

More tools will be added here.

## Quick start

```bash
git clone https://github.com/sigmarift/brain-research-tools.git
cd brain-research-tools
```

1. **Add the ACE files.** Copy `ace_lib.py` and `helpful_functions.py` from the BRAIN ACE API package into [`ace/`](ace). They're not included here (see below).
2. **Add your BRAIN credentials** where ACE expects them (`~/secrets/platform-brain.json`). Never put credentials in this repo.
3. **Open a tool** and follow its README, e.g. [`tools/python-alpha-kit`](tools/python-alpha-kit).

## Why the ACE files aren't included

`ace_lib.py` and `helpful_functions.py` are WorldQuant's files, and they carry no license that allows redistribution. Each user adds them locally, in one shared place: the [`ace/`](ace) folder. Tools find them there automatically, and `.gitignore` keeps them out of commits. Offline features work without them.

## Repository layout

```
brain-research-tools/
├── ace/                    <- put ace_lib.py + helpful_functions.py here (not committed)
├── tools/
│   └── python-alpha-kit/   <- one folder per tool, each with its own README and tests
├── .github/workflows/      <- runs every tool's offline tests on each push and PR
├── CONTRIBUTING.md
└── LICENSE                 <- MIT (covers this repo's code only)
```

## Contributing

Improvements are very welcome: bug fixes, new operators, new research directions, docs, or whole new tools. See [CONTRIBUTING.md](CONTRIBUTING.md).

If you build these tools into your own apps, a link back helps others find the original.

## Safety

- Never commit credentials, session files (`*.pkl`), API keys or the ACE files. `.gitignore` blocks the usual names, but check `git status` before pushing.
- Don't post your alpha expressions or results publicly unless you mean to share them.
