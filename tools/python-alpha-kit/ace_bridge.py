"""
ace_bridge.py  -  finds WorldQuant's ACE library, which this repo does not ship.

The ACE files (ace_lib.py and helpful_functions.py) come from the BRAIN
platform and are not ours to redistribute. Put them in ONE of these places:

  1. the repo's shared  ace/  folder        (recommended, used by every tool)
  2. this tool's folder (next to this file)
  3. any folder named in the ACE_LIB_DIR environment variable

The kit imports them only when it needs the API, so offline work
(building and dry-running alphas) runs without them.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED = ("ace_lib.py", "helpful_functions.py")

HELP = """
The ACE library files were not found: {missing}

This repo does not include them (they are WorldQuant's files). To add them:
  1. Get ace_lib.py and helpful_functions.py from the ACE API package on the BRAIN platform.
  2. Copy both into the repo's  ace/  folder  ({ace_dir}).
     (or next to this tool, or into a folder set in the ACE_LIB_DIR environment variable)
  3. Restart the notebook kernel.

Offline mode (offline=True) works without them.
Searched: {searched}
"""


def candidate_dirs() -> list[Path]:
    dirs = []
    if os.environ.get("ACE_LIB_DIR"):
        dirs.append(Path(os.environ["ACE_LIB_DIR"]).expanduser())
    dirs.append(HERE)
    for parent in list(HERE.parents)[:3]:
        dirs.append(parent / "ace")
    dirs.append(Path.cwd())
    dirs.append(Path.cwd() / "ace")
    return list(dict.fromkeys(d.resolve() for d in dirs))


def find_ace_dir() -> Path | None:
    for d in candidate_dirs():
        if all((d / f).exists() for f in REQUIRED):
            return d
    return None


def available() -> bool:
    return "ace_lib" in sys.modules or find_ace_dir() is not None


def load():
    if "ace_lib" in sys.modules:
        return sys.modules["ace_lib"]
    d = find_ace_dir()
    if d is None:
        found_any = {f for c in candidate_dirs() for f in REQUIRED if (c / f).exists()}
        missing = ", ".join(f for f in REQUIRED if f not in found_any) or ", ".join(REQUIRED)
        roots = [p / "ace" for p in list(HERE.parents)[:3] if (p / "tools").exists()]
        ace_dir = roots[0] if roots else HERE
        raise ImportError(HELP.format(missing=missing, ace_dir=ace_dir,
                                      searched=", ".join(str(c) for c in candidate_dirs())))
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    return importlib.import_module("ace_lib")


class _LazyACE:
    """Behaves like the ace_lib module, imported on first use."""

    def __getattr__(self, name):
        return getattr(load(), name)

    def __setattr__(self, name, value):
        setattr(load(), name, value)


ace = _LazyACE()
