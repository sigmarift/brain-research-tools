"""
session_manager.py  -  reuse a saved BRAIN session instead of logging in.

    import session_manager as sm
    s = sm.get_session()          # loads ~/secrets/session.pkl if still valid
    sm.install()                  # make ace_lib reuse/save it on every relogin

Flow of get_session():
  1. Load ~/secrets/session.pkl (accepts a pickled requests.Session, a
     cookie jar, a dict of cookies, or {"cookies": ..., "headers": ...}).
  2. Put its cookies into ace_lib's SingleSession.
  3. Ask the API how long the token has left (GET /authentication).
     If more than `min_remaining` seconds remain -> use it, no login.
  4. Otherwise log in with ace_lib's normal flow (credentials from
     ~/secrets/platform-brain.json, biometrics prompt if needed) and save
     the fresh session back to session.pkl.

The pickle written is a plain requests.Session (cookies + headers only, no
password), so other tools that pickle/unpickle requests sessions can share it.
A small session.meta.json next to it records when it expires, so an expired
file is skipped without a network call.

Only load session.pkl files you created: unpickling runs code from the file.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import requests

from ace_bridge import ace

SECRETS_DIR = Path(os.path.expanduser("~")) / "secrets"
SESSION_PKL = SECRETS_DIR / "session.pkl"
SESSION_META = SECRETS_DIR / "session.meta.json"

# ace_lib re-logs in below 2000 s; stay above that so we do not hand back a
# session ace_lib will immediately throw away.
DEFAULT_MIN_REMAINING = 2400

_original_start_session = None


def _orig_start():
    global _original_start_session
    if _original_start_session is None:
        _original_start_session = ace.start_session
    return _original_start_session


def _log(msg: str) -> None:
    ace.logger.info(f"[session_manager] {msg}")


def _extract(obj) -> tuple[Optional[requests.cookies.RequestsCookieJar], dict]:
    """Pull cookies/headers out of whatever was pickled."""
    if isinstance(obj, requests.Session):
        return obj.cookies, dict(obj.headers)
    if isinstance(obj, requests.cookies.RequestsCookieJar):
        return obj, {}
    if isinstance(obj, dict):
        if "cookies" in obj:
            c = obj["cookies"]
            jar = c if isinstance(c, requests.cookies.RequestsCookieJar) \
                else requests.utils.cookiejar_from_dict(dict(c))
            return jar, dict(obj.get("headers") or {})
        if obj and all(isinstance(v, str) for v in obj.values()):
            return requests.utils.cookiejar_from_dict(obj), {}
    return None, {}


def _read_meta() -> dict:
    try:
        return json.loads(SESSION_META.read_text())
    except Exception:  # noqa: BLE001
        return {}


def load_saved_session(path: Path = SESSION_PKL) -> Optional[ace.SingleSession]:
    """SingleSession populated from session.pkl, or None if missing/unreadable."""
    if not path.exists():
        _log(f"no saved session at {path}")
        return None
    meta = _read_meta()
    if meta.get("expires_at") and meta["expires_at"] < time.time():
        _log("saved session already expired (per session.meta.json)")
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:  # noqa: BLE001
        _log(f"could not unpickle {path.name}: {type(e).__name__}: {e}")
        return None
    if isinstance(obj, ace.SingleSession):
        return obj
    jar, headers = _extract(obj)
    if jar is None:
        _log(f"{path.name} holds a {type(obj).__name__}; no cookies found")
        return None
    s = ace.SingleSession()
    s.cookies.clear()          # singleton may still hold a stale token
    s.cookies.update(jar)
    for k, v in headers.items():
        if k.lower() not in ("content-length", "authorization"):
            s.headers[k] = v
    return s


def save_session(s: requests.Session, remaining: Optional[int] = None,
                 path: Path = SESSION_PKL) -> None:
    """Write a plain requests.Session (cookies + headers) and expiry metadata."""
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(SECRETS_DIR, 0o700)
    except OSError:
        pass
    plain = requests.Session()
    plain.cookies.update(s.cookies)
    plain.headers.update({k: v for k, v in s.headers.items() if k.lower() != "authorization"})
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(plain, f)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    if remaining is None:
        remaining = ace.check_session_timeout(s)
    meta = {"saved_at": time.time(), "remaining_at_save": int(remaining or 0)}
    if remaining:               # 0 can mean "check failed", so do not mark expired
        meta["expires_at"] = time.time() + int(remaining)
    SESSION_META.write_text(json.dumps(meta))
    _log(f"saved session ({int(remaining or 0)} s left) to {path}")


def session_time_left(s: Optional[requests.Session] = None) -> int:
    """Seconds left on the token (0 if expired / no session)."""
    s = s or load_saved_session()
    return 0 if s is None else int(ace.check_session_timeout(s) or 0)


def get_session(min_remaining: int = DEFAULT_MIN_REMAINING,
                force_login: bool = False) -> ace.SingleSession:
    """Reuse ~/secrets/session.pkl if it has > min_remaining seconds, else log in."""
    if not force_login:
        s = load_saved_session()
        if s is not None:
            left = int(ace.check_session_timeout(s) or 0)
            if left > min_remaining:
                _log(f"reusing saved session: {left // 60} min left, no login needed")
                return s
            _log(f"saved session has {left} s left (< {min_remaining}); logging in")
    ace.SingleSession().cookies.clear()   # never mix a stale token with the new one
    s = _orig_start()()
    save_session(s)
    return s


def _start_session_patched() -> ace.SingleSession:
    # ace_lib calls start_session() when < 2000 s remain; at that point the
    # saved file is the same near-expired token, so force a real login.
    return get_session(force_login=session_time_left(ace.SingleSession()) <= DEFAULT_MIN_REMAINING)


def install() -> None:
    """Route ace_lib.start_session (used by check_session_and_relogin inside every
    simulate_* call) through this manager, so relogins also refresh session.pkl."""
    _orig_start()
    ace.start_session = _start_session_patched
    _log("ace_lib.start_session now reuses and refreshes ~/secrets/session.pkl")


def uninstall() -> None:
    ace.start_session = _orig_start()
