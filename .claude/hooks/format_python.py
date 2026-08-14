"""PostToolUse hook: format and auto-fix a Python file after Claude edits it.

Wired up in `.claude/settings.json` on the Write|Edit matcher. Claude Code pipes
the tool-call payload to this script on stdin as JSON; we pull the edited path out
of it and run ruff.

Why a script instead of the usual one-liner: the documented pattern shells out to
`jq`, which is not installed on this machine. This project already guarantees a
Python venv, so parsing the payload in Python is both more reliable and far easier
to read than a chain of `sed` calls trying to unescape Windows paths.

**This hook never blocks an edit.** It exits 0 unconditionally. A formatter that
can fail a write is a formatter that will eventually stop you from fixing a bug,
so ruff's exit status is deliberately discarded.

Run it by hand the same way Claude Code does:

    echo '{"tool_input":{"file_path":"model/config.py"}}' \
        | ./.venv/Scripts/python.exe .claude/hooks/format_python.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# .claude/hooks/format_python.py -> project root is two levels up. Derived from
# __file__ rather than the cwd, so the hook works no matter where it is invoked.
ROOT = Path(__file__).resolve().parent.parent.parent
RUFF = ROOT / ".venv" / "Scripts" / "ruff.exe"
if not RUFF.exists():                      # non-Windows fallback
    RUFF = ROOT / ".venv" / "bin" / "ruff"


def edited_path(payload: dict) -> Path | None:
    """Pull the edited file path out of the hook payload.

    `tool_response.filePath` is the authoritative post-write path; `tool_input`
    is the fallback for tools that don't echo it back.
    """
    raw = (
        (payload.get("tool_response") or {}).get("filePath")
        or (payload.get("tool_input") or {}).get("file_path")
    )
    return Path(raw) if raw else None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0                            # nothing parseable; stay out of the way

    path = edited_path(payload)
    if path is None:
        return 0

    # .py only, on purpose. ruff can format notebooks too, but rewriting
    # model/eda.ipynb would churn its committed cell source and executed
    # outputs on every unrelated edit.
    if path.suffix != ".py" or not path.exists():
        return 0

    if not RUFF.exists():
        print(f"[hook] ruff not found at {RUFF}; skipping", file=sys.stderr)
        return 0

    for args in (["format"], ["check", "--fix"]):
        subprocess.run(
            [str(RUFF), *args, "--quiet", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            # check=False on purpose: `ruff check` exits non-zero when it finds
            # issues it cannot auto-fix, which is normal and must not surface as
            # a hook failure.
            check=False,
        )

    print(f"[hook] ruff formatted {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
