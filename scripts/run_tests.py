#!/usr/bin/env python3
"""Run the MACS test suite with pytest when available."""

import importlib.util
import os
import subprocess
import sys
from datetime import datetime, timezone


def _ensure_test_env() -> None:
    """Set up environment defaults to keep tests deterministic."""
    os.environ.setdefault("OPENAI_API_KEY", "dummy")


def _has_pytest() -> bool:
    """Return True when pytest is available in the current environment."""
    return importlib.util.find_spec("pytest") is not None


def _ensure_pytest() -> bool:
    """Install pytest if missing. Returns True when available."""
    if _has_pytest():
        return True
    print("[test-runner] pytest not found; installing...")
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "pytest"],
        check=False,
    )
    if completed.returncode == 0 and _has_pytest():
        return True
    print("[test-runner] pytest install failed; retrying with --user.")
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--user", "pytest"],
        check=False,
    )
    return completed.returncode == 0 and _has_pytest()


def main() -> int:
    _ensure_test_env()
    if not _ensure_pytest():
        print("[test-runner] ERROR: pytest is required but could not be installed.")
        return 1

    cmd = [sys.executable, "-m", "pytest"]
    print("[test-runner] Running:", " ".join(cmd))

    output_path = os.path.join(os.getcwd(), "test_results.txt")
    start_time = datetime.now(timezone.utc).isoformat()
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"[test-runner] started_utc={start_time}\n")
        handle.write(f"[test-runner] command={' '.join(cmd)}\n\n")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
        handle.write(f"\n[test-runner] exit_code={return_code}\n")
        end_time = datetime.now(timezone.utc).isoformat()
        handle.write(f"[test-runner] finished_utc={end_time}\n")

    print(f"[test-runner] Output saved to: {output_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
