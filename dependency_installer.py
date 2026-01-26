"""Utility for installing project dependencies before GUI imports."""

from __future__ import annotations

import configparser
import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError:  # pragma: no cover - packaging should be available in most envs
    Requirement = None  # type: ignore

    def canonicalize_name(name: str) -> str:  # type: ignore
        return name.lower().replace("-", "_")


def _ensure_pyvis_installed() -> None:
    """Install ``pyvis`` using pip when it is not already available."""

    try:
        importlib.import_module("pyvis")
        print("[Dependency Check] PyVis is already installed.")
        return
    except ImportError:
        print("[Dependency Check] PyVis not found. Attempting to install via pip...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvis"])
        print("[Dependency Check] PyVis installed successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "[Dependency Check] Failed to install PyVis automatically. "
            "Please install it manually using 'pip install pyvis'."
        )


def _install_from_requirements(requirements_path: str) -> None:
    """Install dependencies listed in ``requirements_path`` using pip.

    Parameters
    ----------
    requirements_path: str
        Path to the requirements file.
    """
    if not os.path.exists(requirements_path):
        print("[Dependency Check] requirements.txt not found. Skipping installation.")
        return

    print("[Dependency Check] Checking and installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("[Dependency Check] All dependencies are satisfied.")
    except subprocess.CalledProcessError:
        print(
            "[Dependency Check] Failed to install dependencies. Please install them manually using 'pip install -r requirements.txt'"
        )
    except FileNotFoundError:
        print("[Dependency Check] 'pip' command not found. Please ensure pip is installed and in your PATH.")


def _auto_install_enabled(routes_path: str) -> bool:
    """Return ``True`` if auto-installation is enabled in routes INI file."""
    config = configparser.ConfigParser()
    try:
        config.read(routes_path)
        return config.getboolean("options", "auto_install_deps")
    except (configparser.Error, ValueError):
        return False


def maybe_install_deps(routes_path: str | None = None) -> None:
    """Install dependencies if enabled in ``routes_path``.

    Parameters
    ----------
    routes_path: str | None
        Path to the routes INI file. If ``None``, ``predefined_routes.ini`` in the
        current directory is used.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if routes_path is None:
        routes_path = os.path.join(base_dir, "predefined_routes.ini")

    if _auto_install_enabled(routes_path):
        _ensure_pyvis_installed()
        requirements = os.path.join(base_dir, "requirements.txt")
        _install_from_requirements(requirements)
    else:
        print("[Dependency Check] Auto installation disabled via routes configuration.")


def _load_requirements(requirements_path: str) -> list[str]:
    """Return normalized requirement lines from ``requirements_path``."""
    requirements: list[str] = []
    try:
        with open(requirements_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if " #" in line:
                    line = line.split(" #", 1)[0].strip()
                requirements.append(line)
    except OSError as exc:
        print(f"[Dependency Check] Failed to read requirements: {exc}")
    return requirements


def _requirements_hash(requirements: list[str]) -> str:
    """Compute a stable hash for the requirements list."""
    joined = "\n".join(requirements).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _parse_requirement_name(requirement_line: str) -> tuple[str | None, object | None]:
    """Parse a requirement line into a name and specifier (if available)."""
    if Requirement is None:
        name = requirement_line.split(";")[0].split("[")[0]
        for token in ["==", ">=", "<=", "~=", ">", "<", "!="]:
            if token in name:
                name = name.split(token)[0]
        return name.strip() or None, None
    try:
        req = Requirement(requirement_line)
        return req.name, req.specifier
    except Exception:
        return None, None


def _resolve_installed(requirements: list[str]) -> tuple[list[str], dict[str, str]]:
    """Return missing requirement lines and installed versions for known packages."""
    missing: list[str] = []
    installed: dict[str, str] = {}
    for requirement_line in requirements:
        name, specifier = _parse_requirement_name(requirement_line)
        if not name:
            continue
        normalized = canonicalize_name(name)
        try:
            version = metadata.version(name)
            installed[normalized] = version
            if specifier is not None and hasattr(specifier, "contains"):
                if not specifier.contains(version, prereleases=True):
                    missing.append(requirement_line)
        except metadata.PackageNotFoundError:
            missing.append(requirement_line)
    return missing, installed


def _load_state(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def ensure_requirements_installed(
    requirements_path: str | None = None,
    cache_path: str | None = None,
    *,
    force: bool = False,
) -> bool:
    """Ensure dependencies listed in requirements are installed.

    Uses a cache file to skip repeated checks when requirements are unchanged.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if requirements_path is None:
        requirements_path = os.path.join(base_dir, "requirements.txt")
    if cache_path is None:
        cache_path = os.path.join(base_dir, ".cache", "deps_state.json")

    if not os.path.exists(requirements_path):
        print("[Dependency Check] requirements.txt not found; skipping dependency check.")
        return True

    requirements = _load_requirements(requirements_path)
    req_hash = _requirements_hash(requirements)
    cache = _load_state(cache_path)
    refresh_requested = os.getenv("MACS_DEPS_REFRESH") in {"1", "true", "True"}
    if (
        not force
        and not refresh_requested
        and cache.get("requirements_hash") == req_hash
        and cache.get("python_executable") == sys.executable
        and cache.get("status") == "ok"
    ):
        print("[Dependency Check] Cached dependency state is valid; skipping check.")
        return True

    missing, installed = _resolve_installed(requirements)
    if missing:
        print(f"[Dependency Check] Missing {len(missing)} dependencies; installing...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", requirements_path]
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[Dependency Check] Installation failed: {exc}")
            _save_state(
                cache_path,
                {
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "python_executable": sys.executable,
                    "requirements_hash": req_hash,
                    "requirements": requirements,
                    "installed": installed,
                    "missing": missing,
                    "status": "error",
                },
            )
            return False

    missing_after, installed_after = _resolve_installed(requirements)
    status = "ok" if not missing_after else "error"
    _save_state(
        cache_path,
        {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "requirements_hash": req_hash,
            "requirements": requirements,
            "installed": installed_after,
            "missing": missing_after,
            "status": status,
        },
    )
    if missing_after:
        print(
            "[Dependency Check] Dependencies still missing after install: "
            + ", ".join(missing_after)
        )
    else:
        print("[Dependency Check] Dependencies are satisfied.")
    return not missing_after
