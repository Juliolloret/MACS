"""Utility for installing project dependencies before GUI imports."""

from __future__ import annotations

import configparser
import os
import subprocess
import sys


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
        requirements = os.path.join(base_dir, "requirements.txt")
        _install_from_requirements(requirements)
    else:
        print("[Dependency Check] Auto installation disabled via routes configuration.")
