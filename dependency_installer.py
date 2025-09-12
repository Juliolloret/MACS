"""Utility for installing project dependencies before GUI imports."""

from __future__ import annotations

import configparser
import os
import shutil
import subprocess
import sys


def _check_and_install_graphviz():
    """Check for Graphviz and install it if it is not found."""
    if shutil.which("dot"):
        print("[Dependency Check] Graphviz is already installed.")
        return

    print("[Dependency Check] Graphviz not found. Attempting to install...")
    platform = sys.platform
    if platform.startswith("linux"):
        # Linux
        try:
            print("[Dependency Check] Updating package list...")
            subprocess.run(["sudo", "apt-get", "update"], check=True, capture_output=True)
            print("[Dependency Check] Installing graphviz...")
            subprocess.run(
                ["sudo", "apt-get", "install", "-y", "graphviz"],
                check=True,
                capture_output=True,
            )
            print("[Dependency Check] Graphviz installed successfully using apt-get.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(
                "[Dependency Check] Failed to install Graphviz using apt-get. "
                "Please install it manually. Error: %s",
                e,
            )
    elif platform == "darwin":
        # macOS
        if not shutil.which("brew"):
            print(
                "[Dependency Check] Homebrew not found. Please install Homebrew first, "
                "then run 'brew install graphviz'."
            )
            return
        try:
            subprocess.run(["brew", "install", "graphviz"], check=True)
            print("[Dependency Check] Graphviz installed successfully using Homebrew.")
        except subprocess.CalledProcessError as e:
            print(
                "[Dependency Check] Failed to install Graphviz using Homebrew. "
                "Please install it manually. Error: %s",
                e,
            )
    elif platform == "win32":
        # Windows
        if not shutil.which("choco"):
            print(
                "[Dependency Check] Chocolatey not found. Please install Chocolatey first, "
                "then run 'choco install graphviz'."
            )
            return
        try:
            subprocess.run(["choco", "install", "graphviz", "-y"], check=True)
            print("[Dependency Check] Graphviz installed successfully using Chocolatey.")
        except subprocess.CalledProcessError as e:
            print(
                "[Dependency Check] Failed to install Graphviz using Chocolatey. "
                "Please install it manually. Error: %s",
                e,
            )
    else:
        print(f"[Dependency Check] Unsupported platform: {platform}. Please install Graphviz manually.")


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
        _check_and_install_graphviz()
        requirements = os.path.join(base_dir, "requirements.txt")
        _install_from_requirements(requirements)
    else:
        print("[Dependency Check] Auto installation disabled via routes configuration.")
