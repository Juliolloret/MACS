# Contributing to MACS

Thank you for your interest in contributing to Multi-Agent Collaboration System (MACS).
This guide is written for beginners and explains each step in plain language.

## 1. Prerequisites

- Python 3.10 or newer (3.8+ may work depending on dependencies).
- Git (recommended).

Check versions:

```bash
python --version
git --version
```

## 2. Fork and Clone (GitHub)

1. On GitHub, click the "Fork" button to create your own copy.
2. Clone your fork:

```bash
git clone https://github.com/your-username/MACS.git
cd MACS
```

## 3. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 5. Create a Branch

```bash
git checkout -b my-feature
```

Use a clear name like `fix-readme` or `add-new-agent`.

## 6. Make Changes

- Keep changes focused on one goal.
- Update documentation if behavior changes.
- Add tests for new functionality.

## 7. Run Tests

MACS provides a test runner that saves output to `test_results.txt`:

```bash
python scripts/run_tests.py
```

You can also run pytest directly:

```bash
pytest
```

## 8. Commit and Push

```bash
git add .
git commit -m "Short description of your change"
git push origin my-feature
```

## 9. Open a Pull Request

- Go to your fork on GitHub.
- Click "New Pull Request".
- Describe what you changed and why.

## 10. Code Style

- Follow PEP8 for Python.
- Keep functions small and readable.
- Add docstrings for public functions and classes.

## 11. Reporting Issues

- Use the GitHub Issues tab.
- Include steps to reproduce and full error logs.

## Developer Notes

### Test Configuration Discrepancy

`cli_test.py` uses `config_cli_test_integrated.json`. This workflow may drift
from `config.json`. If you change core agents, update both configs.

Thank you for helping make MACS better.
