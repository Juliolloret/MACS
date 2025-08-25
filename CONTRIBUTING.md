# Contributing to MACS

Thank you for your interest in contributing to Multi Agent Chem Search (MACS)! Your contributions help improve the project for everyone.

## How to Contribute

1. **Fork the Repository**
   - Click the “Fork” button on the top right to create your own copy.

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/your-username/MACS.git
   cd MACS
   ```

3. **Create a Branch**

   Create a new branch for your feature or fix:
   ```bash
   git checkout -b my-feature
   ```

4. **Make Changes**

   - Write clear, well-documented Python code.
   - Add or update tests if necessary.

5. **Test Your Changes**

   - Make sure all tests pass before submitting a pull request.

6. **Commit and Push**
   ```bash
   git add .
   git commit -m "Describe your changes"
   git push origin my-feature
   ```

7. **Open a Pull Request**

   - Go to your fork on GitHub and open a pull request to the `main` branch.
   - Provide a clear description of your changes and reference any relevant issues.

## Code Style

- Follow PEP8 guidelines for Python code.
- Include docstrings for functions and classes.
- Use meaningful commit messages.

## Reporting Issues

- Use the Issues tab to report bugs or suggest features.
- Include as much detail as possible (Python version, OS, steps to reproduce, etc.).

## Community

- Be respectful and considerate in all interactions.
- Review the [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## Developer Notes

### Test Configuration Discrepancy
Please be aware that the CLI test script (`cli_test.py`) uses its own configuration file, `config_cli_test_integrated.json`. The agent workflow defined in this test configuration may be out of sync with the primary `config.json` used by the GUI application.

Specifically, it may reference older or different agents (e.g., `MultiDocSynthesizerAgent`, `WebResearcherAgent`). When making changes to the core agent architecture, please consider updating the test configuration as well to maintain consistency.

**Thank you for helping make MACS better!**
