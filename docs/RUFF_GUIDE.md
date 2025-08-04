# Ruff Configuration Guide

This project uses [Ruff](https://docs.astral.sh/ruff/) as the primary tool for code formatting and linting, replacing black, isort, flake8, and pylint.

## Why Ruff?

- **Fast**: 10-100x faster than traditional Python linters
- **All-in-one**: Replaces multiple tools (black, isort, flake8, pylint)
- **Compatible**: Drop-in replacement with familiar rule sets
- **Modern**: Built with Rust for performance

## Common Commands

### Linting

```bash
# Check for linting issues
ruff check flows/ tests/

# Check and automatically fix issues
ruff check --fix flows/ tests/

# Check specific file
ruff check flows/data_ingestion.py

# Show all available rules
ruff linter
```

### Formatting

```bash
# Format code (replaces black)
ruff format flows/ tests/

# Check formatting without making changes
ruff format --check flows/ tests/

# Format specific file
ruff format flows/data_ingestion.py
```

### Combined Workflow

```bash
# Recommended workflow: lint and format
ruff check --fix flows/ tests/ && ruff format flows/ tests/
```

## Configuration

All Ruff configuration is in `pyproject.toml` under `[tool.ruff]` sections:

### Main Configuration

- **Line length**: 99 characters
- **Target Python version**: 3.10+
- **Rule sets**: Comprehensive set including pycodestyle, Pyflakes, isort, bugbear, etc.

### Enabled Rule Sets

- `E`, `W`: pycodestyle errors and warnings
- `F`: Pyflakes (undefined names, unused imports)
- `I`: isort (import sorting)
- `B`: flake8-bugbear (common bugs)
- `C4`: flake8-comprehensions (better comprehensions)
- `UP`: pyupgrade (modern Python syntax)
- `ARG`: flake8-unused-arguments
- `SIM`: flake8-simplify (code simplification)
- `TCH`: flake8-type-checking (type checking imports)
- `PTH`: flake8-use-pathlib (prefer pathlib)
- `ERA`: eradicate (remove commented-out code)
- `PL`: Pylint rules
- `RUF`: Ruff-specific rules

### Ignored Rules

- `E501`: Line too long (handled by line-length setting)
- `B008`: Function calls in argument defaults (common in frameworks)
- `PLR0913`: Too many arguments (relaxed for ML code)
- `PLR0915`: Too many statements (relaxed for ML code)
- `PLR2004`: Magic values (common in ML/data science)

### Per-File Ignores

- **Test files**: Relaxed rules for fixtures and test patterns
- ****init**.py**: Allow unused imports (common for package exports)

## IDE Integration

### VS Code

Add to your `.vscode/settings.json`:

```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.fixAll.ruff": true
    }
}
```

### PyCharm

1. Install the Ruff plugin
2. Configure in Settings → Tools → Ruff
3. Enable "Run Ruff on save"

## Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```

## Migration from Other Tools

### From Black

- Ruff format is nearly identical to Black
- Same line length and formatting style
- No configuration changes needed

### From isort

- Import sorting is handled by Ruff's `I` rules
- Configuration in `[tool.ruff.isort]` section
- Same known-first-party and sorting behavior

### From flake8

- Most flake8 rules are covered by Ruff's `E`, `W`, `F` rules
- Plugin functionality covered by additional rule sets
- Ignore patterns work the same way

### From pylint

- Core pylint rules available in Ruff's `PL` rule set
- Performance and complexity rules included
- More focused on common issues

## Performance Comparison

| Tool | Time (large codebase) | Ruff Equivalent |
|------|----------------------|-----------------|
| black | ~2.5s | ruff format (~0.1s) |
| isort | ~1.8s | ruff check I* (~0.05s) |
| flake8 | ~8.2s | ruff check E,W,F (~0.1s) |
| pylint | ~45s | ruff check PL* (~0.2s) |
| **Total** | **~57.5s** | **~0.45s** |

*Times are approximate and may vary based on codebase size and complexity.*

## Troubleshooting

### Common Issues

1. **Rule conflicts**: Use `# noqa: RULE_CODE` to ignore specific lines

   ```python
   long_variable_name = some_function_with_many_args(arg1, arg2, arg3, arg4)  # noqa: E501
   ```

2. **Import sorting conflicts**: Ensure `known-first-party` is correctly configured

   ```toml
   [tool.ruff.isort]
   known-first-party = ["air_quality_ml_project", "flows"]
   ```

3. **False positives in tests**: Use per-file ignores for test-specific patterns

   ```toml
   [tool.ruff.per-file-ignores]
   "tests/*" = ["ARG001", "PLR2004"]
   ```

### Getting Help

```bash
# Show help for specific rule
ruff rule E501

# List all available rules
ruff linter

# Show configuration
ruff config

# Check Ruff version
ruff --version
```

## Best Practices

1. **Run before committing**: Always run `ruff check --fix && ruff format`
2. **Use in CI/CD**: Add Ruff checks to your pipeline
3. **Configure IDE**: Set up automatic formatting and linting
4. **Gradual adoption**: Start with basic rules, add more over time
5. **Team consistency**: Share configuration via `pyproject.toml`
