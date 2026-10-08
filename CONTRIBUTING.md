# Contributing to launcheon

Thanks for your interest in contributing!

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/kyutai-labs/launcheon.git
cd launcheon
uv sync --group dev
```

## Before opening a PR

Run the full check suite (formatting, linting, typing, tests), which mirrors CI:

```bash
bash verify.sh
```

- Formatting/linting: `ruff` (line length 100, import sorting enabled)
- Typing: `pyright`
- Tests: `pytest tests/`

## Guidelines

- Keep PRs focused: one bug fix or one feature per PR.
- Add or update tests when changing behavior.
- New CLI commands must be documented in `docs/doc_cli.md`.

## Reporting issues

Please include: your launcheon version, a minimal grid file that reproduces the
problem, the exact command you ran, and the full output (run with `NO_COLOR=1`
for readable logs).
