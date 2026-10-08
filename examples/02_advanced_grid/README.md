# Advanced experiment grid

This example builds on [01_simple_grid](../01_simple_grid/README.md) and showcases more advanced sweep definitions with the Python API:

* `kwargs_nicknames`: short nicknames for long flag names and values (e.g. long paths)
* `kwargs_groups`: groups of flags whose values are tied together (e.g. an architecture name mapping to its width and depth)
* launcheon secret keys such as `{{launcheon.exp.port}}` or `{{launcheon.exp.log_dir}}`, resolved per experiment

Run it with any `ExperimentGrid` command, for instance:

```bash
uv run python examples/02_advanced_grid/python_api_usage.py print name
uv run python examples/02_advanced_grid/python_api_usage.py table
```
