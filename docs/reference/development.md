# Development

## Setting up

```bash
git clone https://github.com/torimcd/cesm-hawc
cd cesm-hawc
micromamba env create -f environment.yml
micromamba activate hawc_env
pip install -e ".[sim,dev,docs]"
```

## Running the tests

```bash
pytest
```

Tests that need `sasktran2` or `hawcsimulator` are skipped automatically when
those aren't installed, so the base-tier tests run in a minimal environment.
Fixture-based tests use the small example columns bundled in
`src/cesm_hawc/data/` and need no external data.

## Building the documentation

The documentation is a [Jupyter Book](https://jupyterbook.org) project in
`docs/`, configured by `docs/myst.yml`. Jupyter Book 2 needs Node.js 18 or
later.

```bash
cd docs
jupyter book start          # live preview in the browser
jupyter book build --html   # static site in docs/_build/html
```

To add a page, create a Markdown (`.md`) or notebook (`.ipynb`) file under
`docs/` and add it to the `toc` in `myst.yml`.

## Repository layout

| Path | Contents |
|------|----------|
| `src/cesm_hawc/` | The package |
| `tests/` | Unit tests |
| `examples/` | Worked examples |
| `scripts/` | Environment setup, SLURM templates and research scripts |
| `docs/` | This documentation |
| `config.example.toml` | Template configuration |

## Contributing

Bug reports and pull requests are welcome on
[GitHub](https://github.com/torimcd/cesm-hawc/issues).
