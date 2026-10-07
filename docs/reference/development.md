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
Fixture-based tests use the small example column bundled in
`src/cesm_hawc/data/` and need no external data.

## Building the documentation

The documentation is a [Jupyter Book](https://jupyterbook.org) project in
`docs/`, configured by `docs/myst.yml`. Jupyter Book 2 needs Node.js 18 or
later.

```bash
python docs/generate_api.py   # write the API reference pages
cd docs
jupyter book start            # live preview in the browser
jupyter book build --html     # static site in docs/_build/html
```

To add a page, create a Markdown (`.md`) or notebook (`.ipynb`) file under
`docs/` and add it to the `toc` in `myst.yml`.

### API reference

The pages under [Python API](#api-cesm_hawc) are generated from the
docstrings in `src/cesm_hawc` by `docs/generate_api.py`. They are written to
`docs/reference/api/`, which is not committed, so run the script before
building. New modules, classes and public functions (names not starting
with `_`) appear automatically.

Docstrings follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)
format, with units in brackets, e.g. `Latitude [degrees]`. Check them with:

```bash
ruff check src/
```

Cross-references such as ``:func:`run_ali_simulation` `` and
``:meth:`~cesm_hawc.waccm.WACCMAtmosphere.get_column_profiles` `` become
links between API pages.

### Continuous integration

`.github/workflows/docs.yml` checks the docstrings, generates the API
reference and builds the book on every pull request and push. Pushes to
`main` also publish the site to GitHub Pages.

## Repository layout

| Path | Contents |
|------|----------|
| `src/cesm_hawc/` | The package |
| `tests/` | Unit tests |
| `scripts/` | HPC environment setup and the SLURM job script |
| `docs/` | This documentation |
| `config.example.toml` | Template configuration |

## Contributing

Bug reports and pull requests are welcome on
[GitHub](https://github.com/torimcd/cesm-hawc/issues).
