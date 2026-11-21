# Contributing to divelet-ml

## Setup

You need [uv][uv]. Building the site also needs [micromamba][micromamba], and the browser smoke test needs Google Chrome plus Playwright's WebKit (`uv run --group browser playwright install webkit`, once).

```bash
git clone https://github.com/decitre/divelet-ml.git
cd divelet-ml
uv sync
```

`uv sync` creates `.venv` with the `divelet_ml` package (editable) and the `dev` group, which provides bumpver.

## Test

These are the same checks [CI][build-yml] runs on every push and pull request:

```bash
# divelet_ml package tests, the 11.3 notebook's attention properties, and every notebook in content/ run in CPython
# (-n auto runs them in parallel workers; CI runs it once per Python version, 3.11 and 3.13)
uv run --group test pytest tests/test_divelet_ml.py tests/test_attention_scoring.py tests/test_notebooks.py -v -n auto

# divelet_ml package tests with a coverage report (CI uploads it to Codecov)
uv run --group test pytest tests/test_divelet_ml.py --cov --cov-report=term-missing

# build the xeus-python and Pyodide sites into workspace/public/
./build.sh

# run every notebook in both sites in headless Chrome and in WebKit as an iPad (after build.sh)
uv run --group browser pytest tests/smoke_test.py -v
```

The CPython run is a fast first check. Only the smoke test shows that a notebook runs in JupyterLite.

In CI, a smoke job whose tests fail runs the failed ones once more, with a warning, because the Pyodide kernel in WebKit sometimes does not start on the runner. A test that fails twice fails the job, which blocks the deployment. When the kernel does not become ready, the test fails with the state of the page and a screenshot, which the job uploads as an artifact even if the second run passes.

To browse the built sites, serve `workspace/public/` over HTTP, for example with `python -m http.server -d workspace/public`, and open `/xeus/` or `/pyodide/`.

## Full-run figures

Notebooks that train less in the browser than the book (those that call `d2l.browser_epochs` or `d2l.browser_examples`) show, after each training cell, a figure of what the book's full training gives. `tools/full_run_figures.py` makes them: it runs each notebook in CPython, where those functions return the book's numbers, draws the model's progress board once, and writes it into a markdown cell tagged `full-run`, with the final values. The random start is seeded, so the same code gives the same figures.

```bash
uv run --group test python tools/full_run_figures.py [notebook ...]
```

Run it again after changing the library or a training cell of those notebooks, and commit what it changes. `tests/test_full_run_figures.py` checks that every such notebook has its figures. A new notebook that trains less in the browser goes into `TARGETS` in the script.

## Changes

Work on a branch and open a pull request against `main`. A pull request must pass the "Build site" workflow. A push to `main` deploys the site to GitHub Pages.

## Version bump

The version lives in `src/divelet_ml/__init__.py` (`__version__`). Hatch reads it from there, and `[tool.bumpver]` in `pyproject.toml` keeps `current_version` in sync. Don't edit either one by hand. From an up-to-date `main` with a clean working tree:

```bash
uv run bumpver update --patch   # 0.1.0 -> 0.1.1; or --minor, --major
uv run bumpver update --patch --dry   # preview the change first
```

bumpver commits `bump version X -> Y`, tags it `Y` (bare version, no `v`), and pushes both to `origin`.

## Release

Publishing a GitHub release uploads the package to PyPI through [release.yml][release-yml], using trusted publishing (no token). A version on PyPI can never be replaced, so check the build first:

1. **Bump the version** as above.
2. **Test the release build without publishing:**

   ```bash
   gh workflow run release.yml --ref main -f publish=false
   ```

   It builds the sdist and wheel, runs `twine check --strict`, and checks that the installed `__version__` matches the package version.
3. **Write the release notes** in `release-notes.md`, an untracked local file. Use `### Breaking`, `### Added` and `### Fixed` sections as needed, with no version heading (the release title shows it), and refer to issues as `(#N)`.
4. **Publish the release** on the tag bumpver created:

   ```bash
   gh release create X.Y.Z --title vX.Y.Z --notes-file release-notes.md
   ```

   This triggers the publish job. Check the run under Actions → Release, then the page at [pypi.org/project/divelet-ml][pypi].

A draft release doesn't trigger the workflow. A manual run with `publish=true` on `main` also publishes, but use a release so each PyPI version has a tag and notes.

[uv]: https://docs.astral.sh/uv/
[micromamba]: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
[build-yml]: .github/workflows/build.yml
[release-yml]: .github/workflows/release.yml
[pypi]: https://pypi.org/project/divelet-ml/
