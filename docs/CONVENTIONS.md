# Conventions

How the project is put together, so that a section or a package can be added the way the others were. [CONTRIBUTING.md](../CONTRIBUTING.md) has the commands.

## Dependencies and packages

- **One list.** The packages a notebook may import are `dependencies` in `pyproject.toml` (`numpy`, `matplotlib`, `pandas`, `autograd`), plus the `divelet_ml` package itself. `requires-python` is 3.11 or later; the Pyodide kernel
  (0.7.0) runs Python 3.13, and xeus-python 5 runs 3.13 too.
- **How the browser kernels get them.** `build.sh` reads that list with [jl-forge](https://github.com/decitre/jl-forge) (pinned, `jl-forge==0.8.0`): for xeus-python, `jl-forge xeus-env --pyproject` makes the conda environment and
  `jl-forge xeus-mount src/divelet_ml` mounts the package from source; for Pyodide, `jl-forge pyodide-wheels` and `pyodide-dist` collect the wheels and a pruned Pyodide, and one wheel of `divelet_ml` is built next to the notebooks.
  The two sites are built in separate environments, because JupyterLite copies every installed extension.
- **A new dependency** has to be installable in both kernels, and the smoke test then fails on any request it makes to another host. Add it to `pyproject.toml`, build, and run the notebook that needs it in both kernels.
- **Development tools** live in `[dependency-groups]` (`dev`, `test`, `browser`), locked in `uv.lock`; they are not in the browser.
- **Versions.** `bumpver` changes `pyproject.toml` and `src/divelet_ml/__init__.py` together. A release is: notes in `release-notes.md` on `main`, `uv run bumpver update --minor`, the release workflow as a dry run (`publish=false`), then `gh release create`, which publishes to PyPI.

## The library (`src/divelet_ml`)

- The code is adapted from the book's `d2l/torch.py` (MIT-0), with each module's docstring saying so; what the book saves with `#@save` and a later section uses is added, with a test, when a section needs it.
- Gradients come from autograd: models compute with `autograd.numpy`, parameters are `float32` arrays held as attributes, and optimizers take the gradients as an argument (`step(grads)`). [LIMITATIONS.md](LIMITATIONS.md) lists the differences from PyTorch.
- A change to a public signature needs a version decision, because the notebooks and the PyPI package depend on it (`nn.LazyConv2d`'s argument order is waiting for one).

## Notebooks (`content/`)

- A folder per chapter, `NN_chapter-name/`, and a file per section, `NN.MM_section-name.ipynb`; the section numbers are the book's.
- Each notebook: a credit cell, a navigation cell (top and bottom), the section, the forum cell, and an "Attribution and changes" cell that lists every change. Paragraphs and cells that the book does not have carry the tablet image and the cell tag `divelet-ml`.
- Training that is too slow for a browser uses `d2l.browser_epochs` or `d2l.browser_examples`, a tablet-marked note, a `RUN_MS_NOTEBOOKS` entry in `tests/smoke_test.py`, a target in `tools/full_run_figures.py` and the figures it writes.
- A notebook that takes more than 300 s per cell in CPython has a `CELL_TIMEOUTS` entry in `tests/test_notebooks.py`.
- The converter that drafts them is in [`tools/d2l_convert`](../tools/d2l_convert/README.md); the conventions and the checks for adapting a section are in the skill [`adapt-d2l-section`](../.claude/skills/adapt-d2l-section/SKILL.md).

## Data that a notebook reads

A browser cannot read other hosts, so `build.sh` downloads the file once (cached in `~/.cache/jl-forge`), checks its SHA-256, and copies it to `data/<name>/` on the site. A helper in `data.py` reads it from the site in JupyterLite
(`_read_url`, then the SHA-1 the book uses) and downloads and checks the book's copy elsewhere. The licence is checked before the file is hosted, and the credit goes in [NOTICE.md](../NOTICE.md) and in the notebooks that read it.

## Pull requests

A pull request lists what was checked and what was not. Commit messages and descriptions carry no attribution lines for tools. Each notebook is checked in CPython, in both kernels in Chrome, and looked at rendered; see the skill.
