---
name: adapt-d2l-section
description: Adapt a section of Dive into Deep Learning to a divelet-ml notebook, or fix one already adapted. Use when adding a chapter or section to content/, converting the book's markdown, or reviewing an adapted notebook for leftovers and rendering problems.
---

# Adapting a section of the book

The book's sources are pinned at commit `23d7a5aecceee57d1292c56e90cce307f183bb0a` of `d2l-ai/d2l-en`. A section is `chapter_<name>/<file>.md`; the chapter order is in the root `index.md`, and the bibliography is `d2l.bib`. The notebooks of Chapters 2 to 9 and 11.3 in `content/` are the reference for every convention below: read one before writing a new one.

## Convert the markup (every item below has been missed at least once)

- **Cell type**: prose cells are `markdown`. A cell typed `md` loads, passes `nbformat.read`, and JupyterLab shows it as raw text.
- **Highlight markers**: `[**text**]` and `(**text**)` become `text` (also in headings: `## (**VGG Blocks**)` is `## VGG Blocks`). Keep bold that is inside the marker. Spans can run over several lines.
- **Tabs**: keep the PyTorch tab only. A cell names its tabs on its first line as `%%tab pytorch, mxnet` (most sections) or `#@tab pytorch, mxnet` (all of Chapter 12); `all` means every tab, and a cell with no marker is shared. Code behind a commented tab line (`# %%tab jax`) goes too. `:begin_tab:` blocks keep their text only when it names `pytorch`; the `[Discussions](...)` link of the PyTorch tab becomes the "Discussions of the original section" cell.
- **References**: `:numref:` becomes `[Section 5.4](url)`, `[Fig. 7.2.1](url)` or `[Chapter 5](url)`, with the numbers the book shows (the labels are in the section files: sections by their position in the chapter's `index.md`, figures counted per section, chapters by position in the root `index.md`); `:eqref:` becomes `[(4.1.8)](url)`. Every display equation `$$...$$` of a section is numbered, labelled or not: add `\tag{9.2.1}` before the closing `$$`.
- **Citations**: `:cite:` is `(Author et al., 2014)` and `:citet:` is `Author et al. (2014)`, from `d2l.bib` (one author: the name; two: `A and B`; three or more: `A et al.`; `key*1` is `key`; field names can be capitalized). Look for raw `:cite`/`:citet` afterwards.
- **Figures**: embed the book's own images as data URIs. A figure the book reproduces from a paper or a photo agency ("courtesy of", Getty) is not embedded: write `*Figure not reproduced here (third-party image): <caption> See [the book's figure](https://d2l.ai/_images/<file>).*`
- Drop `:label:`, `:width:`, `:height:`, `:eqlabel:` lines and `%matplotlib inline`.

## Adapt the code

- Imports: `import autograd.numpy as np`, `from divelet_ml import nn`, `import divelet_ml as d2l`. Every `d2l.name` must exist in `divelet_ml.__all__`: check, and add what the book saves from earlier sections to the library (with a test) when a section needs it.
- NumPy draws `float64`: cast parameters to `float32`. Drop `nn.Parameter`; arrays held as attributes are parameters, except the names a module lists in `buffers`. In-place updates (`Y += X`) become `Y = Y + X`; `torch.cat` is `np.concatenate`.
- `nn.LazyConv2d` takes `padding` before `stride`, unlike PyTorch: pass both by keyword.
- Drop `num_gpus` and `try_gpu`. Keep the book's `#@save` comments.
- Training that is too slow for a browser: `d2l.browser_epochs` and `d2l.browser_examples`, a tablet-marked note saying so, an entry in `RUN_MS_NOTEBOOKS` of `tests/smoke_test.py`, the notebook in `TARGETS` of `tools/full_run_figures.py`, and `uv run --group test python tools/full_run_figures.py <notebook>`. A model that cannot train at all gets the "not run" note of 8.1 (cost from `spikes/js-conv/count_macs.py`) and the book's cell as a code block.
- Paragraphs the book does not have carry the tablet image and the cell tag `divelet-ml`.

## Fit the notebook into the project

Header credit cell, previous/next cell (top and bottom; link the neighbouring chapters, with `../`), the section's body, the forum cell, "Attribution and changes" listing every change. Then README rows, ROADMAP ticks and status, and the nav cells of the neighbours.

## Check before the pull request

1. `uv run --group test pytest tests/test_notebooks.py tests/test_full_run_figures.py` runs the notebooks, validates the notebook schema, and fails on markers, directives and tab leftovers.
2. `./build.sh` (minutes; wait for its own exit line, not for the launcher's notification), then `uv run --group browser pytest tests/smoke_test.py -k "chrome and 08."` and `"ipad and 08."` with the chapter's number. The smoke test runs code cells only: it passed with prose shown as raw text and with stray markers.
3. **Look at the rendered notebook.** Screenshot, from the built site with Playwright, the top, a figure, a tablet-marked cell and the bottom. The notebook scrolls inside `.jp-WindowedPanel-outer`, and cells outside the window are not in the DOM until scrolled to.
4. Commit messages and pull request descriptions carry no attribution lines. The description lists what was checked and what was not.
