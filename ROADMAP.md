# Roadmap

divelet-ml aims to make [Dive into Deep Learning](https://d2l.ai/)
accessible through JupyterLite while preserving the educational structure,
mathematical explanations, learning objectives, and exercises of the original
textbook.

Development will proceed incrementally, starting with a compatibility
assessment and a small set of representative chapters.

**Current status:** Early development. Static compatibility analysis has been
performed. Chapters 2 (*Preliminaries*), 3 (*Linear Neural Networks for
Regression*), 4 (*Linear Neural Networks for Classification*) and 5
(*Multilayer Perceptrons*) are adapted in full, along with 11.3 *Attention
Scoring Functions*. The adapted notebooks run in JupyterLite with both the Pyodide and
xeus-python kernels, on the site published at <https://decitre.github.io/divelet-ml/>.
GitHub Actions tests, builds and deploys the site on every push to `main`.

## Phase 1 — Compatibility Assessment

**Status: In progress**

The first phase evaluates the feasibility of adapting the book for browser-native
Python execution.

The PyTorch edition serves as the primary implementation reference.

### Tasks

- [x] Establish an initial upstream revision of the book.
- [x] Develop a static analysis tool for the book's Markdown sources.
- [x] Identify framework-specific code variants.
- [x] Classify chapters by estimated adaptation difficulty.
- [x] Identify initial candidate chapters.
- [ ] Review remaining parser warnings and ambiguous code variants.
- [x] Verify upstream code licenses and third-party asset requirements
      (`LICENSE-SUMMARY`: text CC BY-SA 4.0, sample code MIT-0). Covers the
      material adapted so far; datasets and images that later chapters use
      are reviewed as they are adapted.
- [ ] Publish the compatibility assessment and methodology.

### Initial Assessment

The initial audit was performed against the book's commit:

`23d7a5aecceee57d1292c56e90cce307f183bb0a`

The latest static audit identified:

| Metric | Result |
|---|---:|
| Source Markdown files | 195 |
| Framework-specific and shared variant rows | 465 |
| Executable variant rows | 412 |
| Rows requiring further verification | 13 |
| Parser warnings | 14 |

These figures describe the source inventory, not the number of browser-compatible
chapters.

Static compatibility classifications are preliminary estimates. They do not
establish whether a chapter executes successfully in JupyterLite.

## Phase 2 — Initial Chapter Adaptations

**Status: Complete**

The first implementation phase focuses on three representative sections.

### 1. Array Operations (2.1)

**Upstream source:** `chapter_preliminaries/ndarray.md`

**Status:** Adapted as
[`content/02_preliminaries/02.01_ndarray.ipynb`](content/02_preliminaries/02.01_ndarray.ipynb).
Runs in CPython (`tests/test_notebooks.py`) and in JupyterLite with Pyodide
and xeus-python (`tests/smoke_test.py`). Its closing "Attribution and changes"
section lists the upstream revision, attribution, licenses and changes.

Objectives:

- Introduce multidimensional arrays and basic numerical operations.
- Replace framework-specific tensor operations with NumPy where appropriate.
- Preserve the original mathematical explanations and exercises.
- Validate execution in browser-native Python environments.

### 2. Synthetic Regression Data (3.3)

**Upstream source:** `chapter_linear-regression/synthetic-regression-data.md`

**Status:** Adapted as
[`content/03_linear-regression/03.03_synthetic-regression-data.ipynb`](content/03_linear-regression/03.03_synthetic-regression-data.ipynb),
validated like *Array Operations*. It introduces the `divelet_ml` package
([`src/divelet_ml`](src/divelet_ml/__init__.py)), divelet-ml's NumPy counterpart of the `d2l`
library: `DataModule`, `save_hyperparameters` and `add_to_class` from the book,
and `ArrayLoader` in place of a framework's data loader. As in the original,
the data are unseeded and differ between runs.

Objectives:

- Generate synthetic datasets using NumPy.
- Preserve the original statistical and mathematical concepts.
- Evaluate numerical reproducibility.
- Adapt random-number generation and data-handling operations as needed.

### 3. Attention Scoring Functions (11.3)

**Upstream source:**
`chapter_attention-mechanisms-and-transformers/attention-scoring-functions.md`

**Status:** Adapted as
[`content/11_attention-mechanisms-and-transformers/11.03_attention-scoring-functions.ipynb`](content/11_attention-mechanisms-and-transformers/11.03_attention-scoring-functions.ipynb),
validated like *Array Operations*, and the first section with plots.
`tests/test_attention_scoring.py` checks the notebook's functions against the
section's equations: masked weights are zero, the others sum to one, outputs are
weighted averages of the values, and identical keys give equal weights.

Findings:

- The whole section reproduces in NumPy: it computes forward passes only, with
  no training and no automatic differentiation.
- `divelet_ml.nn` stands in for the parts of `torch.nn` it uses (`Module`,
  `Dropout`, `LazyLinear`, `softmax`), so the book's classes keep their shape.
- `d2l.show_heatmaps` needs matplotlib, now a dependency of `divelet_ml`
  and loaded by both browser kernels.
- Training attention models, as later sections do, will need gradients, which
  NumPy does not provide.

Objectives:

- Preserve the mathematical formulation of attention scoring.
- Explore NumPy implementations of relevant operations.
- Evaluate the feasibility of reproducing the original examples without
  requiring a full deep learning framework.
- Identify performance and implementation limitations.

### Acceptance Criteria

Each adapted chapter must:

- [x] Preserve the original learning objectives.
- [x] Retain the relevant mathematical explanations and exercises.
- [x] Identify its upstream source and revision.
- [x] Document substantive modifications.
- [x] Use browser-compatible dependencies.
- [x] Execute successfully in every JupyterLite runtime for which
      compatibility is claimed.
- [x] Produce results consistent with the intended educational examples.
- [x] Include applicable licensing and attribution notices.

All three sections meet these criteria (checked 2026-11-16): each keeps its
source's headings, display equations and exercises. Every non-random output
of 2.1 and 3.3 matches the book's PyTorch output up to print precision, and
`tests/test_attention_scoring.py` checks 11.3's results against its equations.

## Phase 3 — JupyterLite Infrastructure

**Status: In progress**

Establish a reproducible browser-native learning environment.

### Tasks

- [x] Configure a JupyterLite distribution for divelet-ml.
- [x] Integrate the Pyodide kernel.
- [x] Integrate the xeus-python kernel.
- [ ] Establish dependency and package-management conventions.
- [ ] Define dataset-loading and caching strategies.
- [x] Provide a reproducible local build process (`build.sh`, using
      [jl-forge](https://github.com/decitre/jl-forge)).
- [x] Publish a browser-accessible demonstration
      (<https://decitre.github.io/divelet-ml/>, deployed by GitHub Actions).
- [ ] Document browser and runtime limitations.

Pyodide and xeus-python will be evaluated separately. Compatibility with one
runtime does not imply compatibility with the other.

## Phase 4 — Automated Validation

**Status: In progress**

Introduce automated testing to verify that adapted notebooks remain
executable and educationally consistent.

### Tasks

- [ ] Add tests for numerical correctness (started:
      `tests/test_attention_scoring.py` for 11.3).
- [ ] Validate representative notebook outputs.
- [x] Introduce browser-based execution tests (`tests/smoke_test.py`: every
      notebook runs in both kernels, with no request to other hosts and no
      browser console error).
- [x] Integrate tests into GitHub Actions (`tests/test_divelet_ml.py`,
      `tests/test_attention_scoring.py` and `tests/test_notebooks.py` in CPython, then the browser smoke test on the
      built site, before deploying).
- [x] Track compatibility separately for Pyodide and xeus-python (the smoke
      test runs each notebook on each site and checks the site's kernel).
- [ ] Document known failures and unsupported features.

Where possible, tests should verify mathematical properties rather than
depend exclusively on exact numerical outputs.

## Phase 5 — Expand Chapter Coverage

**Status: In progress**

After validating the initial chapters, progressively extend coverage to
additional topics of the book.

### Chapter 2 — Preliminaries

Adapting the whole chapter is a condition for the outreach below.

- [x] 2.1 *Data Manipulation*.
- [x] 2.2 *Data Preprocessing*. pandas is now a dependency of `divelet_ml`.
      In Pyodide, the folder the notebook creates makes the kernel's drive
      request its listing from the site; the smoke test allows those 404s.
- [x] 2.3 *Linear Algebra*.
- [x] 2.4 *Calculus*. `divelet_ml` now has the book's plotting helpers
      (`d2l.plt`, `set_figsize`, `set_axes`, `plot`), which 2.6 and later
      chapters use.
- [x] 2.5 *Automatic Differentiation*: NumPy has no automatic
      differentiation. The notebook uses the
      [autograd](https://github.com/HIPS/autograd) library, whose `grad`
      follows JAX's, and so follows the book's JAX edition of the section.
      autograd is a dependency of `divelet_ml`; Pyodide ships version 1.7.0,
      xeus-python gets 1.9.1 from conda-forge. Later chapters that train
      models will compute gradients the same way.
- [x] 2.6 *Probability and Statistics*. PyTorch's `Multinomial` becomes
      `np.random.multinomial`, as in the book's JAX edition.
- [x] 2.7 *Documentation*. `dir(torch.distributions)` becomes
      `dir(np.random)`, as in the book's MXNet edition.

### Chapter 3 — Linear Neural Networks for Regression

- [x] 3.1 *Linear Regression*. The section's three figures are the first
      book images adapted: the neuron drawing is Wikimedia Commons'
      `Neuron.svg` (CC BY-SA 3.0), credited in the notebook.
- [x] 3.2 *Object-Oriented Design for Implementation*. `divelet_ml` now has
      the book's `ProgressBoard`, `Module`, `DataModule` and `Trainer`, which
      later chapters build on.
- [x] 3.3 *Synthetic Regression Data*.
- [x] 3.4 *Linear Regression Implementation from Scratch*. Training
      computes gradients with autograd, as planned in 2.5:
      `d2l.param_grads` differentiates a training step with respect to the
      model's parameter arrays, in place of `loss.backward()`, and the
      optimizer's `step(grads)` updates them.
- [x] 3.5 *Concise Implementation of Linear Regression*. `divelet_ml.nn`
      gains `MSELoss`, `Module.parameters()` and initializers for
      `LazyLinear`, and `divelet_ml.optim.SGD` stands in for
      `torch.optim.SGD`.
- [x] 3.6 *Generalization* (no code).
- [x] 3.7 *Weight Decay*. `optim.SGD` takes PyTorch's parameter groups with
      `weight_decay`.

### Chapter 4 — Linear Neural Networks for Classification

Upstream: `chapter_linear-classification/`. The first chapter with a real
dataset (Fashion-MNIST) and the first where the browser's compute budget
matters.

- **Where the data lives.** Fashion-MNIST (about 30 MB) cannot go in
  `content/`: xeus-python packs all of `content/` into the archive every
  notebook's kernel downloads at startup. `build.sh` copies the dataset's four
  original gzip files (MIT license, Zalando SE; the license is copied with them)
  to `data/fashion-mnist/` at the root of the site, and `d2l.load_fashion_mnist`
  fetches them from there with a synchronous request (both kernels have the
  `js` module) and decodes them with NumPy, in place of torchvision. Outside
  JupyterLite, it downloads them once to `../data`. This closes "Define
  dataset-loading and caching strategies" in Phase 3 for datasets the site can
  host; the browser's HTTP cache avoids a second download.
- **Training budget.** Softmax regression takes about 0.5 s per epoch in
  xeus-python and 1.3 s in Pyodide, so 4.4 and 4.5 train for the book's 10
  epochs in both.

- [x] 4.1 *Softmax Regression* (no code).
- [x] 4.2 *The Image Classification Dataset*: `FashionMNIST`, `show_images`.
- [x] 4.3 *The Base Classification Model*: `Classifier`.
- [x] 4.4 *Softmax Regression Implementation from Scratch*, with gradients
      from autograd as in 3.4.
- [x] 4.5 *Concise Implementation of Softmax Regression*: `nn.Sequential`,
      `nn.Flatten` and `nn.cross_entropy`, which is numerically stable.
- [x] 4.6 *Generalization in Classification* (no code).
- [x] 4.7 *Environment and Distribution Shift* (no code). Two of the book's
      three figures are stock photographs (Getty Images) and are linked, not
      embedded.

### Chapter 5 — Multilayer Perceptrons

Upstream: `chapter_multilayer-perceptrons/`. Builds on Chapter 4's data and
classifier.

- **Pyodide's speed.** Pyodide has no BLAS. One epoch of the book's MLP (256
  hidden units, minibatches of 256) took 0.8 s on desktop CPython, 4.1 s in
  xeus-python and 33.9 s in Pyodide (hand-written NumPy loops). Ten epochs would
  take about six minutes in Pyodide, past the smoke test's 120 s limit, so
  `d2l.browser_epochs(10)` gives 10 epochs outside JupyterLite, 3 in xeus-python
  and 1 in Pyodide. Each notebook that trains MLPs (5.2 and 5.6) says so and
  that its accuracies are lower than the book's. Not yet measured: the cost of
  the `Trainer`'s per-epoch plot in each kernel.

- [x] 5.1 *Multilayer Perceptrons*: activation plots, with gradients from
      autograd.
- [x] 5.2 *Implementation of Multilayer Perceptrons*.
- [x] 5.3 *Forward Propagation, Backward Propagation, and Computational
      Graphs* (no code).
- [x] 5.4 *Numerical Stability and Initialization*.
- [x] 5.5 *Generalization in Deep Learning* (no code).
- [x] 5.6 *Dropout*: `nn.Dropout` is differentiated by autograd.
- [x] 5.7 *Predicting House Prices on Kaggle*. The two CSV files are not
      hosted: Kaggle's competition rules (7.B) forbid making the data available
      to people who have not joined the competition, although the data page
      lists the license as MIT. In JupyterLite, the notebook shows the book's
      links; the reader downloads the files and uploads them with the file
      browser, and `d2l.download` finds them by name in the file browser (in
      both kernels, under `/drive`). The smoke test does the same upload in the
      Lab app, and skips the Notebook app, which has no file browser. See
      `NOTICE.md`; the MIT copyright text for the data is still to be added.

### Later chapters

Not yet planned in detail:

- Convolutional neural networks.
- Recurrent neural networks.
- Attention mechanisms and transformers (11.3 is done; the rest of Chapter 11
  builds on the Chapter 3–5 framework).

Adaptation priorities will depend on educational value, browser feasibility,
implementation complexity, and runtime performance.

Not every original implementation will necessarily be reproducible in a
browser-only environment.

## Phase 6 — Browser-Native Machine Learning

**Status: Exploratory**

Evaluate whether selected examples can benefit from browser-native machine
learning capabilities beyond NumPy.

Potential technologies include:

- [Transformers.js](https://huggingface.co/docs/transformers.js/)
- [lmLite](https://github.com/decitre/lmlite)
- WebAssembly-based numerical libraries.
- WebGPU-based inference or computation.

These technologies will be considered where they support the educational
objectives of a chapter.

They are not intended to replace foundational implementations with opaque
pretrained models.

## Project Principles

Throughout development, divelet-ml will prioritize:

1. **Educational fidelity:** Preserve the concepts and progression of the book.
2. **Browser-native execution:** Avoid requiring a remote computation server.
3. **Simplicity:** Prefer NumPy and lightweight dependencies where practical.
4. **Transparency:** Document implementation differences and limitations.
5. **Reproducibility:** Make examples testable and their execution requirements
   explicit.
6. **Licensing compliance:** Preserve upstream attribution and applicable
   licensing conditions.

## Upstream and Attribution

divelet-ml is an independent adaptation of
[Dive into Deep Learning](https://github.com/d2l-ai/d2l-en).

It is not an official edition and is not affiliated with or endorsed by the
book's original authors or maintainers.

See [NOTICE.md](NOTICE.md) and [CONTENT-LICENSE.md](CONTENT-LICENSE.md)
for attribution and licensing information.

### Outreach

Present divelet-ml to the book's maintainers and readers once there is enough to show:
at least Chapter 2 (Preliminaries) adapted, the upstream code licenses verified
(Phase 1), and the site published.

- [ ] Open an issue on `d2l-ai/d2l-en` presenting divelet-ml to the maintainers:
  what it is, that it is independent and unendorsed, and how it attributes the book.
  Ask whether they would welcome a link from the book.
- [ ] Reply in the [book's forum](https://discuss.d2l.ai/c/english-version/5)
  thread of each adapted section with a link to its divelet-ml notebook, rather than
  opening a new topic.

## Feedback

Suggestions, compatibility reports, and contributions are welcome through
[GitHub Issues](https://github.com/decitre/divelet-ml/issues).