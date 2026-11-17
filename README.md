<img src="https://raw.githubusercontent.com/decitre/divelet-ml/main/divelet-ml.svg"
     alt="Divelet ML — Deep learning, in your browser"
     width="600">

[![Build site][ci-badge]][ci-url]
[![Coverage][codecov-badge]][codecov-url]
[![Try xeus-python][xeus-badge]][xeus-tree]
[![Try Pyodide][pyodide-badge]][pyodide-tree]
[![Content: CC BY-SA 4.0][license-badge]][license-url]
[![Code: Apache 2.0][apache-badge]][apache]

# divelet-ml

**divelet for ML: Dive into Deep Learning, adapted for JupyterLite.**

divelet-ml is an experimental, browser-native adaptation of the Jupyter notebooks that accompany the book [*Dive into Deep Learning*][d2l-site] ("the book" below), designed to make its educational content and interactive examples accessible through [JupyterLite][jupyterlite] without requiring a local Python installation or a remote computation server.

The project aims to preserve the book's chapter structure, mathematical explanations, learning objectives, and exercises while adapting its framework-dependent Python implementations for execution directly in the browser.

**Project status:** Early development. Chapters 2 to 5 are adapted in full, and Chapters 6 and 7 are planned next. Adapted sections run in the browser with no installation. A section link opens its notebook with the xeus-python kernel; **[p]** opens it with Pyodide instead.

| Section | Original |
|---|---|
| 1 Introduction (no code) | [d2l.ai][d2l-1] |
| [2.1 Data Manipulation][xeus-2.1] [\[p\]][pyodide-2.1] | [d2l.ai][d2l-2.1] |
| [2.2 Data Preprocessing][xeus-2.2] [\[p\]][pyodide-2.2] | [d2l.ai][d2l-2.2] |
| [2.3 Linear Algebra][xeus-2.3] [\[p\]][pyodide-2.3] | [d2l.ai][d2l-2.3] |
| [2.4 Calculus][xeus-2.4] [\[p\]][pyodide-2.4] | [d2l.ai][d2l-2.4] |
| [2.5 Automatic Differentiation][xeus-2.5] [\[p\]][pyodide-2.5] | [d2l.ai][d2l-2.5] |
| [2.6 Probability and Statistics][xeus-2.6] [\[p\]][pyodide-2.6] | [d2l.ai][d2l-2.6] |
| [2.7 Documentation][xeus-2.7] [\[p\]][pyodide-2.7] | [d2l.ai][d2l-2.7] |
| [3.1 Linear Regression][xeus-3.1] [\[p\]][pyodide-3.1] | [d2l.ai][d2l-3.1] |
| [3.2 Object-Oriented Design for Implementation][xeus-3.2] [\[p\]][pyodide-3.2] | [d2l.ai][d2l-3.2] |
| [3.3 Synthetic Regression Data][xeus-3.3] [\[p\]][pyodide-3.3] | [d2l.ai][d2l-3.3] |
| [3.4 Linear Regression Implementation from Scratch][xeus-3.4] [\[p\]][pyodide-3.4] | [d2l.ai][d2l-3.4] |
| [3.5 Concise Implementation of Linear Regression][xeus-3.5] [\[p\]][pyodide-3.5] | [d2l.ai][d2l-3.5] |
| [3.6 Generalization][xeus-3.6] [\[p\]][pyodide-3.6] | [d2l.ai][d2l-3.6] |
| [3.7 Weight Decay][xeus-3.7] [\[p\]][pyodide-3.7] | [d2l.ai][d2l-3.7] |
| [4.1 Softmax Regression][xeus-4.1] [\[p\]][pyodide-4.1] | [d2l.ai][d2l-4.1] |
| [4.2 The Image Classification Dataset][xeus-4.2] [\[p\]][pyodide-4.2] | [d2l.ai][d2l-4.2] |
| [4.3 The Base Classification Model][xeus-4.3] [\[p\]][pyodide-4.3] | [d2l.ai][d2l-4.3] |
| [4.4 Softmax Regression Implementation from Scratch][xeus-4.4] [\[p\]][pyodide-4.4] | [d2l.ai][d2l-4.4] |
| [4.5 Concise Implementation of Softmax Regression][xeus-4.5] [\[p\]][pyodide-4.5] | [d2l.ai][d2l-4.5] |
| [4.6 Generalization in Classification][xeus-4.6] [\[p\]][pyodide-4.6] | [d2l.ai][d2l-4.6] |
| [4.7 Environment and Distribution Shift][xeus-4.7] [\[p\]][pyodide-4.7] | [d2l.ai][d2l-4.7] |
| [5.1 Multilayer Perceptrons][xeus-5.1] [\[p\]][pyodide-5.1] | [d2l.ai][d2l-5.1] |
| [5.2 Implementation of Multilayer Perceptrons][xeus-5.2] [\[p\]][pyodide-5.2] | [d2l.ai][d2l-5.2] |
| [5.3 Forward Propagation, Backward Propagation, and Computational Graphs][xeus-5.3] [\[p\]][pyodide-5.3] | [d2l.ai][d2l-5.3] |
| [5.4 Numerical Stability and Initialization][xeus-5.4] [\[p\]][pyodide-5.4] | [d2l.ai][d2l-5.4] |
| [5.5 Generalization in Deep Learning][xeus-5.5] [\[p\]][pyodide-5.5] | [d2l.ai][d2l-5.5] |
| [5.6 Dropout][xeus-5.6] [\[p\]][pyodide-5.6] | [d2l.ai][d2l-5.6] |
| [5.7 Predicting House Prices on Kaggle][xeus-5.7] [\[p\]][pyodide-5.7] | [d2l.ai][d2l-5.7] |
| [11.3 Attention Scoring Functions][xeus-11.3] [\[p\]][pyodide-11.3] | [d2l.ai][d2l-11.3] |

The notebooks' sources are in [`content/`][content].

Build the site locally with `./build.sh` (needs uv and micromamba).

## Python package

The notebooks use `divelet_ml`, a small NumPy counterpart of the book's `d2l` library. The site's browser kernels already include it. To use it elsewhere:

```bash
pip install divelet-ml
```

```python
import divelet_ml as d2l
```

It provides `add_to_class`, `HyperParameters`, `DataModule`, `check_shape`, the plotting helpers `plt`, `set_figsize`, `set_axes`, `plot` and `show_heatmaps`, and the training framework of Chapter 3: `ProgressBoard`, `Module`, `Trainer`, `SyntheticRegressionData`, `SGD`, `LinearRegressionScratch`, `LinearRegression`, `Classifier` and `SoftmaxRegression`, all adapted from the book's `d2l/torch.py`. `ArrayLoader` yields minibatches of NumPy arrays in place of PyTorch's `DataLoader`, and `param_grads` computes gradients with [autograd](https://github.com/HIPS/autograd) in place of PyTorch's `backward`. Its `divelet_ml.nn` module (`from divelet_ml import nn`) has NumPy stand-ins for the parts of `torch.nn` the notebooks use: `Module`, `Dropout`, `LazyLinear`, `MSELoss`, `Sequential`, `Flatten`, `ReLU`, `cross_entropy` and `softmax`; `divelet_ml.optim` has `SGD`, in place of `torch.optim.SGD`. Plots use matplotlib. The package grows only as adapted chapters need more of it. Its code is licensed under the Apache License 2.0; the notebooks are not part of the package.

## Motivation

Dive into Deep Learning is an open-source, interactive textbook combining mathematical explanations with executable code.

Its implementations rely primarily on deep learning frameworks such as PyTorch, TensorFlow, JAX, and MXNet. These environments can be difficult to reproduce in a browser-only Python runtime.

divelet-ml explores whether the same educational progression can be maintained using lightweight, browser-compatible implementations.

The objective is not to reproduce the entire deep learning framework ecosystem in the browser, but to make the underlying concepts accessible through executable, pedagogically faithful examples.

## Design Principles

- **Educational fidelity:** Preserve the book's explanations, mathematical formulations, learning objectives, and exercises wherever possible.
- **Browser-native execution:** Run examples entirely in the browser without requiring a remote Python kernel.
- **NumPy first:** Prefer NumPy implementations when they adequately illustrate the underlying concepts.
- **Framework independence:** Avoid unnecessary dependencies on full deep learning frameworks.
- **Progressive adaptation:** Adapt chapters individually, prioritizing educational value and technical feasibility.
- **Reproducibility:** Document runtime requirements, implementation differences, and compatibility limitations.

## Technical Approach

### JupyterLite

divelet-ml targets JupyterLite with two Python execution environments:

- **Pyodide:** Python compiled to WebAssembly, with access to a browser-compatible scientific Python ecosystem.
- **xeus-python:** A browser-native Python kernel built using WebAssembly.

Compatibility will be evaluated independently for each environment.

### NumPy and Deep Learning

Where practical, framework-specific operations will be replaced with NumPy equivalents.

This includes tensor manipulation, linear algebra, probability, optimization, and selected neural network implementations.

Some chapters may require more substantial adaptation, particularly those involving automatic differentiation, GPU acceleration, large datasets, or computationally expensive training.

The goal is to preserve the concepts being taught rather than provide identical framework APIs.

### Transformers.js and lmLite

For selected chapters involving modern language models or inference, divelet-ml may explore integration with [Transformers.js][transformers-js] through [lmLite][lmlite].

lmLite provides Python access to browser-native machine learning inference.

This integration is optional and is not intended to replace the mathematical and algorithmic foundations of the book.

## Compatibility Assessment

An initial static audit of the book's source repository has been developed to identify:

- Framework-specific implementations and dependencies.
- Candidate chapters for direct adaptation.
- Potential incompatibilities with browser-native Python runtimes.
- Estimated adaptation effort.
- Opportunities for alternative browser-native implementations.

The audit distinguishes static compatibility estimates from actual execution results.

**Static analysis does not establish that a chapter runs successfully in JupyterLite.** Browser execution and educational fidelity must be validated separately.

The initial audit references the book's commit:

`23d7a5aecceee57d1292c56e90cce307f183bb0a`

## Development Plan

Chapters 2 (*Preliminaries*), 3 (*Linear Neural Networks for Regression*), 4 (*Linear Neural Networks for Classification*) and 5 (*Multilayer Perceptrons*) are adapted in full, together with section 11.3 (*Attention Scoring Functions*), which tested the approach on a later chapter. Chapters 4 and 5 are the first with a real dataset and longer training: Fashion-MNIST is served from the site, and in the browser the notebooks that train multilayer perceptrons run fewer epochs than the book, as each notebook says.

The PyTorch implementations serve as the primary reference.

Each adaptation is reviewed for educational fidelity, browser compatibility, dependencies, and reproducibility, and runs in the browser under both kernels in CI.

Later chapters follow the results of these.

See [ROADMAP.md][roadmap] for further details.

## Relationship to Dive into Deep Learning

divelet-ml is an independent adaptation of the original [Dive into Deep Learning][d2l-repo] project.

It is not an official edition of the book and is not affiliated with or endorsed by the original authors.

The project intends to preserve the educational contributions of the book while adapting selected implementations for browser-native execution.

Adapted chapters will identify their original sources and document substantive modifications.

## License and Attribution

The original Dive into Deep Learning book content is distributed under the [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

Adapted educational content derived from the book will retain the applicable attribution and ShareAlike requirements.

divelet-ml's own source code (the `divelet_ml` package, build scripts and tests) is licensed under the [Apache License 2.0][apache]. Third-party materials may be subject to separate licenses; these will be reviewed and their required notices preserved before redistribution.

See [CONTENT-LICENSE.md][license-url] and [NOTICE.md][notice] for licensing and attribution details.

## Feedback and Contributions

divelet-ml is an experimental project exploring browser-native deep learning education.

Feedback is particularly welcome on:

- The educational usefulness of framework-independent implementations.
- The fidelity of NumPy adaptations to the book's original examples.
- JupyterLite compatibility, performance, and runtime limitations.
- Opportunities for browser-native machine learning beyond traditional Python frameworks.

Please share suggestions, bug reports, and use cases through [GitHub Issues][issues]. To set up a development environment, run the tests or make a release, see [CONTRIBUTING.md][contributing].

[ci-badge]: https://github.com/decitre/divelet-ml/actions/workflows/build.yml/badge.svg
[ci-url]: https://github.com/decitre/divelet-ml/actions/workflows/build.yml
[codecov-badge]: https://img.shields.io/codecov/c/github/decitre/divelet-ml/main?flag=python&label=Coverage&logo=codecov
[codecov-url]: https://codecov.io/gh/decitre/divelet-ml
[xeus-tree]: https://decitre.github.io/divelet-ml/xeus/tree/
[pyodide-tree]: https://decitre.github.io/divelet-ml/pyodide/tree/
[xeus-badge]: https://img.shields.io/badge/try-xeus--python-f7dc1e.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0Ni43IDQ2LjciPgogIDxwYXRoCiAgICBmaWxsPSIjNDBCM0MzIgogICAgZD0iTTUgMGMyLjUuOCA0LjUgMiA0LjYgMi4xIDEuOSAxLjMgMy42IDMuMSA0LjggNS4xLjQuNiA1LjIgNy44IDguNiAxMi45LS44IDEuMi0xLjUgMi4zLTIuMSAzLjEgMC0uMS0uMS0uMS0uMS0uMkMxNy42IDE3LjkgMTEuNCA4LjYgMTEgNy45Yy0uOS0xLjMtMi0yLjUtMy41LTMuNCAwIDAtMS41LTEtMy41LTEuNkMzLjIgMi42IDEuOCAyLjQgMCAyLjJ2NDIuNGMxLjgtLjIgMy4yLS40IDQtLjcgMi0uNiAzLjUtMS42IDMuNS0xLjYgMS41LTEgMi43LTIuMiAzLjUtMy40LjQtLjcgNi42LTkuOSAxMC4xLTE1LjJDMjQgMTkuMSAzMS40IDguMSAzMS45IDcuM2MxLjItMiAyLjktMy44IDQuOC01LjEgMC0uMSAyLjEtMS4zIDQuNi0yLjFMNDEuOCAwSDV6CiAgICAgICBNNDYuNyAyLjJjLTEuOC4yLTMuMi40LTQgLjctMi4xLjYtMy42IDEuNi0zLjYgMS42LTEuNS45LTIuNiAyLjEtMy41IDMuNC0uNC43LTYuNiA5LjktMTAuMSAxNS4yLTIuOSA0LjQtMTAuMyAxNS41LTEwLjggMTYuMi0xLjIgMi0yLjkgMy44LTQuOCA1LjEgMCAuMS0yLjEgMS4zLTQuNiAyLjFsLS41LjFoMzYuOGwtLjQtLjFjLTIuNS0uOC00LjUtMi00LjYtMi4xLTEuOS0xLjMtMy42LTMuMS00LjgtNS4xLS40LS42LTUuMi03LjgtOC42LTEyLjkuOC0xLjIgMS41LTIuMyAyLjEtMy4xIDAgLjEuMS4xLjEuMiAzLjUgNS4zIDkuNyAxNC41IDEwLjEgMTUuMi45IDEuMyAyIDIuNSAzLjUgMy40IDAgMCAxLjUgMSAzLjUgMS42LjguMyAyLjIuNSA0IC43VjIuMnoiCiAgLz4KPC9zdmc+Cgo=
[pyodide-badge]: https://img.shields.io/badge/try-Pyodide-f7dc1e.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI2OCIgaGVpZ2h0PSI2OCIgdmlld0JveD0iMCAwIDE3Ljk5MiAxNy45OTIiPjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKC40NiAuNDQ4KSI+PGNpcmNsZSBjeD0iODMuNzI3IiBjeT0iODUuNjMzIiByPSI2LjU5OCIgc3R5bGU9ImZpbGw6I2Y3ZGMxZTtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6LjI2NDU4MzMyIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNzUuMTkxIC03OS4wMjUpIi8+PHJlY3Qgd2lkdGg9IjUuNjUzIiBoZWlnaHQ9IjEuMDEiIHg9IjgwLjgzMSIgeT0iOTIuMzQ0IiByeD0iLjQxMyIgcnk9Ii40MTMiIHN0eWxlPSJmaWxsOiNmOWY5Zjk7c3Ryb2tlOm5vbmU7c3Ryb2tlLXdpZHRoOjEuOTk5OTk5ODgiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC03NS4xOTEgLTc5LjAyNSkiLz48cmVjdCB3aWR0aD0iNS42NDEiIGhlaWdodD0iMS4wMzEiIHg9IjgwLjgzNyIgeT0iOTMuNjcxIiByeD0iLjQyMSIgcnk9Ii40MjEiIHN0eWxlPSJmaWxsOiNmOWY5Zjk7c3Ryb2tlOm5vbmU7c3Ryb2tlLXdpZHRoOjEuOTk5OTk5NzYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC03NS4xOTEgLTc5LjAyNSkiLz48cmVjdCB3aWR0aD0iMy4yNCIgaGVpZ2h0PSIxLjAyIiB4PSI4Mi4wMzgiIHk9Ijk1LjAxOSIgcng9Ii40MTciIHJ5PSIuNDE3IiBzdHlsZT0iZmlsbDojZjlmOWY5O3N0cm9rZTpub25lO3N0cm9rZS13aWR0aDoxLjk5OTk5OTg4IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNzUuMTkxIC03OS4wMjUpIi8+PGcgc3R5bGU9ImZpbGw6IzFhMWExYSI+PGcgc3R5bGU9ImZpbGw6IzFhMWExYTtzdHJva2Utd2lkdGg6LjI2NDU4MzM1Ij48cGF0aCBkPSJNODcuMDY1IDkwLjI4OXEtLjk5Ni0uMDQxLTEuNDc1LTEuMjE2LS4zNTYtLjg4NC0uMzU2LTIuMTY2IDAtLjA2Ni4wMDgtLjE3LjA1OC0uODcyLjMyMy0xLjYuNDMtMS4xNzMgMS4yMTUtMS4xNzMuMDQyIDAgLjA4Ny4wMDQuNzE1LjA0NS43MTUuODMgMCAuODU3LS43MzYgMS45MDItLjQ2Ny42NjYtLjk4IDEuMDguMTQxLjc4NS4zMSAxLjE3OC4zMTUuNzI3LjkzNS43MjcuMDQxIDAgLjA4Ny0uMDA4LjEzNi0uMDI1LjM4LS4xOTguMjM2LS4xNjIuMzY4LS4xNjIuMDU4IDAgLjExMi4wMzguMTIuMDgyLjEyLjIyMy0uMDA5LjIxNS0uMzA2LjQ0Mi0uMzcyLjI2OS0uODA3LjI2OXptLS4yOC01LjcwMXEtLjI0NS4wMDQtLjQzLjIxLS40NzIuNTM0LS41MTcgMi4yMy4yNDgtLjI1My40OTYtLjU5Ni42ODYtLjkzOS42ODYtMS41ODQgMC0uMDMzLS4wMTctLjA3OC0uMDY2LS4xODItLjIxOS0uMTgyeiIgYXJpYS1sYWJlbD0ibCIgc3R5bGU9ImZpbGw6IzFhMWExYTtzdHJva2Utd2lkdGg6LjI2NDU4MzM1IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNzYuNDA5IC03OS40NzMpIi8+PC9nPjxwYXRoIGQ9Ik04NC45OTggODMuNjYzYS42OTQuNjk0IDAgMCAxLS42OTQuNjk1LjY5NC42OTQgMCAwIDEtLjY5NC0uNjk1LjY5NC42OTQgMCAwIDEgLjY5NC0uNjk0LjY5NC42OTQgMCAwIDEgLjY5NC42OTR6IiBzdHlsZT0iZmlsbDojMWExYTFhO3N0cm9rZTpub25lO3N0cm9rZS13aWR0aDouMjY0NTgzMzIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC03Ni40MDkgLTc5LjQ3MykiLz48ZyBzdHlsZT0iZmlsbDojMWExYTFhO3N0cm9rZS13aWR0aDouMjY0NTgzMzUiPjxwYXRoIGQ9Ik04NC4xNTUgODQuNjY3cS4yNjkuMDA0LjI3Ny4zNWwtLjQ1OSAzLjI4M3ExLjAzNC0uNzkgMS4xNjItLjg5My4xNTMtLjEzMi4zMTQtLjEzNi4wNTggMCAuMTEyLjAzNy4xMi4wODMuMTIuMjExLS4wMDUuMTk5LS4yMjQuMzg1LS42MDcuNTEyLTEuNjA4IDEuMjZsLS4xOSAxLjM2OXEtLjAxMi4wOTUtLjAzNy4xODYtLjE5LjcyNy0uOTU1LjcyNy0uMzggMC0uNjctLjI4NS0uMjg1LS4yODUtLjI4NS0uNjQgMC0uMzczLjE3OC0uNjEyLjA5OS0uMTI0LjM4OC0uMzRsLjk5Mi0uNzQ0cS41MjUtMy40NjguNTM4LTMuNjQ2IDAtLjAyNS4wMDQtLjA0NS4wNDUtLjQ2Ny4zNDMtLjQ2N3ptLTEuMDg3IDUuNTM1LjA3LS41MDQtLjQ1OS4zNTFxLS4yOTguMjIzLS4zMDIuNDIyIDAgLjA3LjAyNS4xMzIuMDcuMTgyLjI2NS4xODIuMDc4IDAgLjE0LS4wMjUuMTk5LS4wODcuMjYtLjU1OHptLjg5Ny02LjMyNXEwLS4wNzQuMDM3LS4xNC4wODctLjE2Mi4yNjktLjE2Mi4wNzQgMCAuMTQ0LjAzNy4xNTguMDg3LjE1OC4yNjUgMCAuMDc0LS4wMzguMTQ1LS4wODYuMTU3LS4yNjQuMTU3LS4wNzUgMC0uMTQ1LS4wMzgtLjE2MS0uMDgyLS4xNjEtLjI2NHoiIGFyaWEtbGFiZWw9ImoiIHN0eWxlPSJmaWxsOiMxYTFhMWE7c3Ryb2tlLXdpZHRoOi4yNjQ1ODMzNSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTc2LjQwOSAtNzkuNDczKSIvPjwvZz48L2c+PC9nPjwvc3ZnPiA=
[license-badge]: https://img.shields.io/badge/Content-CC%20BY--SA%204.0-lightgrey.svg
[apache-badge]: https://img.shields.io/badge/Code-Apache%202.0-blue.svg
[license-url]: https://github.com/decitre/divelet-ml/blob/main/CONTENT-LICENSE.md
[d2l-site]: https://d2l.ai/
[jupyterlite]: https://jupyterlite.readthedocs.io/
[d2l-1]: https://d2l.ai/chapter_introduction/index.html
[content]: https://github.com/decitre/divelet-ml/tree/main/content
[xeus-2.1]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=02_preliminaries/02.01_ndarray.ipynb
[pyodide-2.1]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=02_preliminaries/02.01_ndarray.ipynb
[d2l-2.1]: https://d2l.ai/chapter_preliminaries/ndarray.html
[xeus-2.2]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=02_preliminaries/02.02_data-preprocessing.ipynb
[pyodide-2.2]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=02_preliminaries/02.02_data-preprocessing.ipynb
[d2l-2.2]: https://d2l.ai/chapter_preliminaries/pandas.html
[xeus-2.3]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=02_preliminaries/02.03_linear-algebra.ipynb
[pyodide-2.3]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=02_preliminaries/02.03_linear-algebra.ipynb
[d2l-2.3]: https://d2l.ai/chapter_preliminaries/linear-algebra.html
[xeus-2.4]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=02_preliminaries/02.04_calculus.ipynb
[pyodide-2.4]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=02_preliminaries/02.04_calculus.ipynb
[d2l-2.4]: https://d2l.ai/chapter_preliminaries/calculus.html
[xeus-2.5]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=02_preliminaries/02.05_automatic-differentiation.ipynb
[pyodide-2.5]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=02_preliminaries/02.05_automatic-differentiation.ipynb
[d2l-2.5]: https://d2l.ai/chapter_preliminaries/autograd.html
[xeus-2.6]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=02_preliminaries/02.06_probability.ipynb
[pyodide-2.6]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=02_preliminaries/02.06_probability.ipynb
[d2l-2.6]: https://d2l.ai/chapter_preliminaries/probability.html
[xeus-2.7]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=02_preliminaries/02.07_documentation.ipynb
[pyodide-2.7]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=02_preliminaries/02.07_documentation.ipynb
[d2l-2.7]: https://d2l.ai/chapter_preliminaries/lookup-api.html
[xeus-3.1]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=03_linear-regression/03.01_linear-regression.ipynb
[pyodide-3.1]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=03_linear-regression/03.01_linear-regression.ipynb
[d2l-3.1]: https://d2l.ai/chapter_linear-regression/linear-regression.html
[xeus-3.2]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=03_linear-regression/03.02_oo-design.ipynb
[pyodide-3.2]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=03_linear-regression/03.02_oo-design.ipynb
[d2l-3.2]: https://d2l.ai/chapter_linear-regression/oo-design.html
[xeus-3.3]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=03_linear-regression/03.03_synthetic-regression-data.ipynb
[pyodide-3.3]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=03_linear-regression/03.03_synthetic-regression-data.ipynb
[d2l-3.3]: https://d2l.ai/chapter_linear-regression/synthetic-regression-data.html
[xeus-3.4]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=03_linear-regression/03.04_linear-regression-scratch.ipynb
[pyodide-3.4]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=03_linear-regression/03.04_linear-regression-scratch.ipynb
[d2l-3.4]: https://d2l.ai/chapter_linear-regression/linear-regression-scratch.html
[xeus-3.5]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=03_linear-regression/03.05_linear-regression-concise.ipynb
[pyodide-3.5]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=03_linear-regression/03.05_linear-regression-concise.ipynb
[d2l-3.5]: https://d2l.ai/chapter_linear-regression/linear-regression-concise.html
[xeus-3.6]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=03_linear-regression/03.06_generalization.ipynb
[pyodide-3.6]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=03_linear-regression/03.06_generalization.ipynb
[d2l-3.6]: https://d2l.ai/chapter_linear-regression/generalization.html
[xeus-3.7]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=03_linear-regression/03.07_weight-decay.ipynb
[xeus-4.1]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=04_linear-classification/04.01_softmax-regression.ipynb
[xeus-4.2]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=04_linear-classification/04.02_image-classification-dataset.ipynb
[xeus-4.3]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=04_linear-classification/04.03_classification.ipynb
[xeus-4.4]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=04_linear-classification/04.04_softmax-regression-scratch.ipynb
[xeus-4.5]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=04_linear-classification/04.05_softmax-regression-concise.ipynb
[xeus-4.6]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=04_linear-classification/04.06_generalization-classification.ipynb
[xeus-4.7]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=04_linear-classification/04.07_environment-and-distribution-shift.ipynb
[xeus-5.1]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=05_multilayer-perceptrons/05.01_mlp.ipynb
[xeus-5.2]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=05_multilayer-perceptrons/05.02_mlp-implementation.ipynb
[xeus-5.3]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=05_multilayer-perceptrons/05.03_backprop.ipynb
[xeus-5.4]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=05_multilayer-perceptrons/05.04_numerical-stability-and-init.ipynb
[xeus-5.5]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=05_multilayer-perceptrons/05.05_generalization-deep.ipynb
[xeus-5.6]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=05_multilayer-perceptrons/05.06_dropout.ipynb
[xeus-5.7]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=05_multilayer-perceptrons/05.07_kaggle-house-price.ipynb
[pyodide-3.7]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=03_linear-regression/03.07_weight-decay.ipynb
[pyodide-4.1]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=04_linear-classification/04.01_softmax-regression.ipynb
[pyodide-4.2]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=04_linear-classification/04.02_image-classification-dataset.ipynb
[pyodide-4.3]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=04_linear-classification/04.03_classification.ipynb
[pyodide-4.4]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=04_linear-classification/04.04_softmax-regression-scratch.ipynb
[pyodide-4.5]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=04_linear-classification/04.05_softmax-regression-concise.ipynb
[pyodide-4.6]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=04_linear-classification/04.06_generalization-classification.ipynb
[pyodide-4.7]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=04_linear-classification/04.07_environment-and-distribution-shift.ipynb
[pyodide-5.1]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=05_multilayer-perceptrons/05.01_mlp.ipynb
[pyodide-5.2]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=05_multilayer-perceptrons/05.02_mlp-implementation.ipynb
[pyodide-5.3]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=05_multilayer-perceptrons/05.03_backprop.ipynb
[pyodide-5.4]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=05_multilayer-perceptrons/05.04_numerical-stability-and-init.ipynb
[pyodide-5.5]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=05_multilayer-perceptrons/05.05_generalization-deep.ipynb
[pyodide-5.6]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=05_multilayer-perceptrons/05.06_dropout.ipynb
[pyodide-5.7]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=05_multilayer-perceptrons/05.07_kaggle-house-price.ipynb
[d2l-3.7]: https://d2l.ai/chapter_linear-regression/weight-decay.html
[d2l-4.1]: https://d2l.ai/chapter_linear-classification/softmax-regression.html
[d2l-4.2]: https://d2l.ai/chapter_linear-classification/image-classification-dataset.html
[d2l-4.3]: https://d2l.ai/chapter_linear-classification/classification.html
[d2l-4.4]: https://d2l.ai/chapter_linear-classification/softmax-regression-scratch.html
[d2l-4.5]: https://d2l.ai/chapter_linear-classification/softmax-regression-concise.html
[d2l-4.6]: https://d2l.ai/chapter_linear-classification/generalization-classification.html
[d2l-4.7]: https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html
[d2l-5.1]: https://d2l.ai/chapter_multilayer-perceptrons/mlp.html
[d2l-5.2]: https://d2l.ai/chapter_multilayer-perceptrons/mlp-implementation.html
[d2l-5.3]: https://d2l.ai/chapter_multilayer-perceptrons/backprop.html
[d2l-5.4]: https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html
[d2l-5.5]: https://d2l.ai/chapter_multilayer-perceptrons/generalization-deep.html
[d2l-5.6]: https://d2l.ai/chapter_multilayer-perceptrons/dropout.html
[d2l-5.7]: https://d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html
[xeus-11.3]: https://decitre.github.io/divelet-ml/xeus/notebooks/index.html?path=11_attention-mechanisms-and-transformers/11.03_attention-scoring-functions.ipynb
[pyodide-11.3]: https://decitre.github.io/divelet-ml/pyodide/notebooks/index.html?path=11_attention-mechanisms-and-transformers/11.03_attention-scoring-functions.ipynb
[d2l-11.3]: https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html
[transformers-js]: https://huggingface.co/docs/transformers.js/
[lmlite]: https://github.com/decitre/lmlite
[roadmap]: https://github.com/decitre/divelet-ml/blob/main/ROADMAP.md
[d2l-repo]: https://github.com/d2l-ai/d2l-en
[cc-by-sa]: https://creativecommons.org/licenses/by-sa/4.0/
[notice]: https://github.com/decitre/divelet-ml/blob/main/NOTICE.md
[issues]: https://github.com/decitre/divelet-ml/issues
[contributing]: https://github.com/decitre/divelet-ml/blob/main/CONTRIBUTING.md
[apache]: https://github.com/decitre/divelet-ml/blob/main/LICENSE
