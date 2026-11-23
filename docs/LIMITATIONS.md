# Limitations and known gaps

What divelet-ml cannot do, or does differently from the book, and why. It is the one place for these; the notebooks say the same where it matters for them, and
[ROADMAP.md](../ROADMAP.md) has the numbers behind the training budgets.

## Runtime

- **One thread, no GPU.** A browser kernel runs on one thread and the library does not use WebGPU, so `num_gpus`, `try_gpu` and `device` arguments are removed from the book's code.
- **Pyodide's NumPy has no BLAS.** Matrix products are slower there than in xeus-python, by a factor that grows with the width of the matrices: measured for the book's translation models,
  xeus-python is 2 to 6 times slower than CPython and Pyodide 4 to 6 times for small models and 21 to 41 times for 256-unit ones (`spikes/seq-cost/README.md`).
- **The first load is slow.** A kernel downloads its packages the first time (the smoke test allows four minutes), and the browser caches them afterwards.
- **Memory.** A WebAssembly kernel holds at most a few gigabytes, and one array at most 2 GB (NumPy raises "array is too big"). Fashion-MNIST at 224 x 224 would take about 12 GB and at 96 x 96 about 2.2 GB, so the notebooks that
  need those cannot load the book's data.

## Training is shorter, or absent

- **Reduced training**, marked in the notebook with the tablet and `d2l.browser_epochs` or `d2l.browser_examples` (the book's values outside the browser, fewer in xeus-python and fewer still in Pyodide), with a figure of the book's full run after
  each training cell (`tools/full_run_figures.py`): 5.2, 5.6, 7.6, 8.5, 9.5, 9.6, 10.1 to 10.3 and 12.11. In the browser the curves show less training and the accuracies are lower than the book's.
- **No training**: 8.1 to 8.4 and 8.6 to 8.8 build their models, run a forward pass and show the layer shapes, because an epoch of the book's data takes hours in Pyodide.
- Not adapted yet: the sections that [ROADMAP.md](../ROADMAP.md) lists as not done, among them most of Chapter 11 and Chapters 13 to 21.

## Data

- **A page cannot read files from other hosts**: the book's data host sends no CORS headers. The site hosts what it can, with its checksum, and `build.sh` copies it in: Fashion-MNIST, *The Time Machine*, the airfoil self-noise data
  and the English to French sentence pairs. They are credited in [NOTICE.md](../NOTICE.md). The smoke test fails if a notebook requests any other host.
- **Kaggle's house-price files** (5.7) cannot be hosted: the reader downloads them and uploads them with the file browser.
- The site as a whole has to stay under GitHub Pages' 1 GB limit (about 290 MB at the last count).

## The library is not PyTorch

`divelet_ml` is NumPy plus [autograd](https://github.com/HIPS/autograd): a model is an object whose floating-point arrays are held as attributes, and `d2l.param_grads` (which `Trainer` calls) differentiates a function with respect to them.

- **Parameters** are arrays held as attributes of a module or its submodules; arrays inside a list or a dict are not found. State that is not a parameter is listed in the module's `buffers`.
- **Subsets of PyTorch's API.** `optim.SGD` has no Nesterov momentum or dampening; `Adagrad` no `lr_decay` or `initial_accumulator_value`; `RMSprop` no `momentum`, `centered` or `maximize`; `Adam` no `amsgrad`, `maximize` or decoupled
  weight decay; `nn.LSTM` and `nn.GRU` are not bidirectional (10.4's `BiGRU` is defined but cannot be built); `lr_scheduler` has `MultiStepLR` only.
- **`nn.LazyConv2d` takes `padding` before `stride`**, unlike PyTorch: pass both by keyword. This is a known difference that has not been fixed, because it would change a public signature.
- **NumPy draws `float64`**: parameters are cast to `float32`. NumPy 2 also promotes a `float32` array to `float64` when it meets a `np.float64` scalar (`np.sqrt(h)`): use Python floats.
- **Under autograd**: no in-place updates of traced values (`Y += X` becomes `Y = Y + X`), `torch.cat` is `np.concatenate`, and use `autograd.numpy` for anything inside a model. `np.dot` of an N-dimensional array and a matrix is slow in NumPy; use `@`.
- Only the part of the book's `d2l` that adapted chapters use exists; `divelet_ml.__all__` lists it.

## Plots

Matplotlib draws inline as SVG. A progress plot that redraws at every update costs real time in the browser (5.2 found it a real cost), which the training budgets include. 3D plots work.

## What is tested, and where

- **CPython**: every notebook runs without error or stderr output (`tests/test_notebooks.py`), and the library is checked against reference implementations (`tests/test_divelet_ml.py`). The notebooks' printed values are **not** checked.
- **Browsers**: `tests/smoke_test.py` runs every notebook in both kernels, in desktop Chrome and in WebKit emulating an iPad, and fails on a request to another host or a console error. Real Safari, real iPads and Firefox are not tested.
- **In CI**, the smoke test covers Chapters 2 to 5 and 11 only. Chapters 6 to 10 and 12 are smoke-tested on the author's machine before each pull request, not in CI.
- The Pyodide kernel in WebKit sometimes does not start on a CI runner; a failed smoke job runs its failed tests once more.
