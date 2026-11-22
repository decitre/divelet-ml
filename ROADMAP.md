# Roadmap

divelet-ml aims to make [Dive into Deep Learning](https://d2l.ai/)
accessible through JupyterLite while preserving the educational structure,
mathematical explanations, learning objectives, and exercises of the original
textbook.

Development proceeds incrementally, one chapter at a time.

**Current status:** Early development. Static compatibility analysis has been
performed. Chapters 2 (*Preliminaries*), 3 (*Linear Neural Networks for
Regression*), 4 (*Linear Neural Networks for Classification*) and 5
(*Multilayer Perceptrons*), 6 (*Builders' Guide*) and 7 (*Convolutional Neural
Networks*) are adapted in full, along with 9 (*Recurrent Neural Networks*), 8 (*Modern Convolutional Neural Networks*, whose models are built and shown but trained only in 8.5) 11.3 *Attention Scoring Functions* and 12 (*Optimization Algorithms*). The adapted notebooks run in JupyterLite with both the Pyodide and
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
- [x] Define dataset-loading and caching strategies (Chapter 4: datasets the site can
      host are served from its own root and cached by the browser; Kaggle's
      house-price files, which cannot be hosted, are uploaded by the reader, 5.7).
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
- [x] *Extra*: Automatic Differentiation from Scratch, after 2.5. Not a section of the
      book: a divelet-ml notebook that builds a scalar autograd engine and a tiny
      network, adapted from [micrograd](https://github.com/karpathy/micrograd) (MIT)
      and checked against autograd's `grad` and by numerical differentiation. 2.5, 3.4
      and 5.3 point to it.
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
      `NOTICE.md`.

### Chapter 6 — Builders' Guide

Upstream: `chapter_builders-guide/`. How models are built, initialized, saved and
run. Most of it is about PyTorch's `nn.Module` machinery, so the work is to give
`divelet_ml.nn` the same surface, following the book's JAX and MXNet editions
where PyTorch has no NumPy equivalent. Little compute: no section trains.

- [x] 6.1 *Layers and Modules*: `Sequential`, `ReLU` and `LazyLinear` exist. The
      book's own `MySequential` registers layers with `add_module` and walks
      `children()`, so `Module` needs those two.
- [x] 6.2 *Parameter Management*: needs `state_dict()`, `named_parameters()` and
      a `.data` equivalent (parameters are plain arrays here). Tied parameters
      (the same layer twice): check that `param_grads` counts a shared array once.
- [x] 6.3 *Parameter Initialization*: `LazyLinear`'s `weight_init` takes a
      function of the shape, like Flax's initializers, so add `normal`,
      `constant`, `zeros` and `xavier_uniform` (the last is also LeNet's, in 7.6)
      and an `apply`.
- [x] 6.4 *Lazy Initialization*: `LazyLinear` already waits for its first
      input. Show `weight is None` before the forward pass, in place of
      PyTorch's `UninitializedParameter`.
- [x] 6.5 *Custom Layers*: arrays held as attributes are parameters, so
      `nn.Parameter` is not needed; say so in the notebook.
- [x] 6.6 *File I/O*: `torch.save` and `torch.load` become `np.save`, `np.load`
      and `load_state_dict`. Writing files already works in both kernels (2.2),
      but check that the files survive in the Notebook app's drive.
- [x] 6.7 *GPUs*: a browser kernel has no GPU. Decision for review: keep the
      section with `num_gpus()` returning 0 and `cpu`, `gpu`, `try_gpu` shims, so
      the book's code runs unchanged, and say that WebGPU is Phase 6, or drop
      the section and say why.

### Chapter 7 — Convolutional Neural Networks

Upstream: `chapter_convolutional-neural-networks/`. The first chapter whose
layers are not matrix products, and whose training cost is a risk.

- **Convolution and pooling with autograd.** `corr2d`, `nn.LazyConv2d`,
  `nn.MaxPool2d`, `nn.AvgPool2d` and `nn.Sigmoid` are new. The convolution must
  be differentiable by autograd. Candidates: `autograd.scipy.signal.convolve`
  (needs scipy, which Pyodide ships and conda-forge has, as a new dependency)
  or an im2col matrix product (plain NumPy, uses `@`). Spike first: implement
  both, check the gradients against finite differences, and time them in each
  kernel before writing notebooks.
- **LeNet's training budget (7.6).** The notebook trains 10 epochs on 10,000
  Fashion-MNIST images, not the book's 60,000, outside the browser and in
  xeus-python, and on 2,000 in Pyodide (`d2l.browser_examples`). The smoke test
  gives it the default 120 s limit: it passed in CI, but went past it on one
  developer laptop under load, so raise its `RUN_MS_NOTEBOOKS` entry if it
  becomes flaky. Per-epoch times in each kernel are not recorded here.

- [x] 7.1 *From Fully Connected Layers to Convolutions* (no code).
- [x] 7.2 *Convolutions for Images*: `corr2d`, a `Conv2D` from scratch, and the
      edge-detection kernel learned by gradient descent, with `param_grads` in
      place of `loss.backward()`.
- [x] 7.3 *Padding and Stride*: `LazyConv2d` with `padding` and `stride`.
- [x] 7.4 *Multiple Input and Multiple Output Channels*: `corr2d_multi_in`,
      `corr2d_multi_in_out`, and the 1×1 convolution.
- [x] 7.5 *Pooling*: `pool2d`, `MaxPool2d`, `AvgPool2d`.
- [x] 7.6 *Convolutional Neural Networks (LeNet)*: `LeNet(Classifier)`,
      Xavier initialization from 6.3, `layer_summary`. `try_gpu` goes (6.7).
      Outside the browser it trains on 10,000 images, not the book's 60,000.

### Chapter 8 — Modern Convolutional Neural Networks

Upstream: `chapter_convolutional-modern/`. AlexNet, VGG, NiN, GoogLeNet, batch
normalization, ResNet, DenseNet and design principles. Builds on Chapter 7's
convolutions and LeNet.

- **Most of it cannot train in a browser.** The book trains AlexNet and NiN on
  224 × 224 images and VGG and GoogLeNet on 96 × 96 images. LeNet at 28 × 28 was already
  at the limit (10,000 images, 2,000 in Pyodide), and these models are far larger.
  Decision: 8.5 *Batch Normalization* keeps its training, on LeNet at 28 × 28 as in
  the book; every other section builds its model, runs a forward pass and shows the
  layer shapes (`layer_summary`), and says in the notebook that training is left out and why.
- **Measured, so no spike is needed for the forward pass.** `spikes/js-conv/` times a
  convolution in Pyodide (about 1.5 GMAC/s) and counts the multiply-adds of every model from
  the book's layer shapes (`count_macs.py`). One forward pass of one image takes under a second
  in Pyodide for every model at the book's size, except VGG-11 at full width (7.5 GMAC, about
  5 s), so `layer_summary` can use the book's input sizes. Training is out of reach in NumPy:
  an epoch of 60,000 images takes 4 to 31 hours in Pyodide for AlexNet, VGG, NiN, GoogLeNet,
  ResNet-18 and DenseNet, and even RegNetX-32 takes over an hour. Even the batch-normalized LeNet
  of 8.5 takes about 34 s per epoch there, so its notebook uses `d2l.browser_epochs` and
  `d2l.browser_examples` like 7.6. A model that stayed on the GPU in TensorFlow.js would take
  about 3 to 19 minutes per epoch on one Mac (the iPad is not measured). A JS-backed `nn` that
  hands single ops to TensorFlow.js from Python does not get there: the second spike
  (`kernel_bench.py`, Pyodide in a worker calling TensorFlow.js on WebGL, copies included)
  sped up Chapter 5's MLP step by 1.8 times and slowed down 7.6's LeNet step (0.7 times),
  because each call copies its arrays in and out and costs about 0.75 ms. Only tensors that
  stay on the GPU between ops would gain much, which is a different library from NumPy and
  is not part of this plan. All figures are estimates from one machine, with backward
  counted as twice the forward pass.
- **New in `divelet_ml`** (added, with tests, and used by 8.1 to 8.8): `nn.BatchNorm1d`, `nn.BatchNorm2d` and their lazy
  versions, with the training/evaluation switch and running statistics that are buffers (a module lists them in `buffers`,
  so they are not parameters, but they are in `state_dict()`); `nn.AdaptiveAvgPool2d((1, 1))`; `groups` for `conv2d`,
  `Conv2d` and `LazyConv2d`, which ResNeXt and RegNet need; `Sequential.add_module`, which ResNet and DenseNet use to add
  named blocks; and `init_cnn` and `Classifier.layer_summary`, which the book saves from its earlier chapters. Channel
  concatenation and the residual sum are `np.concatenate` and `+` in the notebooks. Checked: AlexNet, VGG-11, GoogLeNet,
  ResNet-18, RegNetX-32 and DenseNet all build with these layers and give a `(1, 10)` output at the book's sizes; one
  forward pass takes 0.01 to 0.7 s in CPython. `FashionMNIST(resize=...)` exists, but loading all 60,000 images at
  224 x 224 takes about 12 GB, so the notebooks cannot load the book's resized data and show shapes only.

- [x] 8.1 *Deep Convolutional Neural Networks (AlexNet)*.
- [x] 8.2 *Networks Using Blocks (VGG)*.
- [x] 8.3 *Network in Network (NiN)*.
- [x] 8.4 *Multi-Branch Networks (GoogLeNet)*.
- [x] 8.5 *Batch Normalization*: trains LeNet with batch norm.
- [x] 8.6 *Residual Networks (ResNet) and ResNeXt*.
- [x] 8.7 *Densely Connected Networks (DenseNet)*.
- [x] 8.8 *Designing Convolution Network Architectures*.

### Chapter 9 — Recurrent Neural Networks

Upstream: `chapter_recurrent-neural-networks/`. Sequences, text, language models and
recurrent networks. The text models are small enough for a browser; training them is the cost.

- **The text.** Sections 9.2 to 9.6 use *The Time Machine*, which the book downloads
  from its own S3 bucket, and a browser cannot read that (the problem of 5.7). The book is
  in the public domain, so the plan is to host the text on the site like Fashion-MNIST:
  `build.sh` copies it to `data/timemachine/` with a checksum, and `d2l.download` finds it
  there. License checked (2026-11-19): the copy is `timemachine.txt`, 179 KB, SHA-1
  `090b5e7e70c295757f55df93cb0a180b9691891a` (the value the book's `download` checks), titled
  "The Time Machine, by H. G. Wells [1898]". It contains no Project Gutenberg header, footer,
  trademark or license text, and it is not byte-identical to Gutenberg's current ebook 35
  (the two share most passages, at about 32,800 and 32,900 words), so its exact source is unknown.
  The work is in the public domain: first published in 1895 (the file's title says 1898), and Wells died in 1946. Gutenberg's
  permissions page says that its public-domain ebooks can be redistributed by anyone, with or
  without the trademark, and that crediting Gutenberg is optional; only commercial use of the
  trademark needs royalties, and the file does not use it. Gutenberg follows US law; the
  UK and EU term (life plus 70 years) also ended on 1 January 2017, which is from general
  knowledge and not checked against a source here. So hosting the file is fine. Credit
  H. G. Wells and the book's copy in `NOTICE.md` when the file is added, with a link to Gutenberg's
  ebook 35 as a place to read the text.
- **Training budget.** Measured (`spikes/rnn-epoch/`): one epoch of the book's 9.5 language
  model (28 characters, 32 hidden units, minibatches of 1024 sequences of 32 steps, 10 training
  and 5 validation minibatches, forward and backward with autograd, gradient clipping) takes
  0.2 s in CPython, 0.6 s in xeus-python and 1.9 s in Pyodide, in Chrome on one Mac. The
  book's 100 epochs would take about 1 minute in xeus-python and 3 minutes in Pyodide, so
  unlike Chapters 5 and 7 this chapter can keep the book's batch size, steps and probably
  its epochs. Not measured: the `Trainer`'s per-epoch plot (5.2 found it a real cost), a
  CI runner, WebKit and the iPad, which are slower. With the plot, the notebook as it runs
  (Run All in the Lab app, Chrome, one Mac) takes 125 s for 100 epochs in xeus-python and 66 s
  for 30 epochs in Pyodide, about 1.2 s and 2.2 s per epoch. 9.5 and 9.6 therefore use
  `d2l.browser_epochs(100, pyodide=25, xeus=50)` and a `RUN_MS_NOTEBOOKS` limit of 300 s, and
  say so. After 20 epochs the validation perplexity is 10.6 against 7.5 after the book's
  100 epochs (this implementation, seed 0), and the generated text is a few words, not yet
  sentences.
- **New in `divelet_ml`** (added, with tests, and used by 9.2 to 9.6): `nn.RNN`, written as a loop
  over time steps that autograd differentiates; `Vocab` and `TimeMachine`, which reads the hosted text
  in JupyterLite and downloads the book's copy elsewhere; gradient clipping in the `Trainer`
  (`clip_gradients`); `RNNScratch`, `RNNLMScratch`, `RNN`, `RNNLM` with `predict`; and `check_len`.
  `build.sh` copies the text to `data/timemachine/` with its checksum. Not done: `d2l.browser_epochs`
  values for these notebooks, and the one-hot encoding is NumPy's `eye`, not a layer.

- [x] 9.1 *Working with Sequences*.
- [x] 9.2 *Converting Raw Text into Sequence Data*: reads the hosted text through `d2l.time_machine_text`.
- [x] 9.3 *Language Models*.
- [x] 9.4 *Recurrent Neural Networks*.
- [x] 9.5 *Recurrent Neural Network Implementation from Scratch*.
- [x] 9.6 *Concise Implementation of Recurrent Neural Networks*.
- [x] 9.7 *Backpropagation Through Time* (no code).

### Chapter 10 — Modern Recurrent Neural Networks

Upstream: `chapter_recurrent-modern/`. LSTM, GRU, deep and bidirectional RNNs, the translation data set, the encoder-decoder
architecture, sequence-to-sequence learning and beam search. Builds on Chapter 9's `nn.RNN`, `RNNLMScratch`, `RNNLM`,
`TimeMachine` and the `Trainer` with gradient clipping.

- **What the book really trains.** 10.1 to 10.3 are Chapter 9's character model with another cell: the Time Machine, 32 hidden
  units, minibatches of 1024 sequences of 32 steps (50 epochs for LSTM and GRU, 100 for the deep GRU), each notebook training a
  scratch and a concise model. 10.4 builds a bidirectional RNN and does not train it; 10.5, 10.6 and 10.8 have no training (10.8, beam
  search, has no code). 10.7 trains the GRU encoder-decoder on English to French, and the book's translation data class
  `MTFraEng` keeps **512 training and 128 validation sentence pairs and builds the vocabularies from those 640 pairs only**
  (195 source and 213 target tokens with `min_freq=2`), so 30 epochs at batch size 128 are 120 steps. The whole data set (167,130 pairs,
  vocabularies of 10,012 and 17,851 tokens) is not what the book trains on, and could not be: see below.
- **Measured** (`spikes/seq-cost/`: plain NumPy and autograd, forward and backward; one Mac, Chrome, one run each, on a machine with little else
  running; xeus-python has NumPy 2.5.4 and Pyodide 2.2.5). One epoch of the character model (1024 x 32, 10 training and 5 validation minibatches, 32 hidden units)
  without the progress plot, in CPython, xeus-python and Pyodide: tanh RNN 0.60, 1.24 and 2.25 s; LSTM 1.05, 2.72 and 6.11 s; GRU 1.09, 2.64 and 5.14 s; two-layer GRU
  1.77, 4.67 and 9.56 s. The spike's RNN is about 2 times slower than the library's in xeus-python (1.24 s against the 0.6 s of 9.5, whose `nn.RNN` is tuned) and 1.2 times in
  Pyodide (2.25 s against 1.9 s), so the library's epoch with the plot (about +0.6 s in xeus-python and +0.3 s in Pyodide, from 9.5) is about 1.9 s for the LSTM and the GRU and 2.9 s for the two-layer GRU in
  xeus-python, and 5.5, 4.7 and 8.4 s in Pyodide. One training step of the book's GRU encoder-decoder (256 embedding and hidden units, two layers, batch 128, 9 steps, 195 and 213 tokens)
  takes 94 ms in CPython, 0.58 s in xeus-python and **3.8 s in Pyodide**: the book's 120 steps are 11 s, 70 s and **7.6 minutes**. With 128 units it is 52 ms, 0.21 s and 0.90 s (108 s in Pyodide). The ratio to CPython is 2 to 3
  times in xeus-python and 4 to 6 times in Pyodide for the small character models, but 6 times and **41 times** for the 256-unit encoder-decoder: Pyodide's NumPy has no BLAS, and
  a model with wide matrices pays for it (`d2l.browser_epochs`' docstring says 8 times; the wide models are far worse). With the whole data set's vocabularies a step takes 1.1 s in CPython (the
  browser kernels cannot hold its output layer: a WebAssembly array is limited to 2 GB), so an epoch of 1,306 steps is about 23 minutes in CPython and out of reach in a browser.
- **Decisions.** 10.1 and 10.2 use `d2l.browser_epochs(50, pyodide=20, xeus=50)` for each of their two trainings: about 190 s in xeus-python and 190 to 220 s in Pyodide per notebook.
  10.3 uses `d2l.browser_epochs(100, pyodide=15, xeus=40)`: about 230 s and 250 s. 10.7 keeps the book's data and sizes and uses `d2l.browser_epochs(30, pyodide=15, xeus=30)`: 70 s in
  xeus-python and about 230 s in Pyodide (15 epochs are 60 steps; the alternative, 128 units for all 30 epochs, is 108 s in Pyodide and a different model). Each gets a tablet-marked note, a
  `RUN_MS_NOTEBOOKS` limit of 480 s, and, for 10.1 to 10.3, the figures of the full runs from `tools/full_run_figures.py` as in 9.5 and 9.6. The translations are poor, as in the book, because
  512 sentences is all it trains on. These are estimates for the library's classes from the spike's numbers: confirm each notebook's time in the browser when it is written.
- **The data.** `fra-eng.zip` is the Tatoeba English to French file of ManyThings' Anki pairs, which the book downloads from its own bucket
  and a browser cannot read (the problem of 5.7). 3,420,152 bytes (11.7 MB as `fra.txt`, 167,130 lines), SHA-1
  `94646ad1522d915e7b0f9296181140edcf86a4f5` (the value the book's `download` checks), SHA-256 `fa96382b2c8b609e1662654c1e6e6baeb8d30b3bf63dfb4a5a261cc36347df13`.
  License checked (2026-11-22): its `_about.txt` says the pairs are released under the licence of the source, the Tatoeba Project, and links
  CC BY 2.0 and the attribution "www.manythings.org/anki and tatoeba.org"; Tatoeba's terms of use say that its sentences are under CC BY 2.0
  France (some under CC0), the condition being attribution of the author. The file carries no per-sentence author names, so the credit
  can only be to the Tatoeba contributors and ManyThings, as the book does: **to settle before hosting** (host it with that credit, or keep the
  download-and-upload route of 5.7). Plan: `build.sh` copies the zip to `data/fra-eng/` with its checksum (3.4 MB of the site's 1 GB), `MTFraEng` reads it
  in JupyterLite and downloads it elsewhere, and `NOTICE.md` and the notebooks credit Tatoeba, ManyThings and CC BY 2.0 FR.
- **New in `divelet_ml`** (to add, each with a test, and run in Pyodide before a notebook uses it): **`nn.Linear` and `nn.LazyLinear` flatten the leading dimensions before the product**
  (`(X.reshape(-1, in) @ W.T).reshape(...)`): autograd's backward of a `matmul` of a three-dimensional array and a matrix builds a (batch, in, out) intermediate, which the spike measured as 2.5 to 4 times
  slower than the flattened product at the book's sizes, and as 2.3 GB for the whole data set's output layer, over WebAssembly's limit (CPython has the memory, so only the browser fails); it speeds up the
  sequence models of Chapters 9 to 11 too, and `d2l.MultiHeadAttention`'s projections need the same care; `nn.Embedding`; `nn.LSTM` and `nn.GRU` as
  loops over time steps like `nn.RNN` (with `num_layers` and `dropout`, and `bidirectional` if 10.4 needs it); `d2l.extract` (a zip); `MTFraEng` with
  `build`, `truncate_pad` and `get_dataloader`; `Encoder`, `Decoder`, `EncoderDecoder`, `Seq2SeqEncoder`, `Seq2Seq` (loss with the padding mask,
  `predict_step`), `init_seq2seq` (Xavier on the GRU weights) and `bleu`. The notebooks' `LSTMScratch`, `GRUScratch`, `StackedRNNScratch` and `BiRNNScratch` are
  `#@save` and move to the library when a later section uses them (`d2l.GRU` in 10.7). The `Trainer` already passes `batch[:-1]` to the model, which the
  four-tensor translation batches fit. An embedding that autograd indexes has a dense gradient of vocabulary size by width: harmless at 200 tokens.

- [ ] 10.1 *Long Short-Term Memory (LSTM)*: reduced epochs, two trainings.
- [ ] 10.2 *Gated Recurrent Units (GRU)*: as 10.1.
- [ ] 10.3 *Deep Recurrent Neural Networks*: reduced epochs.
- [ ] 10.4 *Bidirectional Recurrent Neural Networks*: builds the model, no training.
- [ ] 10.5 *Machine Translation and the Dataset*: needs the hosted data.
- [ ] 10.6 *The Encoder--Decoder Architecture*.
- [ ] 10.7 *Sequence-to-Sequence Learning for Machine Translation*: the book's 30 epochs on 512 pairs.
- [ ] 10.8 *Beam Search* (no code).

### Chapter 11 — Attention Mechanisms and Transformers

Upstream: `chapter_attention-mechanisms-and-transformers/`. Queries, keys and values, attention pooling, scoring functions
(11.3, done), Bahdanau attention, multi-head attention, self-attention and positional encoding, the Transformer, the vision Transformer and
large-scale pretraining. Builds on Chapter 10's sequence-to-sequence code and on 11.3.

- **What the book trains.** 11.1 to 11.3, 11.5 and 11.6 only compute forward passes and plot them (11.2 is Nadaraya-Watson kernel regression on 40 points). 11.4
  trains the GRU encoder-decoder with Bahdanau attention and 11.7 the Transformer, both on the 512 pairs of Chapter 10's `MTFraEng` for 30 epochs
  at batch size 128. 11.8 trains a vision Transformer on Fashion-MNIST resized to 96 x 96 for 10 epochs. 11.9 has no code.
- **Measured** (`spikes/seq-cost/`, as in Chapter 10). One training step of the book's Transformer (256 hidden units, 2 blocks of 4 heads, a 64-unit feed-forward layer, batch 128, 9 steps, 195 and 213 tokens)
  takes 169 ms in CPython, 1.07 s in xeus-python and **3.57 s in Pyodide**, so the book's 120 steps are 20 s, 2.1 minutes and **7.1 minutes**. With 128 hidden units a step is 71 ms, 0.33 s and 0.91 s (110 s in
  Pyodide), and with 64 it is 35 ms, 0.12 s and 0.28 s. (A first version of the spike used `np.dot` on three-dimensional arrays, 200 times slower than `@` there, and made the Transformer look 6 times worse: the library's
  `nn.Linear` uses `@`.) The Bahdanau decoder costs about one and a half times 10.7's GRU step (an estimate: not run). **11.8's vision Transformer** takes about 241 million multiply-adds per
  image forward (36 patches of 16 x 16 into 512 units, then 2 blocks of 37 tokens with 512 units and a 2048-unit MLP), about 0.7 billion with the backward pass, which at Pyodide's 1.5 GMAC/s (`spikes/js-conv`;
  matrices that wide are slower in the spike above, so this is generous) is half a second per image and about 8 hours for one epoch of 60,000; and the 96 x 96 training images in `float32` are 2.2 GB, more than a browser kernel
  holds. Estimates from one machine, with backward counted as twice the forward pass.
- **Decisions.** 11.1 to 11.3, 11.5 and 11.6 run as the book has them. 11.7 keeps the book's width and uses `d2l.browser_epochs(30, pyodide=15, xeus=30)`: about 2.1 minutes in xeus-python and 3.6 minutes in
  Pyodide, with a tablet-marked note (the earlier plan to cut the width to 128 is dropped: only Pyodide needs a cut, and it can have fewer epochs). 11.4 uses `d2l.browser_epochs(30, pyodide=10, xeus=30)`
  (estimated 105 s and 230 s, from 1.5 times 10.7's step; measure it when written). Each gets a `RUN_MS_NOTEBOOKS` limit of 480 s. 11.8 builds the model, runs a forward pass on one image and shows
  the shapes, says why it does not train (cost from `spikes/js-conv/count_macs.py`) and keeps the book's training cell as a code block, as 8.1 does. 11.9 is prose (the
  pretrained models of its discussion are Phase 6's, through lmLite).
- **New in `divelet_ml`:** `masked_softmax`, `DotProductAttention` and `AdditiveAttention` move from the 11.3 notebook into the library (the book saves them; 11.4, 11.5 and 11.7 use
  them) with tests; `MultiHeadAttention`, `PositionalEncoding`, `AttentionDecoder`, the Bahdanau decoder, `AddNorm`, `PositionWiseFFN` and the Transformer
  encoder and decoder blocks; `nn.LayerNorm` and `nn.GELU`; for 11.8 `PatchEmbedding`, `ViTMLP`, `ViTBlock` and `ViT` (`cls_token` and `pos_embedding` are arrays held as
  attributes, so they are parameters; `LazyConv2d` takes `padding` before `stride`, so pass both by keyword). `torch.repeat_interleave` is `np.repeat`, and `torch.bmm` is `np.matmul`.
  Needs Chapter 10's `nn.Embedding`, `MTFraEng`, `Seq2Seq` and `bleu` first.

- [ ] 11.1 *Queries, Keys, and Values*.
- [ ] 11.2 *Attention Pooling by Similarity*.
- [x] 11.3 *Attention Scoring Functions*.
- [ ] 11.4 *The Bahdanau Attention Mechanism*: after Chapter 10's library.
- [ ] 11.5 *Multi-Head Attention*.
- [ ] 11.6 *Self-Attention and Positional Encoding*.
- [ ] 11.7 *The Transformer Architecture*: the book's width, fewer epochs in Pyodide.
- [ ] 11.8 *Transformers for Vision*: builds the model, no training.
- [ ] 11.9 *Large-Scale Pretraining with Transformers* (no code).

**Order of work for Chapters 10 and 11.** The browser timing spike is done (above). The library and the data hosting for Chapter 10 (a PR, like #41), then 10.1 to 10.4, then 10.5 to 10.8, then
Chapter 11's library, then 11.1, 11.2, 11.5 and 11.6, then 11.4, 11.7 and 11.8, then 11.9. The smoke test's limits and the `browser_epochs` values are set for each notebook from its own run in the browser.

### Chapter 12 — Optimization Algorithms

Upstream: `chapter_optimization/`. Eleven sections: optimization and deep learning, convexity, gradient descent,
stochastic and minibatch gradient descent, momentum, Adagrad, RMSProp, Adadelta, Adam, and learning rate scheduling.
Compute is small: most sections minimize one- and two-variable functions and plot the paths, and 12.5 to 12.10 train a
linear regression on 1,500 examples with five features. The exception is 12.11.

- **Measured** (one epoch of the book's minibatch loop on 1,500 examples, forward and backward by autograd, batch size 1, the
  slowest case): 0.17 s in CPython and 0.26 s in Pyodide, about 0.11 and 0.17 ms per step. The book's two epochs per run cost
  well under a second in either, so 12.1 to 12.10 need no cut. 3D plots (`plot_wireframe`, used in 12.1 and 12.2) work in
  Pyodide's matplotlib.
- **12.11 is the cost.** It trains LeNet on Fashion-MNIST for 30 epochs in five training runs (constant rate, square-root,
  multi-step and cosine schedules, and cosine with warmup; the rate changed by hand and the factor scheduler are not trained). At 0.42 million multiply-adds per image, one
  epoch of 60,000 images is about 50 s in Pyodide (an estimate, as for 8.5), so the book's budget is about 2.5 hours there, and it
  would also exceed the 300 s the CPython notebook test gives a notebook. Decision: use `d2l.browser_epochs` and
  `d2l.browser_examples` as in 7.6, with values measured when the notebook is written so that the CPython test runs in minutes, a
  `RUN_MS_NOTEBOOKS` limit, and a figure of the book's full runs for each training cell (about an hour of CPython for the five
  runs, made once with `tools/full_run_figures.py`; CI does not run it).
- **The data** of 12.5 to 12.10 is the airfoil self-noise data set (NASA; UCI), which the book downloads from its own bucket, and
  a browser cannot read that (the problem of 5.7 and 9.2). License checked (2026-11-21): UCI's page for the data set says CC BY 4.0
  (its API leaves the field empty); the creators are Thomas Brooks, D. Pope and Michael Marcolini. The book's copy,
  `airfoil_self_noise.dat`, is 59,984 bytes with 1,503 rows of six numbers and SHA-1 `76e5be1548fd8222e5074cf0faae75edff8cf93f`,
  the value the book's `download` checks; it has the same number of rows as UCI's, but I did not compare the two byte for byte.
  Plan: host the file on the site like Fashion-MNIST (`build.sh` copies it to `data/airfoil/` with its checksum) and credit its
  creators and the CC BY 4.0 license in `NOTICE.md` and in the notebooks that use it.
- **Tab syntax.** All eleven sections mark their cells with `#@tab pytorch`, `#@tab mxnet`, `#@tab all` and similar, not the
  `%%tab` of the earlier chapters: the conversion has to read both.
- **New in `divelet_ml`** (added, with tests, and run in Pyodide with synthetic data; no notebook uses them yet): the book's older helpers that these sections use and save for each other: `train_2d`, `show_trace_2d`,
  `annotate`; `Timer`, `Animator`, `Accumulator`; `linreg`, `squared_loss`, `evaluate_loss`, `load_array`, `get_data_ch11`,
  `train_ch11`, `train_concise_ch11`; and for 12.11 `load_data_fashion_mnist` and the training function, which becomes the
  `Trainer`'s autograd loop. The book's optimizer functions read `p.grad`; the adapted ones take the gradients as an argument, as
  `optim.SGD.step(grads)` does. The concise versions need `optim.SGD` with momentum, `Adagrad`, `RMSprop`, `Adadelta` and `Adam`, and
  `lr_scheduler.MultiStepLR` with `get_last_lr()` (12.11 also changes `param_groups[0]["lr"]` by hand, so the optimizers need
  `param_groups`). 12.5 times loops against matrix products with `Timer`: the notebook says that the times in a browser differ from
  the book's.
- **Done before the notebooks:** the library work above, the hosted data (`build.sh` copies `airfoil_self_noise.dat` to `data/airfoil/` with its
  checksum, `d2l.airfoil_self_noise` reads it in JupyterLite, and `NOTICE.md` credits it), and `nn.MSELoss(reduction=...)`. In the
  adapted optimizer functions, the signature is `trainer_fn(params, grads, states, hyperparams)`. Left: the measurement for 12.11 and
  the notebooks.

- [x] 12.1 *Optimization and Deep Learning*.
- [x] 12.2 *Convexity*.
- [x] 12.3 *Gradient Descent*.
- [x] 12.4 *Stochastic Gradient Descent*.
- [x] 12.5 *Minibatch Stochastic Gradient Descent*.
- [x] 12.6 *Momentum*.
- [x] 12.7 *Adagrad*.
- [x] 12.8 *RMSProp*.
- [x] 12.9 *Adadelta*.
- [x] 12.10 *Adam*.
- [x] 12.11 *Learning Rate Scheduling*: trains on 2,000 images (1,000 in Pyodide) for the book's 30 epochs; a figure of the full run follows each of its five training cells.

### Later chapters

68 of the book's 165 sections are adapted (Chapters 2 to 9, 11.3 and 12). The rest, in the book's order, with an
estimate of how well each suits a browser kernel (a judgement, to revisit when a
chapter is planned; none besides Chapters 10, 11 and 12, above, is planned in detail):

| Chapter | Sections | Outlook |
|---|---:|---|
| 10 Modern Recurrent Neural Networks | 8 | Planned above |
| 11 Attention Mechanisms and Transformers | 8 left (11.3 done) | Planned above |
| 12 Optimization Algorithms | done | Planned above |
| 13 Computational Performance | 7 | Mostly GPUs, hybridization and multi-GPU: little applies to a browser |
| 14 Computer Vision | 14 | Heavy: large models and pretrained weights. Detection, segmentation and classification of given images can use lmLite's pretrained pipelines (Phase 6); training and fine-tuning cannot run |
| 15 Natural Language Processing: Pretraining | 10 | Heavy: large corpora. Pretrained embeddings and tokenizers can come from lmLite; pretraining cannot run |
| 16 Natural Language Processing: Applications | 7 | Heavy; suits Phase 6 for the parts that apply a pretrained model |
| 17 Reinforcement Learning | 3 | Small environments; unassessed |
| 18 Gaussian Processes | 3 | Mostly NumPy; unassessed |
| 19 Hyperparameter Optimization | 5 | Many training runs; unassessed |
| 20 Generative Adversarial Networks | 2 | Training cost; unassessed |
| 21 Recommender Systems | 10 | Dataset hosting and size to check |
| Appendix: Mathematics for Deep Learning | 11 | Mostly equations; a good candidate |
| Appendix: Tools for Deep Learning | 8 | Jupyter, cloud and GPU setup: little applies |

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

**lmLite, state on 2026-11-22** (inference only, no training; `main` is ahead of release 0.12.1). It runs pretrained
models in the browser, from xeus-python and Pyodide, through Transformers.js and ONNX Runtime Web (WASM or WebGPU), with a
native onnxruntime backend for headless runs. Chat, embeddings and tokenizers; task pipelines (zero-shot classification,
speech recognition, translation, text to speech); images as inputs, with image classification, zero-shot image classification,
image features and object detection; image outputs (background removal, depth estimation, segmentation, image-to-image);
bare ONNX models through `OnnxModel`, NumPy in and NumPy out; and `pipe.map()` over many inputs with progress and
cancellation. Open on its side: text-to-audio generation, a diffusion spike, docs, `OnnxModel.map()` progress.

What it offers divelet-ml: the parts of Chapters 14 to 16 that apply a pretrained model to given data, and any model
exported to ONNX run from NumPy. What it does not: training, so Chapters 8, 11 and 12 stay as they are. No notebook uses it yet.
Constraints: models must be self-hosted (the smoke test rejects other hosts, and the site is about 290 MB of GitHub Pages'
1 GB), and the native backend lets `tests/test_notebooks.py` run such a notebook in CPython.

**Spike (next): 14.3 *Object Detection and Bounding Boxes*.** The section draws the bounding boxes of a dog and a cat on a
photo by hand. Question: can `LLM.pipeline("object-detection")` or an `OnnxModel` detect them, in both kernels and natively,
with a hosted model small enough for the site, and does it fit the section without replacing the book's foundation? The
answer decides whether Chapter 14 sections that apply pretrained models are adapted, and how a notebook hosts and tests its
model.

*Result (`spikes/lmlite-detect`, 2026-11-22):* `Xenova/yolos-tiny` (Apache-2.0, 9.7 MB quantized) run as a bare `OnnxModel` finds the
dog and the cat in both kernels (about 1 s per image after a 4 to 7 s load) and natively (0.1 s), with the same boxes in the two browser
kernels and a few pixels' drift natively. It fits 14.3 as a tablet-marked section after the book's code. Blocked: `OnnxModel` is on
lmLite's `main` but not in its PyPI release 0.12.1, and divelet-ml installs released packages only; wait for a lmLite release that has it,
then adapt 14.3 with a self-hosted model.

*Update, 2026-11-22:* lmLite 0.13.0 is on PyPI (an 88 KB `py3-none-any` wheel, Apache-2.0, Python 3.11 or later; its core dependencies are
`ipywidgets` and `tqdm`, and the extras are `images` (NumPy, Pillow), `audio` and `native`) and has `OnnxModel`, `OnnxModel.map()`, `Pipeline.map()` and the image and speech
pipelines. The spike's native run against that wheel gives the same boxes, so the blocker is gone. Left for the 14.3 notebook: lmLite is not on
conda-forge, so the xeus-python build would mount the package from the wheel (as it does `divelet_ml`) and the Pyodide build take it from PyPI; `build.sh` hosts the model and its
`config.json`; and the notebook must run in the browser kernels of this site's own build, which the spike did not do.

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