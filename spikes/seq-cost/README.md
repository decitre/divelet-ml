# Spike: the cost of Chapters 10 and 11

How long does a training step of the book's translation models, and an epoch of its character models, take in plain NumPy and autograd? The numbers
fix the training budgets in ROADMAP.md (Chapters 10 and 11).

- `step_cost.py`: one forward and backward step of a GRU encoder-decoder (10.7) and of a Transformer (11.7), at the book's sizes and at smaller ones. The book's
  `MTFraEng` keeps 512 training and 128 validation pairs and builds its vocabularies from them (195 and 213 tokens), so those are the book's vocabularies; the last two
  rows use the whole data set's (10,012 and 17,851) to show why it is not used.
- `text_cost.py`: one epoch of the character model of 9.5 and 10.1 to 10.3 with a tanh RNN, an LSTM, a GRU and a two-layer GRU.
- `browser_run.py SCRIPT [xeus] [pyodide]`: runs `step_cost.py` or `text_cost.py` in the JupyterLite kernels of a built site (`workspace/public`) with Playwright and Chrome and prints the output. Run it on a
  quiet machine: it is a timing measurement.

Both scripts need only NumPy and autograd. `python step_cost.py` and `python text_cost.py` print one line per configuration.

## Results (2026-11-22; one Mac, Chrome, one run each, with little else running; xeus-python NumPy 2.5.4, Pyodide NumPy 2.2.5)

Training step, forward and backward, batch 128, 9 steps (ms):

| model | vocabulary | CPython | xeus-python | Pyodide |
|---|---|---:|---:|---:|
| GRU encoder-decoder, 256 units, 2 layers (the book's) | 195 / 213 | 94 | 582 | 3,815 |
| the same, 128 units | 195 / 213 | 52 | 211 | 902 |
| the same, 64 units | 195 / 213 | 31 | 93 | 247 |
| Transformer, 256 units, 2 blocks, 4 heads (the book's) | 195 / 213 | 169 | 1,072 | 3,568 |
| the same, 128 units | 195 / 213 | 71 | 325 | 914 |
| the same, 64 units | 195 / 213 | 35 | 121 | 278 |
| GRU encoder-decoder, 256 units (whole data set) | 10,012 / 17,851 | 1,067 | n/a | n/a |
| Transformer, 256 units (whole data set) | 10,012 / 17,851 | 1,171 | n/a | n/a |

The whole data set's rows do not run in the browser kernels: "array is too big" in xeus-python (and the Pyodide run skips them), see below.

One epoch of the character model, 32 units, 1024 x 32, 10 training and 5 validation minibatches (s):

| model | CPython | xeus-python | Pyodide |
|---|---:|---:|---:|
| tanh RNN | 0.60 | 1.24 | 2.25 |
| LSTM | 1.05 | 2.72 | 6.11 |
| GRU | 1.09 | 2.64 | 5.14 |
| two-layer GRU | 1.77 | 4.67 | 9.56 |

## What it shows

- **Pyodide pays for wide matrices.** The ratio to CPython is 2 to 3 (xeus-python) and 4 to 6 (Pyodide) for the small character models, but 6 and 21 to 41 for the 256-unit models: Pyodide's NumPy has no BLAS. The 8 times in
  `d2l.browser_epochs`' docstring does not hold for them. The book's 120 steps are 7.6 minutes (GRU) and 7.1 minutes (Transformer) in Pyodide, against 70 s and 2.1 minutes in xeus-python.
- **`np.dot` on a three-dimensional array is a trap**: `np.dot(x, W)` with `x` of shape (128, 9, 256) takes 40 ms, `x @ W` 0.2 ms. The first version of `step_cost.py` used `np.dot` and made the Transformer look 6 times
  slower (1.1 s a step against 0.17 s). The library's `nn.Linear` uses `@`, so it was not affected.
- **But `matmul` of a three-dimensional array and a matrix has a costly backward under autograd**: the vjp multiplies batch by batch and builds a (batch, in, out) intermediate before summing. For (128, 9, 256) @ (256, 213) the
  gradient takes 2.5 ms against 0.7 ms with the array flattened to two dimensions (`x.reshape(-1, 256) @ W`), 2.0 against 0.8 ms at 256 x 256, and for the whole data set's (256, 17,851) output layer 245 ms against 57 ms and a
  2.3 GB intermediate, which is what stops the browser kernels. `nn.Linear` should flatten the leading dimensions (ROADMAP, Chapter 10).
- **A NumPy 2 trap**: a `float32` array times a `np.float64` scalar such as `np.sqrt(h)` is `float64`. The first Transformer in `step_cost.py` was in `float64` for that reason; the scripts use Python floats now.
- The spike's own character-model RNN is about 2 times slower than the library's in xeus-python (1.24 s against the 0.6 s measured for 9.5) and 1.2 times in Pyodide (2.25 s against 1.9 s), so ROADMAP.md scales the LSTM and
  GRU epochs by that before using them.

Numbers are from one run each on one machine; the browser runs were in Chrome only (no WebKit or iPad), and `browser_run.py` has to be run on a quiet machine.
