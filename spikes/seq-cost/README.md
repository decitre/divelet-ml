# Spike: the cost of Chapters 10 and 11

How long does a training step of the book's translation models, and an epoch of its character models, take in plain NumPy and autograd? The numbers
fix the training budgets in ROADMAP.md (Chapters 10 and 11).

- `step_cost.py`: one forward and backward step of a GRU encoder-decoder (10.7) and of a Transformer (11.7), at the book's sizes and at smaller ones. The book's
  `MTFraEng` keeps 512 training and 128 validation pairs and builds its vocabularies from them (195 and 213 tokens), so those are the book's vocabularies; the last two
  rows use the whole data set's (10,012 and 17,851) to show why it is not used.
- `text_cost.py`: one epoch of the character model of 9.5 and 10.1 to 10.3 with a tanh RNN, an LSTM, a GRU and a two-layer GRU.
- `browser_run.py`: runs `step_cost.py` in both JupyterLite kernels of a built site (`workspace/public`) with Playwright and Chrome and prints the output. Run it on a
  quiet machine: it is a timing measurement. **Not run yet**: the browser columns below are empty.

Both scripts need only NumPy and autograd. `python step_cost.py` and `python text_cost.py` print one line per configuration.

## CPython (2026-11-22, one Mac, with another job running: upper bounds)

| model | batch, vocabulary | step (ms) |
|---|---|---:|
| GRU encoder-decoder, 256 units, 2 layers (the book's) | 128, 195 / 213 | 105 |
| the same, 128 units | 128, 195 / 213 | 58 |
| the same, 64 units | 128, 195 / 213 | 34 |
| Transformer, 256 units, 2 blocks, 4 heads (the book's) | 128, 195 / 213 | 1,082 |
| the same, 128 units | 128, 195 / 213 | 187 |
| the same, 64 units | 128, 195 / 213 | 70 |
| GRU encoder-decoder, 256 units (whole data set) | 128, 10,012 / 17,851 | 2,774 |
| Transformer, 256 units (whole data set) | 128, 10,012 / 17,851 | 3,467 |

| character model, 32 units, 1024 x 32, 10 training and 5 validation minibatches | one epoch (s) |
|---|---:|
| tanh RNN | 0.40 |
| LSTM | 0.81 |
| GRU | 0.84 |
| two-layer GRU | 1.52 |

The first run of `step_cost.py` had its Transformer in `float64`, because NumPy 2 promotes `float32` arrays that meet a `np.float64` scalar such as `np.sqrt(h)`
(1.28 s instead of 1.08 s per step at the book's size); the scripts use Python floats now. A library written with `np.sqrt` on a scalar would have the same slowdown.
The Transformer's step grows faster than its width (256 units are 5.8 times the 128-unit step for 4 times the arithmetic): this spike's layers are plain autograd
`dot` calls on three-dimensional arrays, and the library's `nn.Linear` does the same.

Estimates for the browser, from the ratios that Chapter 9 measured for its RNN (xeus-python about 3 times CPython, Pyodide 8 to 10 times), are in ROADMAP.md.
