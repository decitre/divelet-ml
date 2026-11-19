"""Spike for the Chapter 8 plan: how long does divelet-ml's NumPy convolution take, forward and with its gradient?

Run in CPython from the repository root:   uv run python spikes/js-conv/numpy_conv.py
Run in the Pyodide of the built site:      node tests/profile_pyodide.mjs spikes/js-conv/numpy_conv.py
Compare with index.html (TensorFlow.js), which times the same cases. The cases are VGG-style 3x3 convolutions at
batch size 8 (the book's VGG uses these widths and sizes divided down for Fashion-MNIST). Keep both files in step.
"""
import sys, time
sys.path.insert(0, '/pkg' if sys.platform == 'emscripten' else 'src')
import numpy as onp
import autograd.numpy as np
from autograd import grad
from divelet_ml import nn

CASES = [('first   1->16  @96x96', 1, 16, 96), ('middle 64->128 @24x24', 64, 128, 24), ('late  128->128 @12x12', 128, 128, 12)]
N = 8
T = time.perf_counter

def timeit(f, n=3):
    f(); t = T()
    for _ in range(n): f()
    return (T() - t) / n * 1e3

print(f'platform {sys.platform}, batch {N}, float32, 3x3 padding 1')
for name, cin, cout, hw in CASES:
    X = onp.random.rand(N, cin, hw, hw).astype('float32'); W = (onp.random.rand(cout, cin, 3, 3).astype('float32') - .5)
    fwd = timeit(lambda: nn.conv2d(X, W, padding=1))
    both = timeit(lambda: grad(lambda w: nn.conv2d(X, w, padding=1).sum())(W))  # forward + backward to the weights
    gmac = N * cout * cin * 9 * hw * hw / 1e9
    print(f'{name}: forward {fwd:8.1f} ms, forward+backward {both:8.1f} ms  ({gmac:.2f} GMAC forward)')

try:  # inside Pyodide: what crossing to JavaScript costs for the middle case's input and output
    import js
    from pyodide.ffi import to_js
    X = onp.random.rand(N, 64, 24, 24).astype('float32'); Y = onp.random.rand(N, 128, 24, 24).astype('float32')
    up = timeit(lambda: to_js(X.ravel()), 10); down = timeit(lambda: js.Float32Array.new(Y.size).to_py(), 10)
    print(f'copy to JS {X.nbytes/1e6:.1f} MB: {up:.2f} ms; copy back {Y.nbytes/1e6:.1f} MB: {down:.2f} ms')
except ImportError:
    pass
