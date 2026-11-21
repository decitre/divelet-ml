"""Helpers for classes and arrays, and gradients with autograd.

`add_to_class`, `HyperParameters` and `check_shape` are adapted from `d2l/torch.py` in Dive into Deep
Learning (https://github.com/d2l-ai/d2l-en, commit 23d7a5aecceee57d1292c56e90cce307f183bb0a),
Copyright 2019 Amazon.com, Inc. or its affiliates, MIT-0 license.
`numpy` and `param_grads` stand in for the book's `d2l.numpy` and PyTorch's `backward`.
"""

import inspect
import time
import sys

import autograd
import autograd.numpy as anp
import numpy as np
from autograd.tracer import getval


def add_to_class(Class):
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


class HyperParameters:
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes."""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


def check_shape(a, shape):
    """Check the shape of an array."""
    assert a.shape == shape, f'array\'s shape {a.shape} != expected shape {shape}'


def check_len(a, n):
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'


def numpy(x):
    """`x` as a NumPy array, also inside a function autograd differentiates, where `x` may be autograd's
    tracing box. In place of the book's `d2l.numpy`."""
    return np.asarray(getval(x))


def param_grads(fn, module, params):
    """The gradients of `fn()` with respect to `params`, arrays held as attributes by `module` or its
    submodules, in the order of `params`. In place of PyTorch's `backward`.

    While autograd traces `fn`, the attributes hold autograd's tracing boxes instead of the arrays; they hold
    the same arrays again afterwards.
    """
    slots = []
    for param in params:
        slot = next(((m, k) for m in module.modules() for k, v in vars(m).items() if v is param), None)
        assert slot is not None, f'parameter of shape {param.shape} is not an attribute of the module'
        slots.append(slot)

    def f(values):
        for (m, k), value in zip(slots, values):
            setattr(m, k, value)
        return fn()

    try:
        return autograd.grad(f)(list(params))
    finally:
        for (m, k), param in zip(slots, params):
            setattr(m, k, param)


def browser_epochs(epochs, pyodide=1, xeus=3):
    """`epochs` outside the browser; in JupyterLite, `pyodide` epochs in the Pyodide kernel, whose NumPy has no
    BLAS and multiplies matrices about 8 times slower, and `xeus` in xeus-python. Training the book's multilayer
    perceptrons for `epochs` epochs takes minutes there. Not in the book."""
    if sys.platform != 'emscripten':
        return epochs
    try:
        import pyjs  # xeus-python's bridge to JavaScript; Pyodide has none. (Both have a `pyodide` entry in sys.modules.)
    except ImportError:
        return min(epochs, pyodide)
    return min(epochs, xeus)


def browser_examples(examples, pyodide, xeus):
    """`examples` training examples outside the browser; in JupyterLite, `pyodide` in the Pyodide kernel and `xeus` in
    xeus-python, as `browser_epochs` does for epochs. Convolutional networks take minutes per epoch of the whole
    training set there. Not in the book."""
    return browser_epochs(examples, pyodide, xeus)


def corr2d(X, K):
    """The two-dimensional cross-correlation of the array `X` with the kernel `K`, as in the book's section 7.2: a
    plain loop over the output, for teaching. `nn.conv2d` is the fast, differentiable version."""
    h, w = K.shape
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


def cpu():
    """The CPU device. A browser kernel has no other: devices are the names 'cpu' and 'cuda:i' only, as in
    the book's `d2l.cpu()` and `d2l.gpu()`, which are PyTorch devices there."""
    return 'cpu'


def gpu(i=0):
    """The name of GPU `i`, which a browser kernel does not have."""
    return f'cuda:{i}'


def num_gpus():
    """The number of GPUs a kernel can use: none in a browser."""
    return 0


def try_gpu(i=0):
    """`gpu(i)` if it exists, otherwise `cpu()`."""
    return gpu(i) if num_gpus() >= i + 1 else cpu()


def try_all_gpus():
    """All available GPUs, or `[cpu()]` if there is none."""
    return [gpu(i) for i in range(num_gpus())] or [cpu()]


class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def linreg(X, w, b):
    """The linear regression model."""
    return anp.matmul(X, w) + b


def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - anp.reshape(y, y_hat.shape)) ** 2 / 2


def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset."""
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        l = loss(out, anp.reshape(y, out.shape))
        metric.add(anp.sum(l), l.size)
    return metric[0] / metric[1]


def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    return float((y_hat.astype(y.dtype) == y).sum())


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy of a model on a dataset. In evaluation mode; `device` is ignored, as a browser kernel has no GPU: the
    name is the book's."""
    if hasattr(net, 'eval'):
        net.eval()
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]
