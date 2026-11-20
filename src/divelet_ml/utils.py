"""Helpers for classes and arrays, and gradients with autograd.

`add_to_class`, `HyperParameters` and `check_shape` are adapted from `d2l/torch.py` in Dive into Deep
Learning (https://github.com/d2l-ai/d2l-en, commit 23d7a5aecceee57d1292c56e90cce307f183bb0a),
Copyright 2019 Amazon.com, Inc. or its affiliates, MIT-0 license.
`numpy` and `param_grads` stand in for the book's `d2l.numpy` and PyTorch's `backward`.
"""

import inspect
import sys

import autograd
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
