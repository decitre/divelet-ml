"""Helpers for classes and arrays, and gradients with autograd.

`add_to_class`, `HyperParameters` and `check_shape` are adapted from `d2l/torch.py` in Dive into Deep
Learning (https://github.com/d2l-ai/d2l-en, commit 23d7a5aecceee57d1292c56e90cce307f183bb0a),
Copyright 2019 Amazon.com, Inc. or its affiliates, MIT-0 license.
`numpy` and `param_grads` stand in for the book's `d2l.numpy` and PyTorch's `backward`.
"""

import inspect

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
