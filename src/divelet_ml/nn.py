"""NumPy stand-ins for the parts of PyTorch's `torch.nn` that adapted chapters use, imported as
`from divelet_ml import nn`.

Forward computation only: there are no gradients, parameter registries or devices. `Module.eval()`
and `Module.train()` switch layers such as `Dropout` between training and inference behavior, as in
PyTorch. `softmax` follows Flax's `nn.softmax(x, axis=-1)`, the JAX counterpart of
`torch.nn.functional.softmax`.
"""

import math

import numpy as np


def softmax(X, axis=-1):
    """Softmax along `axis`, shifted by the maximum so `exp` does not overflow."""
    X_exp = np.exp(X - X.max(axis=axis, keepdims=True))
    return X_exp / X_exp.sum(axis=axis, keepdims=True)


class Module:
    """The base class of layers and models, in place of `torch.nn.Module`.

    Calling a module calls its `forward` method. A module is in training mode until `eval()` is called;
    `train()` and `eval()` also set the mode of the modules held in its attributes.
    """
    training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        for value in vars(self).values():
            if isinstance(value, Module):
                value.train(mode)
        return self

    def eval(self):
        return self.train(False)


class Dropout(Module):
    """In training mode, zero each element with probability `p` and scale the others by 1 / (1 - p).
    In evaluation mode, the identity. In place of `torch.nn.Dropout`."""
    def __init__(self, p=0.5):
        assert 0 <= p <= 1, f'dropout probability {p} is not between 0 and 1'
        self.p = p

    def forward(self, X):
        if not self.training or self.p == 0:
            return X
        if self.p == 1:
            return np.zeros_like(X)
        return X * (np.random.rand(*X.shape) >= self.p) / (1 - self.p)


class LazyLinear(Module):
    """A fully connected layer whose number of inputs is set by its first input, in place of
    `torch.nn.LazyLinear`.

    The weight, of shape (out_features, in_features), and the bias are drawn as in PyTorch, uniformly
    from (-1/sqrt(in_features), 1/sqrt(in_features)).
    """
    def __init__(self, out_features, bias=True):
        self.out_features, self.has_bias = out_features, bias
        self.weight = self.bias = None

    def forward(self, X):
        if self.weight is None:
            in_features = X.shape[-1]
            bound = 1 / math.sqrt(in_features)
            self.weight = np.random.uniform(-bound, bound, (self.out_features, in_features))
            if self.has_bias:
                self.bias = np.random.uniform(-bound, bound, self.out_features)
        Y = X @ self.weight.T
        return Y if self.bias is None else Y + self.bias
