"""NumPy stand-ins for the parts of PyTorch's `torch.nn` that adapted chapters use, imported as
`from divelet_ml import nn`.

Forward computation only: there are no devices, and gradients come from autograd through
`divelet_ml.param_grads`, so code that a model computes its loss with must use autograd's NumPy
(`import autograd.numpy as np`) or array operators. A module's parameters are its floating-point array
attributes. `Module.eval()` and `Module.train()` switch layers such as `Dropout` between training and
inference behavior, as in PyTorch. `softmax` follows Flax's `nn.softmax(x, axis=-1)`, the JAX
counterpart of `torch.nn.functional.softmax`.
"""

import math

import autograd.numpy as anp
import numpy as np
from autograd.tracer import getval


def softmax(X, axis=-1):
    """Softmax along `axis`, shifted by the maximum so `exp` does not overflow."""
    X_exp = np.exp(X - X.max(axis=axis, keepdims=True))
    return X_exp / X_exp.sum(axis=axis, keepdims=True)


def cross_entropy(input, target, reduction='mean'):
    """Cross entropy of the logits `input`, of shape (n, classes), with the integer labels `target`: the
    log-sum-exp is shifted by the maximum, so it neither overflows nor underflows. In place of
    `torch.nn.functional.cross_entropy`; differentiable by autograd."""
    shift = np.max(getval(input), axis=1, keepdims=True)
    lse = anp.log(anp.sum(anp.exp(input - shift), axis=1)) + shift[:, 0]
    l = lse - input[np.arange(len(target)), target]
    return anp.mean(l) if reduction == 'mean' else l


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

    def modules(self):
        """This module, then the modules held in its attributes, recursively."""
        yield self
        for value in vars(self).values():
            if isinstance(value, Module):
                yield from value.modules()

    def parameters(self):
        """The floating-point arrays held in the attributes of this module and its submodules."""
        # shortcut: arrays in lists or dicts are not found; walk them when a model holds layers that way.
        return [v for m in self.modules() for v in vars(m).values()
                if isinstance(v, np.ndarray) and v.dtype.kind == 'f']

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
    from (-1/sqrt(in_features), 1/sqrt(in_features)), unless `weight_init` or `bias_init` is given: a
    function of the shape that returns the initial array, like the initializers of Flax's `nn.Dense`.
    """
    def __init__(self, out_features, bias=True, weight_init=None, bias_init=None):
        self.out_features, self.has_bias = out_features, bias
        self.weight_init, self.bias_init = weight_init, bias_init
        self.weight = self.bias = None

    def forward(self, X):
        if self.weight is None:
            in_features = X.shape[-1]
            bound = 1 / math.sqrt(in_features)
            uniform = lambda shape: np.random.uniform(-bound, bound, shape)
            # The parameters take the input's floating-point type, as PyTorch's float32 does; else float32 data
            # would be multiplied in float64, twice as slow.
            dtype = X.dtype if X.dtype.kind == 'f' else np.float64
            self.weight = np.asarray((self.weight_init or uniform)((self.out_features, in_features)), dtype)
            if self.has_bias:
                self.bias = np.asarray((self.bias_init or uniform)((self.out_features,)), dtype)
        Y = X @ self.weight.T
        return Y if self.bias is None else Y + self.bias


class MSELoss(Module):
    """The mean of the squared differences between `input` and `target`, in place of `torch.nn.MSELoss`."""
    def forward(self, input, target):
        return anp.mean((input - target) ** 2)


class ReLU(Module):
    """The rectified linear unit, in place of `torch.nn.ReLU`."""
    def forward(self, X):
        return anp.maximum(X, 0)


class Flatten(Module):
    """Flatten all dimensions but the first, in place of `torch.nn.Flatten`."""
    def forward(self, X):
        return X.reshape((X.shape[0], -1))


class Sequential(Module):
    """Apply the given modules one after the other, in place of `torch.nn.Sequential`. The modules are
    attributes named '0', '1', ..., as in PyTorch, so `parameters()` finds their arrays."""
    def __init__(self, *layers):
        self.num_layers = len(layers)
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)

    def forward(self, X):
        for i in range(self.num_layers):
            X = getattr(self, str(i))(X)
        return X
