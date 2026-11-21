"""NumPy stand-ins for the parts of PyTorch's `torch.nn` that adapted chapters use, imported as
`from divelet_ml import nn`.

Forward computation only: there are no devices, and gradients come from autograd through
`divelet_ml.param_grads`, so code that a model computes its loss with must use autograd's NumPy
(`import autograd.numpy as np`) or array operators. A module's parameters are its floating-point array
attributes, except the ones a module lists in its `buffers`, such as the running statistics of batch normalization. `Module.eval()` and `Module.train()` switch layers such as `Dropout` between training and
inference behavior, as in PyTorch. `softmax` follows Flax's `nn.softmax(x, axis=-1)`, the JAX
counterpart of `torch.nn.functional.softmax`.
"""

import math

import autograd.numpy as anp
import numpy as np
from autograd.extend import defvjp, primitive
from autograd.tracer import getval
from numpy.lib.stride_tricks import sliding_window_view


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
    buffers = ()  # names of float arrays that are state, not parameters, such as the running statistics of batch normalization

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def modules(self, _seen=None):
        """This module, then the modules held in its attributes, recursively; a module that is held twice
        comes once, so a shared layer's parameters are not counted twice."""
        seen = set() if _seen is None else _seen
        if id(self) in seen:
            return
        seen.add(id(self))
        yield self
        for value in vars(self).values():
            if isinstance(value, Module):
                yield from value.modules(seen)

    def parameters(self):
        """The floating-point arrays held in the attributes of this module and its submodules, except the ones a module
        lists in its `buffers`."""
        # shortcut: arrays in lists or dicts are not found; walk them when a model holds layers that way.
        return [v for m in self.modules() for k, v in vars(m).items()
                if isinstance(v, np.ndarray) and v.dtype.kind == 'f' and k not in m.buffers]

    def add_module(self, name, module):
        """Hold `module` as the attribute `name`."""
        assert isinstance(module, Module), f'{type(module).__name__} is not a Module'
        setattr(self, name, module)

    def children(self):
        """The modules held in the attributes of this module, each once."""
        seen = set()
        for value in vars(self).values():
            if isinstance(value, Module) and id(value) not in seen:
                seen.add(id(value))
                yield value

    def named_parameters(self, prefix=''):
        """Pairs (name, array) of the parameters, named by the attributes that lead to them, such as
        '0.weight'. An array that several modules share comes once."""
        seen = set()
        for name, value in self._named_arrays(prefix, buffers=False):
            if id(value) not in seen:
                seen.add(id(value))
                yield name, value

    def _named_arrays(self, prefix='', buffers=True):
        for k, v in vars(self).items():
            if isinstance(v, np.ndarray) and v.dtype.kind == 'f':
                if buffers or k not in self.buffers:
                    yield prefix + k, v
            elif isinstance(v, Module):
                yield from v._named_arrays(f'{prefix}{k}.', buffers)

    def state_dict(self):
        """A dictionary of the parameters and buffers by name, as `named_parameters` gives the parameters, but with an
        array that several modules share under each of their names, as in PyTorch."""
        return dict(self._named_arrays())

    def load_state_dict(self, state_dict):
        """Replace each named parameter by a copy of the array in `state_dict`. A lazy layer takes the shape of
        what it loads."""
        for name, array in state_dict.items():
            *path, key = name.split('.')
            module = self
            for part in path:
                module = getattr(module, part)
            setattr(module, key, np.array(array))
        return self

    def apply(self, fn):
        """Call `fn` on each submodule, then on this module, as in PyTorch."""
        for child in self.children():
            child.apply(fn)
        fn(self)
        return self

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


class Linear(Module):
    """A fully connected layer, in place of `torch.nn.Linear`: the weight, of shape (out_features,
    in_features), and the bias are drawn as in PyTorch, uniformly from (-1/sqrt(in_features),
    1/sqrt(in_features)), unless `weight_init` or `bias_init` is given: a function of the shape that returns
    the initial array, like the initializers of Flax's `nn.Dense`. They are `float32` unless `dtype` is given."""
    def __init__(self, in_features, out_features, bias=True, weight_init=None, bias_init=None, dtype=np.float32):
        self.out_features, self.has_bias = out_features, bias
        self.weight_init, self.bias_init = weight_init, bias_init
        self.weight = self.bias = None
        self._materialize(in_features, dtype)

    def _materialize(self, in_features, dtype):
        bound = 1 / math.sqrt(in_features)
        uniform = lambda shape: np.random.uniform(-bound, bound, shape)
        self.weight = np.asarray((self.weight_init or uniform)((self.out_features, in_features)), dtype)
        if self.has_bias:
            self.bias = np.asarray((self.bias_init or uniform)((self.out_features,)), dtype)

    def forward(self, X):
        Y = X @ self.weight.T
        return Y if self.bias is None else Y + self.bias


class LazyLinear(Linear):
    """A `Linear` layer whose number of inputs is set by its first input, in place of `torch.nn.LazyLinear`.
    Its parameters are `None` until then, and it becomes a `Linear` once it has them, as in PyTorch."""
    def __init__(self, out_features, bias=True, weight_init=None, bias_init=None):
        self.out_features, self.has_bias = out_features, bias
        self.weight_init, self.bias_init = weight_init, bias_init
        self.weight = self.bias = None

    def forward(self, X):
        if self.weight is None:
            # The parameters take the input's floating-point type, as PyTorch's float32 does; else float32 data
            # would be multiplied in float64, twice as slow.
            self._materialize(X.shape[-1], X.dtype if X.dtype.kind == 'f' else np.float64)
            self.__class__ = Linear
        return Linear.forward(self, X)


class RNN(Module):
    """A one-layer recurrent network with a tanh nonlinearity, in place of `torch.nn.RNN(input_size, hidden_size)`.

    `forward(inputs, state=None)` takes `inputs` of shape (steps, batch, input_size) and a `state` of shape
    (1, batch, hidden_size), zeros if missing, and returns the outputs, of shape (steps, batch, hidden_size), and the
    last state. The parameters are drawn as in PyTorch, uniformly from (-1/sqrt(hidden_size), 1/sqrt(hidden_size)),
    as `float32` unless `dtype` is given. The loop over the steps is Python, which autograd differentiates."""
    def __init__(self, input_size, hidden_size, dtype=np.float32):
        self.input_size, self.hidden_size = input_size, hidden_size
        bound = 1 / math.sqrt(hidden_size)
        uniform = lambda shape: np.random.uniform(-bound, bound, shape).astype(dtype)
        self.weight_ih, self.weight_hh = uniform((hidden_size, input_size)), uniform((hidden_size, hidden_size))
        self.bias_ih, self.bias_hh = uniform((hidden_size,)), uniform((hidden_size,))

    def forward(self, inputs, state=None):
        h = anp.zeros((inputs.shape[1], self.hidden_size), self.weight_ih.dtype) if state is None else state[0]
        outputs = []
        for X in inputs:
            h = anp.tanh(X @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh)
            outputs.append(h)
        return anp.stack(outputs), anp.expand_dims(h, 0)


def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def _windows(X, kernel, stride):
    """The windows of the last two dimensions of `X` that a kernel of this size visits, as a view of shape
    (n, c, out_height, out_width, kernel_height, kernel_width)."""
    return sliding_window_view(X, kernel, axis=(2, 3))[:, :, ::stride[0], ::stride[1]]


def _scatter(shape, dtype, parts, kernel, stride, out):
    """The sum of `parts[..., i, j]`, of shape (n, c, out_height, out_width), placed back at the positions of
    the window offsets (i, j) in an array of zeros: the adjoint of `_windows`."""
    dX = np.zeros(shape, dtype)
    for i in range(kernel[0]):
        for j in range(kernel[1]):
            dX[:, :, i:i + stride[0] * (out[0] - 1) + 1:stride[0], j:j + stride[1] * (out[1] - 1) + 1:stride[1]] += parts(i, j)
    return dX


def _crop(dXp, padding):
    ph, pw = padding
    return dXp[:, :, ph:dXp.shape[2] - ph, pw:dXp.shape[3] - pw]


@primitive
def _corr2d(X, W, stride, padding):
    """The cross-correlation of the images `X` (n, c, h, w) with the kernels `W` (o, c, kh, kw), with
    differentiable `X` and `W`. Written as a NumPy primitive with its own gradients: tracing the slices of a
    loop through autograd costs several times more."""
    Xp = np.pad(X, ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2))
    windows = _windows(Xp, W.shape[2:], stride)
    return np.tensordot(windows, W, axes=([1, 4, 5], [1, 2, 3])).transpose(0, 3, 1, 2)


def _corr2d_vjp_X(ans, X, W, stride, padding):
    kernel, out = W.shape[2:], ans.shape[2:]

    def vjp(g):
        columns = np.tensordot(g, W, axes=([1], [0]))  # (n, out_height, out_width, c, kh, kw)
        shape = X.shape[:2] + (X.shape[2] + 2 * padding[0], X.shape[3] + 2 * padding[1])
        return _crop(_scatter(shape, g.dtype, lambda i, j: columns[..., i, j].transpose(0, 3, 1, 2), kernel, stride, out), padding)
    return vjp


def _corr2d_vjp_W(ans, X, W, stride, padding):
    def vjp(g):
        Xp = np.pad(X, ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2))
        return np.tensordot(g, _windows(Xp, W.shape[2:], stride), axes=([0, 2, 3], [0, 2, 3]))
    return vjp


defvjp(_corr2d, _corr2d_vjp_X, _corr2d_vjp_W)


def conv2d(X, weight, bias=None, stride=1, padding=0, groups=1):
    """The cross-correlation of images `X` of shape (n, c, h, w) with `weight` of shape (o, c / groups, kh, kw), plus
    `bias`, as `torch.nn.functional.conv2d`, differentiable by autograd. `stride` and `padding` are an integer
    or a pair (height, width). With `groups` > 1, the channels are split into that many groups, each convolved with
    its own share of the kernels, and the outputs are joined."""
    stride, padding = _pair(stride), _pair(padding)
    if groups == 1:
        Y = _corr2d(X, weight, stride, padding)
    else:
        assert X.shape[1] % groups == 0 and weight.shape[0] % groups == 0, \
            f'{X.shape[1]} input and {weight.shape[0]} output channels are not divisible by {groups} groups'
        cin, cout = X.shape[1] // groups, weight.shape[0] // groups
        Y = anp.concatenate([_corr2d(X[:, g * cin:(g + 1) * cin], weight[g * cout:(g + 1) * cout], stride, padding)
                             for g in range(groups)], axis=1)
    return Y if bias is None else Y + bias.reshape((1, -1, 1, 1))


def _tiles(X, kernel):
    """`X` cropped to whole windows and viewed as (n, c, out_height, kh, out_width, kw): the windows of a
    pooling whose stride is its window size, which is the usual case and needs no copy."""
    n, c, h, w = X.shape
    oh, ow = h // kernel[0], w // kernel[1]
    return X[:, :, :oh * kernel[0], :ow * kernel[1]].reshape(n, c, oh, kernel[0], ow, kernel[1])


@primitive
def _pool2d(X, kernel, stride, padding, mode):
    if mode == 'avg' and stride == kernel and padding == (0, 0):
        return _tiles(X, kernel).mean(axis=(3, 5))
    Xp = np.pad(X, ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2), constant_values=-np.inf if mode == 'max' else 0)
    windows = _windows(Xp, kernel, stride)
    if mode == 'max':
        return windows.max(axis=(4, 5))
    return windows.mean(axis=(4, 5))


def _pool2d_vjp(ans, X, kernel, stride, padding, mode):
    def vjp(g):
        if mode == 'avg' and stride == kernel and padding == (0, 0):
            dX = np.zeros(X.shape, g.dtype)
            _tiles(dX, kernel)[...] = g[:, :, :, None, :, None] / (kernel[0] * kernel[1])
            return dX
        shape = X.shape[:2] + (X.shape[2] + 2 * padding[0], X.shape[3] + 2 * padding[1])
        if mode == 'avg':
            parts = lambda i, j: g / (kernel[0] * kernel[1])
        else:
            Xp = np.pad(X, ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2), constant_values=-np.inf)
            windows = _windows(Xp, kernel, stride)
            first = windows.reshape(windows.shape[:4] + (-1,)).argmax(axis=-1)  # the first maximum, as PyTorch
            parts = lambda i, j: g * (first == i * kernel[1] + j)
        return _crop(_scatter(shape, g.dtype, parts, kernel, stride, ans.shape[2:]), padding)
    return vjp


defvjp(_pool2d, _pool2d_vjp)


def max_pool2d(X, kernel_size, stride=None, padding=0):
    """The maximum over the windows of `kernel_size` of the last two dimensions of `X`, as
    `torch.nn.functional.max_pool2d`; `stride` is the window size by default. Padding counts as minus infinity."""
    kernel = _pair(kernel_size)
    return _pool2d(X, kernel, _pair(kernel_size if stride is None else stride), _pair(padding), 'max')


def avg_pool2d(X, kernel_size, stride=None, padding=0):
    """The mean over the windows of `kernel_size` of the last two dimensions of `X`, as
    `torch.nn.functional.avg_pool2d`; `stride` is the window size by default. Padding counts as zeros."""
    return _pool2d(X, _pair(kernel_size), _pair(kernel_size if stride is None else stride), _pair(padding), 'avg')


class Conv2d(Module):
    """A two-dimensional convolutional layer (a cross-correlation, as in the book), in place of
    `torch.nn.Conv2d`: the weight, of shape (out_channels, in_channels, kernel_height, kernel_width), and the
    bias are drawn as in PyTorch, uniformly from (-1/sqrt(f), 1/sqrt(f)) with f = in_channels * kernel_height
    * kernel_width, unless `weight_init` or `bias_init` is given. They are `float32` unless `dtype` is given. With `groups`
    > 1, each kernel sees in_channels / groups channels, and f counts those."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True,
                 weight_init=None, bias_init=None, dtype=np.float32, groups=1):
        self.out_channels, self.kernel_size, self.groups = out_channels, _pair(kernel_size), groups
        self.padding, self.stride, self.has_bias = _pair(padding), _pair(stride), bias
        self.weight_init, self.bias_init = weight_init, bias_init
        self.weight = self.bias = None
        self._materialize(in_channels, dtype)

    def _materialize(self, in_channels, dtype):
        assert in_channels % self.groups == 0 and self.out_channels % self.groups == 0, \
            f'{in_channels} input and {self.out_channels} output channels are not divisible by {self.groups} groups'
        in_channels //= self.groups
        bound = 1 / math.sqrt(in_channels * self.kernel_size[0] * self.kernel_size[1])
        uniform = lambda shape: np.random.uniform(-bound, bound, shape)
        self.weight = np.asarray((self.weight_init or uniform)((self.out_channels, in_channels) + self.kernel_size), dtype)
        if self.has_bias:
            self.bias = np.asarray((self.bias_init or uniform)((self.out_channels,)), dtype)

    def forward(self, X):
        return conv2d(X, self.weight, self.bias, self.stride, self.padding, self.groups)


class LazyConv2d(Conv2d):
    """A `Conv2d` layer whose number of input channels is set by its first input, in place of
    `torch.nn.LazyConv2d`. It becomes a `Conv2d` once it has its parameters, as in PyTorch."""
    def __init__(self, out_channels, kernel_size, padding=0, stride=1, bias=True, weight_init=None, bias_init=None, groups=1):
        self.out_channels, self.kernel_size, self.groups = out_channels, _pair(kernel_size), groups
        self.padding, self.stride, self.has_bias = _pair(padding), _pair(stride), bias
        self.weight_init, self.bias_init = weight_init, bias_init
        self.weight = self.bias = None

    def forward(self, X):
        if self.weight is None:
            self._materialize(X.shape[1], X.dtype if X.dtype.kind == 'f' else np.float64)
            self.__class__ = Conv2d
        return Conv2d.forward(self, X)


class MaxPool2d(Module):
    """Maximum pooling, in place of `torch.nn.MaxPool2d`."""
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, X):
        return max_pool2d(X, self.kernel_size, self.stride, self.padding)


class AvgPool2d(MaxPool2d):
    """Average pooling, in place of `torch.nn.AvgPool2d`."""
    def forward(self, X):
        return avg_pool2d(X, self.kernel_size, self.stride, self.padding)


class AdaptiveAvgPool2d(Module):
    """The average of each channel over the whole image, in place of `torch.nn.AdaptiveAvgPool2d((1, 1))`, which is the
    only output size the book uses."""
    def __init__(self, output_size):
        # shortcut: only a 1 x 1 output, the global average pooling of the book; add other sizes when a section needs them.
        assert _pair(output_size) == (1, 1), f'output size {output_size} is not 1 x 1'

    def forward(self, X):
        return anp.mean(X, axis=(2, 3), keepdims=True)


class BatchNorm1d(Module):
    """Batch normalization of the channels (axis 1) of inputs of shape (n, c) or (n, c, l), in place of
    `torch.nn.BatchNorm1d`. In training mode, each channel is normalized with the mean and variance of the batch, and the
    running statistics move toward them by `momentum` (the variance with the unbiased estimate, as in PyTorch); in
    evaluation mode, the running statistics are used. The scale `weight` and shift `bias` are parameters; the running
    statistics are buffers, so they are not parameters but are in `state_dict()`. They are `float32` unless `dtype`
    is given."""
    buffers = ('running_mean', 'running_var')
    ndims = (2, 3)

    def __init__(self, num_features, eps=1e-5, momentum=0.1, dtype=np.float32):
        self.eps, self.momentum = eps, momentum
        self._materialize(num_features, dtype)

    def _materialize(self, num_features, dtype):
        self.weight, self.bias = np.ones(num_features, dtype), np.zeros(num_features, dtype)
        self.running_mean, self.running_var = np.zeros(num_features, dtype), np.ones(num_features, dtype)

    def forward(self, X):
        assert X.ndim in self.ndims, f'{type(self).__name__} takes inputs of {self.ndims} dimensions, not {X.ndim}'
        axes, shape = tuple(i for i in range(X.ndim) if i != 1), (1, -1) + (1,) * (X.ndim - 2)
        if self.training:
            mean = anp.mean(X, axis=axes, keepdims=True)
            var = anp.mean((X - mean) ** 2, axis=axes, keepdims=True)
            n, m = X.size // X.shape[1], self.momentum
            self.running_mean = ((1 - m) * self.running_mean + m * getval(mean).reshape(-1)).astype(self.running_mean.dtype)
            self.running_var = ((1 - m) * self.running_var + m * getval(var).reshape(-1) * n / max(n - 1, 1)).astype(self.running_var.dtype)
        else:
            mean, var = self.running_mean.reshape(shape), self.running_var.reshape(shape)
        return (X - mean) / anp.sqrt(var + self.eps) * self.weight.reshape(shape) + self.bias.reshape(shape)


class BatchNorm2d(BatchNorm1d):
    """Batch normalization of the channels of images of shape (n, c, h, w), in place of `torch.nn.BatchNorm2d`."""
    ndims = (4,)


class LazyBatchNorm1d(BatchNorm1d):
    """A `BatchNorm1d` whose number of features is set by its first input, in place of `torch.nn.LazyBatchNorm1d`. It becomes
    a `BatchNorm1d` once it has its parameters, as in PyTorch."""
    eager = BatchNorm1d

    def __init__(self, eps=1e-5, momentum=0.1):
        self.eps, self.momentum = eps, momentum
        self.weight = self.bias = self.running_mean = self.running_var = None

    def forward(self, X):
        if self.weight is None:
            self._materialize(X.shape[1], X.dtype if X.dtype.kind == 'f' else np.float64)
            self.__class__ = self.eager
        return self.forward(X)


class LazyBatchNorm2d(LazyBatchNorm1d, BatchNorm2d):
    """A `BatchNorm2d` whose number of features is set by its first input, in place of `torch.nn.LazyBatchNorm2d`."""
    eager = BatchNorm2d


def sigmoid(X):
    """The sigmoid function, in place of `torch.sigmoid`."""
    return 1 / (1 + anp.exp(-X))


class Sigmoid(Module):
    """The sigmoid activation, in place of `torch.nn.Sigmoid`."""
    def forward(self, X):
        return sigmoid(X)


class MSELoss(Module):
    """The squared differences between `input` and `target`, averaged (`reduction='mean'`), summed (`'sum'`) or kept
    one by one (`'none'`), in place of `torch.nn.MSELoss`."""
    def __init__(self, reduction='mean'):
        assert reduction in ('mean', 'sum', 'none'), f'unknown reduction {reduction!r}'
        self.reduction = reduction

    def forward(self, input, target):
        l = (input - target) ** 2
        return {'mean': anp.mean, 'sum': anp.sum, 'none': lambda x: x}[self.reduction](l)


def relu(X):
    """The rectified linear unit, in place of `torch.nn.functional.relu`; Flax's `nn.relu`."""
    return anp.maximum(X, 0)


class ReLU(Module):
    """The rectified linear unit, in place of `torch.nn.ReLU`."""
    def forward(self, X):
        return relu(X)


class Flatten(Module):
    """Flatten all dimensions but the first, in place of `torch.nn.Flatten`."""
    def forward(self, X):
        return X.reshape((X.shape[0], -1))


class Sequential(Module):
    """Apply the given modules one after the other, in place of `torch.nn.Sequential`. The modules are
    attributes named '0', '1', ..., as in PyTorch, so `parameters()` finds their arrays. `add_module(name, module)` adds
    a module at the end, under `name`."""
    def __init__(self, *layers):
        self.names = [str(i) for i in range(len(layers))]
        for name, layer in zip(self.names, layers):
            setattr(self, name, layer)

    @property
    def num_layers(self):
        return len(self.names)

    def add_module(self, name, module):
        assert name not in self.names, f'{name} is already a module of this Sequential'
        super().add_module(name, module)
        self.names.append(name)

    def __getitem__(self, i):
        return getattr(self, self.names[i])

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return (getattr(self, name) for name in self.names)

    def forward(self, X):
        for layer in self:
            X = layer(X)
        return X


class init:
    """Initializers that fill an array in place and return it, in place of `torch.nn.init`."""
    @staticmethod
    def normal_(a, mean=0.0, std=1.0):
        a[...] = np.random.normal(mean, std, a.shape)
        return a

    @staticmethod
    def uniform_(a, low=0.0, high=1.0):
        a[...] = np.random.uniform(low, high, a.shape)
        return a

    @staticmethod
    def constant_(a, val):
        a[...] = val
        return a

    @staticmethod
    def zeros_(a):
        return init.constant_(a, 0)

    @staticmethod
    def ones_(a):
        return init.constant_(a, 1)

    @staticmethod
    def xavier_uniform_(a, gain=1.0):
        """Uniform on (-b, b) with b = gain * sqrt(6 / (fan_in + fan_out)); an array of two dimensions is a
        weight (out, in), and a kernel (out, in, height, width) has the window size in both fans."""
        assert a.ndim >= 2, 'xavier_uniform_ needs an array of at least two dimensions'
        window = int(np.prod(a.shape[2:]))
        bound = gain * math.sqrt(6 / ((a.shape[0] + a.shape[1]) * window))
        return init.uniform_(a, -bound, bound)
