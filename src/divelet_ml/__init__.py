"""divelet-ml's NumPy counterpart of the `d2l` library, imported as `import divelet_ml as d2l`.

`add_to_class`, `HyperParameters`, `DataModule`, `check_shape`, `use_svg_display`, `set_figsize`, `set_axes`, `plot`
and `show_heatmaps` are adapted from `d2l/torch.py` in Dive into Deep Learning
(https://github.com/d2l-ai/d2l-en, commit 23d7a5aecceee57d1292c56e90cce307f183bb0a),
Copyright 2019 Amazon.com, Inc. or its affiliates, MIT-0 license (LICENSE-SAMPLECODE).
Only what adapted chapters use is here.

`d2l.plt` is matplotlib's `pyplot`, as in the book. It is imported on first use, so notebooks that draw
no plots don't load matplotlib.
"""

__version__ = "0.1.0"

import inspect

import numpy as np


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


class DataModule(HyperParameters):
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


class ArrayLoader:
    """Minibatches of NumPy arrays, in place of a framework's data loader such as PyTorch's `DataLoader`.

    Iterating yields tuples of `batch_size` rows of each array. The last batch is smaller when
    `batch_size` doesn't divide the number of rows, or left out with `drop_last=True`.
    With `shuffle=True`, the rows are in a new random order on every pass.
    """
    def __init__(self, arrays, batch_size, shuffle=False, drop_last=False):
        assert len({len(a) for a in arrays}) == 1, 'arrays must have the same number of rows'
        self.arrays, self.batch_size, self.shuffle, self.drop_last = arrays, batch_size, shuffle, drop_last

    def __len__(self):
        n = len(self.arrays[0])
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.arrays[0])
        order = np.random.permutation(n) if self.shuffle else np.arange(n)
        for i in range(0, len(self) * self.batch_size, self.batch_size):
            batch = order[i:i + self.batch_size]
            yield tuple(a[batch] for a in self.arrays)


def __getattr__(name):
    if name == 'plt':
        from matplotlib import pyplot as plt
        return plt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def check_shape(a, shape):
    """Check the shape of an array."""
    assert a.shape == shape, f'array\'s shape {a.shape} != expected shape {shape}'


def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    from matplotlib_inline import backend_inline
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    use_svg_display()
    __getattr__('plt').rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if X (array or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = __getattr__('plt').gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    """Show heatmaps of matrices, an array of shape (rows, columns, height, width)."""
    plt = __getattr__('plt')
    use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix, cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
