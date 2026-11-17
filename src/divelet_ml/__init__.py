"""divelet-ml's NumPy counterpart of the `d2l` library, imported as `import divelet_ml as d2l`.

`add_to_class`, `HyperParameters`, `DataModule`, `check_shape`, `use_svg_display`, `set_figsize`, `set_axes`, `plot`,
`show_heatmaps`, `ProgressBoard`, `Module`, `Trainer`, `SyntheticRegressionData`, `SGD`, `LinearRegressionScratch`
and `LinearRegression` are adapted from `d2l/torch.py` in Dive into Deep Learning
(https://github.com/d2l-ai/d2l-en, commit 23d7a5aecceee57d1292c56e90cce307f183bb0a),
Copyright 2019 Amazon.com, Inc. or its affiliates, MIT-0 license (LICENSE-SAMPLECODE).
Only what adapted chapters use is here. The code is in submodules (`utils`, `plotting`, `data`, `training`,
`nn`, `optim`); this module re-exports it, as the names in `__all__`.

Gradients come from autograd, in place of PyTorch's `backward`: `param_grads` differentiates a function with
respect to a model's parameter arrays, and `Trainer` uses it.

`d2l.plt` is matplotlib's `pyplot`, as in the book. It is imported on first use, so notebooks that draw
no plots don't load matplotlib.
"""

__version__ = "0.3.0"

from divelet_ml import nn, optim
from divelet_ml.utils import add_to_class, HyperParameters, check_shape, numpy, param_grads
from divelet_ml.plotting import use_svg_display, set_figsize, set_axes, plot, show_heatmaps, ProgressBoard
from divelet_ml.data import DataModule, ArrayLoader, SyntheticRegressionData
from divelet_ml.training import Module, Trainer, SGD, LinearRegressionScratch, LinearRegression

__all__ = [
    'nn',
    'optim',
    'add_to_class',
    'HyperParameters',
    'check_shape',
    'numpy',
    'param_grads',
    'use_svg_display',
    'set_figsize',
    'set_axes',
    'plot',
    'show_heatmaps',
    'ProgressBoard',
    'DataModule',
    'ArrayLoader',
    'SyntheticRegressionData',
    'Module',
    'Trainer',
    'SGD',
    'LinearRegressionScratch',
    'LinearRegression',
]


def __getattr__(name):
    if name == 'plt':
        from matplotlib import pyplot as plt
        return plt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
