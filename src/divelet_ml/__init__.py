"""divelet-ml's NumPy counterpart of the `d2l` library, imported as `import divelet_ml as d2l`.

`add_to_class`, `HyperParameters`, `DataModule`, `check_shape`, `use_svg_display`, `set_figsize`, `set_axes`, `plot`,
`show_heatmaps`, `show_images`, `ProgressBoard`, `FashionMNIST`, `Classifier`, `SoftmaxRegression`, `Module`, `Trainer`, `SyntheticRegressionData`, `SGD`, `LinearRegressionScratch`,
`LinearRegression`, `Vocab`, `TimeMachine`, `RNNScratch`, `RNNLMScratch`, `RNN`, `RNNLM`, `ResNeXtBlock`, `check_len`, `init_cnn`, and the helpers of Chapter 12 (`Timer`, `Animator`, `train_2d`, `train_ch11` and others) are adapted from `d2l/torch.py` in Dive into Deep Learning
(https://github.com/d2l-ai/d2l-en, commit 23d7a5aecceee57d1292c56e90cce307f183bb0a),
Copyright 2019 Amazon.com, Inc. or its affiliates, MIT-0 license (LICENSE-SAMPLECODE).
Only what adapted chapters use is here. The code is in submodules (`utils`, `plotting`, `data`, `training`,
`nn`, `optim`); this module re-exports it, as the names in `__all__`.

Gradients come from autograd, in place of PyTorch's `backward`: `param_grads` differentiates a function with
respect to a model's parameter arrays, and `Trainer` uses it.

`d2l.plt` is matplotlib's `pyplot`, as in the book. It is imported on first use, so notebooks that draw
no plots don't load matplotlib.
"""

__version__ = "0.10.0"

from divelet_ml import nn, optim
from divelet_ml.utils import (add_to_class, HyperParameters, check_shape, check_len, numpy, Timer, Accumulator, linreg, squared_loss,
                              evaluate_loss, accuracy, evaluate_accuracy_gpu, param_grads, browser_epochs,
                              browser_examples, corr2d, cpu, gpu, num_gpus, try_gpu, try_all_gpus)
from divelet_ml.plotting import use_svg_display, set_figsize, set_axes, plot, show_heatmaps, show_images, ProgressBoard, annotate, Animator
from divelet_ml.data import (DataModule, ArrayLoader, ArrayDataset, load_fashion_mnist, download, DATA_URL, SyntheticRegressionData, FashionMNIST,
                             Vocab, TimeMachine, time_machine_text, airfoil_self_noise, load_array, get_data_ch11, load_data_fashion_mnist)
from divelet_ml.training import Module, Trainer, SGD, LinearRegressionScratch, LinearRegression, Classifier, SoftmaxRegression, init_cnn
from divelet_ml.rnn import RNNScratch, RNNLMScratch, RNN, RNNLM
from divelet_ml.cnn import ResNeXtBlock
from divelet_ml.optimization import train_2d, show_trace_2d, train_ch11, train_concise_ch11

__all__ = [
    'nn',
    'optim',
    'add_to_class',
    'HyperParameters',
    'check_shape',
    'check_len',
    'numpy',
    'param_grads',
    'browser_epochs',
    'browser_examples',
    'corr2d',
    'cpu',
    'gpu',
    'num_gpus',
    'try_gpu',
    'try_all_gpus',
    'use_svg_display',
    'set_figsize',
    'set_axes',
    'plot',
    'show_heatmaps',
    'show_images',
    'ProgressBoard',
    'DataModule',
    'ArrayLoader',
    'SyntheticRegressionData',
    'FashionMNIST',
    'Vocab',
    'TimeMachine',
    'time_machine_text',
    'ArrayDataset',
    'load_fashion_mnist',
    'download',
    'DATA_URL',
    'Module',
    'Trainer',
    'SGD',
    'LinearRegressionScratch',
    'LinearRegression',
    'Classifier',
    'SoftmaxRegression',
    'init_cnn',
    'RNNScratch',
    'RNNLMScratch',
    'RNN',
    'RNNLM',
    'ResNeXtBlock',
    'Timer',
    'Accumulator',
    'linreg',
    'squared_loss',
    'evaluate_loss',
    'accuracy',
    'evaluate_accuracy_gpu',
    'annotate',
    'Animator',
    'airfoil_self_noise',
    'load_array',
    'get_data_ch11',
    'load_data_fashion_mnist',
    'train_2d',
    'show_trace_2d',
    'train_ch11',
    'train_concise_ch11',
]


def __getattr__(name):
    if name == 'plt':
        from matplotlib import pyplot as plt
        return plt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
