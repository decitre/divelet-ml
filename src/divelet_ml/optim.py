"""NumPy stand-in for the part of PyTorch's `torch.optim` that adapted chapters use, imported as
`from divelet_ml import optim`.

An optimizer holds the parameter arrays in `params` and updates them in place with `step(grads)`, where
`grads` are their gradients in the same order, as `divelet_ml.param_grads` returns them.
"""

import numpy as np


class SGD:
    """Minibatch stochastic gradient descent, in place of `torch.optim.SGD`.

    `params` is a list of arrays, or of parameter groups: dicts with `'params'` (an array or a list of
    arrays) and optionally `'weight_decay'`, which overrides the `weight_decay` argument for that group.
    As in PyTorch, weight decay adds `weight_decay * param` to the gradient.
    """
    def __init__(self, params, lr, weight_decay=0):
        groups = list(params)
        if not groups or not isinstance(groups[0], dict):
            groups = [{'params': groups}]
        self.lr, self.params, self.weight_decays = lr, [], []
        for group in groups:
            arrays = group['params']
            arrays = [arrays] if isinstance(arrays, np.ndarray) else list(arrays)
            self.params += arrays
            self.weight_decays += [group.get('weight_decay', weight_decay)] * len(arrays)

    def step(self, grads):
        for param, grad, weight_decay in zip(self.params, grads, self.weight_decays):
            param -= self.lr * (grad + weight_decay * param)
