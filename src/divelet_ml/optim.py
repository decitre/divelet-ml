"""NumPy stand-in for the part of PyTorch's `torch.optim` that adapted chapters use, imported as
`from divelet_ml import optim`.

An optimizer holds the parameter arrays in `params` and updates them in place with `step(grads)`, where
`grads` are their gradients in the same order, as `divelet_ml.param_grads` returns them. As in PyTorch, `params`
can be arrays or parameter groups (dicts with `'params'` and options that override the optimizer's), the groups are
in `param_groups`, and changing a group's `'lr'` changes the step of the next `step`.
"""

import collections

import numpy as np


class Optimizer:
    """The base class of optimizers: it keeps the parameter groups and the state of each parameter."""
    def __init__(self, params, **options):
        groups = list(params)
        if not groups or not isinstance(groups[0], dict):
            groups = [{'params': groups}]
        self.param_groups, self.params, self._group_of, self.state = [], [], [], []
        for group in groups:
            arrays = group['params']
            arrays = [arrays] if isinstance(arrays, np.ndarray) else list(arrays)
            merged = {**options, **{k: v for k, v in group.items() if k != 'params'}, 'params': arrays}
            self.param_groups.append(merged)
            self.params += arrays
            self._group_of += [merged] * len(arrays)
            self.state += [{} for _ in arrays]

    @property
    def lr(self):
        return self.param_groups[0]['lr']

    @lr.setter
    def lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr

    def step(self, grads):
        for param, grad, group, state in zip(self.params, grads, self._group_of, self.state):
            if group['weight_decay']:
                grad = grad + group['weight_decay'] * param
            self.update(param, grad, group, state)

    def update(self, param, grad, group, state):
        raise NotImplementedError


class SGD(Optimizer):
    """Minibatch stochastic gradient descent, in place of `torch.optim.SGD`.

    As in PyTorch, weight decay adds `weight_decay * param` to the gradient, and with `momentum` the update uses a
    velocity `v = momentum * v + grad`, which starts as the first gradient. Nesterov momentum and dampening are not
    supported.
    """
    def __init__(self, params, lr, momentum=0, weight_decay=0):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def update(self, param, grad, group, state):
        if group['momentum']:
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = np.array(grad, copy=True)
            else:
                state['momentum_buffer'] *= group['momentum']
                state['momentum_buffer'] += grad
            grad = state['momentum_buffer']
        param -= group['lr'] * grad


class Adagrad(Optimizer):
    """Adagrad, in place of `torch.optim.Adagrad` (without its `lr_decay` and `initial_accumulator_value`)."""
    def __init__(self, params, lr=0.01, weight_decay=0, eps=1e-10):
        super().__init__(params, lr=lr, weight_decay=weight_decay, eps=eps)

    def update(self, param, grad, group, state):
        total = state.setdefault('sum', np.zeros_like(param))
        total += grad ** 2
        param -= group['lr'] * grad / (np.sqrt(total) + group['eps'])


class RMSprop(Optimizer):
    """RMSProp, in place of `torch.optim.RMSprop` (without its `momentum`, `centered` and `maximize`)."""
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)

    def update(self, param, grad, group, state):
        square_avg = state.setdefault('square_avg', np.zeros_like(param))
        square_avg *= group['alpha']
        square_avg += (1 - group['alpha']) * grad ** 2
        param -= group['lr'] * grad / (np.sqrt(square_avg) + group['eps'])


class Adadelta(Optimizer):
    """Adadelta, in place of `torch.optim.Adadelta`."""
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        super().__init__(params, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)

    def update(self, param, grad, group, state):
        square_avg = state.setdefault('square_avg', np.zeros_like(param))
        acc_delta = state.setdefault('acc_delta', np.zeros_like(param))
        rho, eps = group['rho'], group['eps']
        square_avg *= rho
        square_avg += (1 - rho) * grad ** 2
        delta = np.sqrt(acc_delta + eps) / np.sqrt(square_avg + eps) * grad
        acc_delta *= rho
        acc_delta += (1 - rho) * delta ** 2
        param -= group['lr'] * delta


class Adam(Optimizer):
    """Adam, in place of `torch.optim.Adam` (without its `amsgrad`, `maximize` and decoupled weight decay)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def update(self, param, grad, group, state):
        beta1, beta2 = group['betas']
        state['step'] = t = state.get('step', 0) + 1
        m, v = state.setdefault('exp_avg', np.zeros_like(param)), state.setdefault('exp_avg_sq', np.zeros_like(param))
        m *= beta1
        m += (1 - beta1) * grad
        v *= beta2
        v += (1 - beta2) * grad ** 2
        denominator = np.sqrt(v) / np.sqrt(1 - beta2 ** t) + group['eps']
        param -= group['lr'] / (1 - beta1 ** t) * m / denominator


class lr_scheduler:
    """Learning rate schedules, in place of `torch.optim.lr_scheduler`: `from divelet_ml.optim import lr_scheduler`."""
    class MultiStepLR:
        """Multiply the learning rate of every group by `gamma` when `step()` reaches an epoch of `milestones`."""
        def __init__(self, optimizer, milestones, gamma=0.1):
            self.optimizer, self.milestones, self.gamma, self.last_epoch = optimizer, collections.Counter(milestones), gamma, 0

        def get_last_lr(self):
            return [group['lr'] for group in self.optimizer.param_groups]

        def step(self):
            self.last_epoch += 1
            for group in self.optimizer.param_groups:
                group['lr'] *= self.gamma ** self.milestones.get(self.last_epoch, 0)
