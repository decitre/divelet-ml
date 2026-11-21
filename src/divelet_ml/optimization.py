"""The helpers of Chapter 12 that sections save for each other: the two-variable trainers and plots, and the training loops of
the minibatch sections.

Adapted from `d2l/torch.py` in Dive into Deep Learning (https://github.com/d2l-ai/d2l-en, commit
23d7a5aecceee57d1292c56e90cce307f183bb0a), Copyright 2019 Amazon.com, Inc. or its affiliates, MIT-0 license.
Gradients come from autograd, so the optimizer functions take them as an argument, in place of reading `p.grad`.
"""

import autograd
import autograd.numpy as anp
import numpy as np

from divelet_ml import nn
from divelet_ml.plotting import Animator, _pyplot, set_figsize
from divelet_ml.utils import Timer, evaluate_loss, linreg, param_grads, squared_loss


def train_2d(trainer, steps=20, f_grad=None):
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results


def show_trace_2d(f, results):
    """Show the trace of 2D variables during optimization."""
    set_figsize()
    plt = _pyplot()
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1), indexing='ij')
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')


def train_ch11(trainer_fn, states, hyperparams, data_iter, feature_dim, num_epochs=2):
    """Train a linear regression with `trainer_fn(params, grads, states, hyperparams)`, which updates `params` in place."""
    # Initialization
    w = np.random.normal(0.0, 0.01, (feature_dim, 1)).astype(np.float32)
    b = np.zeros(1, np.float32)
    net, loss = lambda X: linreg(X, w, b), squared_loss
    gradient = autograd.grad(lambda params, X, y: anp.mean(loss(linreg(X, params[0], params[1]), y)))
    # Train
    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            trainer_fn([w, b], gradient([w, b], X, y), states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n / X.shape[0] / len(data_iter), (evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum() / num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]


def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    """Train a linear regression with the optimizer `trainer_fn(net.parameters(), **hyperparams)`."""
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))

    def init_weights(module):
        if type(module) == nn.Linear:
            nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            def objective():
                out = net(X)
                return anp.mean(loss(out, y.reshape(out.shape)))
            optimizer.step(param_grads(objective, net, optimizer.params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` computes squared error without the 1/2 factor
                animator.add(n / X.shape[0] / len(data_iter), (evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum() / num_epochs:.3f} sec/epoch')
