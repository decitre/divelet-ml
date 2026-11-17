"""Models, the trainer and the optimizer of Chapter 3.

Adapted from `d2l/torch.py` in Dive into Deep Learning (https://github.com/d2l-ai/d2l-en, commit
23d7a5aecceee57d1292c56e90cce307f183bb0a), Copyright 2019 Amazon.com, Inc. or its affiliates, MIT-0 license.
"""

import autograd.numpy as anp
import numpy as np

from divelet_ml import nn, optim
from divelet_ml.plotting import ProgressBoard
from divelet_ml.utils import HyperParameters, numpy, param_grads


class Module(nn.Module, HyperParameters):
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, numpy(value),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError


class Trainer(HyperParameters):
    """The base class for training models with data.

    The optimizer that the model's `configure_optimizers` returns holds the parameter arrays in `params` and
    updates them with `step(grads)`, as `SGD` and `optim.SGD` do.
    """
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        # One forward pass creates the parameters of lazy layers before the optimizer collects them,
        # as the book's JAX edition initializes its model with a first batch.
        model(*next(iter(self.train_dataloader))[:-1])
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def prepare_batch(self, batch):
        return batch

    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            grads = param_grads(lambda: self.model.training_step(self.prepare_batch(batch)),
                                self.model, self.optim.params)
            if self.gradient_clip_val > 0:  # To be discussed later
                grads = self.clip_gradients(self.gradient_clip_val, grads)
            self.optim.step(grads)
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1


class SGD(HyperParameters):
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self, grads):
        for param, grad in zip(self.params, grads):
            param -= self.lr * grad


class LinearRegressionScratch(Module):
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = np.random.normal(0, sigma, (num_inputs, 1))
        self.b = np.zeros(1)

    def forward(self, X):
        return X @ self.w + self.b

    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 / 2
        return anp.mean(l)

    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)


class LinearRegression(Module):
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1, weight_init=lambda shape: np.random.normal(0, 0.01, shape),
                                 bias_init=np.zeros)

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), self.lr)

    def get_w_b(self):
        return (self.net.weight, self.net.bias)
