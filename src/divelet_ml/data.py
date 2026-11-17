"""Data modules and a NumPy data loader.

`DataModule` and `SyntheticRegressionData` are adapted from `d2l/torch.py` in Dive into Deep Learning
(https://github.com/d2l-ai/d2l-en, commit 23d7a5aecceee57d1292c56e90cce307f183bb0a),
Copyright 2019 Amazon.com, Inc. or its affiliates, MIT-0 license.
`ArrayLoader` stands in for PyTorch's `DataLoader`.
"""

import numpy as np

from divelet_ml.utils import HyperParameters


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

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        return ArrayLoader(tensors, self.batch_size, shuffle=train)


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


class SyntheticRegressionData(DataModule):
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = np.random.normal(0, 1, (n, len(w)))
        noise = np.random.normal(0, 1, (n, 1)) * noise
        self.y = self.X @ w.reshape((-1, 1)) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
