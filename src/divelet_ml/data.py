"""Data modules and a NumPy data loader.

`DataModule` and `SyntheticRegressionData` are adapted from `d2l/torch.py` in Dive into Deep Learning
(https://github.com/d2l-ai/d2l-en, commit 23d7a5aecceee57d1292c56e90cce307f183bb0a),
Copyright 2019 Amazon.com, Inc. or its affiliates, MIT-0 license.
`ArrayLoader` stands in for PyTorch's `DataLoader`. `FashionMNIST` reads the dataset's four idx files with NumPy,
in place of torchvision.
"""

import gzip
import hashlib
import os
import urllib.request

import numpy as np

from divelet_ml.plotting import show_images
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


FASHION_MNIST_FILES = {'train': ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
                       'val': ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')}
FASHION_MNIST_URL = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/'


def _site_url():
    """The URL of the JupyterLite site that serves this kernel, or None outside JupyterLite. The kernel's own script
    lives in <site root>/{xeus,pyodide}/extensions/..."""
    try:
        import js
    except ImportError:
        return None
    return js.location.href.split('/extensions/')[0].rsplit('/', 1)[0]


# The book's host of datasets. A page cannot read files from it: it sends no CORS headers, so the browser blocks them.
DATA_URL = 'https://d2l-data.s3.amazonaws.com/'
DRIVE = '/drive'  # where JupyterLite's file browser shows in a kernel's file system


def _site_data_url():
    """The `data` folder of the site that serves this kernel, which holds the files the site hosts."""
    return f'{_site_url()}/data/'


def _read_url(url):
    """The bytes at `url`. In JupyterLite, a synchronous request, which a kernel's worker may make."""
    if _site_url() is None:
        with urllib.request.urlopen(url) as response:
            return response.read()
    import js
    request = js.XMLHttpRequest.new()
    request.open('GET', url, False)
    request.responseType = 'arraybuffer'
    request.send()
    assert request.status == 200, f'{url}: HTTP {request.status}'
    return bytes(js.Uint8Array.new(request.response).to_py())


def download(url, folder, sha1_hash=None):
    """Download a file to `folder` and return the local filepath. A file already there with the right SHA-1 hash
    is not downloaded again. In place of the book's `d2l.download`.

    In JupyterLite, a page cannot read files from other hosts, so the reader downloads the file from `url` in the
    browser and uploads it with the file browser; the file is looked for by name anywhere in the file browser."""
    path = os.path.join(folder, url.split('/')[-1])
    if _site_url() is not None:
        path = next((os.path.join(root, url.split('/')[-1]) for root, _, names in os.walk(DRIVE)
                     if url.split('/')[-1] in names), None)
        if path is None:
            raise FileNotFoundError(
                f"{url.split('/')[-1]} is not in the file browser. Download it from {url} (your browser downloads it "
                'when you open the link), upload it with the upload button of the file browser, and run this cell again.')
        assert sha1_hash is None or _sha1(path) == sha1_hash, f'{path} does not have the SHA-1 hash {sha1_hash}'
        return path
    if os.path.exists(path) and (sha1_hash is None or _sha1(path) == sha1_hash):
        return path
    data = _read_url(url)
    os.makedirs(folder, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    assert sha1_hash is None or _sha1(path) == sha1_hash, f'{path} does not have the SHA-1 hash {sha1_hash}'
    return path


def _sha1(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def _fashion_mnist_bytes(name, root):
    """The bytes of one dataset file: in JupyterLite, from the site's `data/fashion-mnist` folder (kept in memory);
    elsewhere, from `root/fashion-mnist`, downloaded there from the dataset's GitHub repository (MIT license,
    Zalando SE) if it is missing."""
    if _site_url() is not None:
        return _read_url(f'{_site_data_url()}fashion-mnist/{name}')
    with open(download(FASHION_MNIST_URL + name, os.path.join(root, 'fashion-mnist')), 'rb') as f:
        return f.read()


def _resize(X, size):
    """Bilinear resize of images `X` of shape (n, h, w) to `size`, with half-pixel centers like PIL's."""
    for axis, new in ((1, size[0]), (2, size[1])):
        old = X.shape[axis]
        if new == old:
            continue
        pos = np.clip((np.arange(new) + 0.5) * old / new - 0.5, 0, old - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, old - 1)
        frac = (pos - lo).astype(np.float32).reshape([-1 if a == axis else 1 for a in range(3)])
        X = np.take(X, lo, axis) * (1 - frac) + np.take(X, hi, axis) * frac
    return X


class ArrayDataset:
    """Pairs (image, label), indexed and measured like a PyTorch dataset; `X` and `y` hold all of them."""
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_fashion_mnist(root, resize=(28, 28)):
    """The training and validation sets of Fashion-MNIST as `ArrayDataset`s: images of shape (1, *resize), float32
    in [0, 1], and int64 labels. In place of torchvision's `FashionMNIST` and its `transforms`."""
    sets = []
    for images, labels in FASHION_MNIST_FILES.values():
        X = np.frombuffer(gzip.decompress(_fashion_mnist_bytes(images, root)), np.uint8, offset=16)
        y = np.frombuffer(gzip.decompress(_fashion_mnist_bytes(labels, root)), np.uint8, offset=8)
        X = X.reshape(-1, 28, 28)
        # In chunks: resizing all 60,000 images at once needs hundreds of MB of temporaries.
        X = np.concatenate([_resize(X[i:i + 5000].astype(np.float32) / 255, resize)
                            for i in range(0, len(X), 5000)])[:, None]
        sets.append(ArrayDataset(X, y.astype(np.int64)))
    return sets


class FashionMNIST(DataModule):
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = load_fashion_mnist(self.root, resize)

    def text_labels(self, indices):
        """Return text labels."""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return self.get_tensorloader((data.X, data.y), train)

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1), nrows, ncols, titles=labels)
