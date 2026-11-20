import hashlib

import numpy as np
import pytest

import divelet_ml as d2l
from divelet_ml import nn, optim


def test_array_loader_covers_every_row_once():
    X, y = np.arange(10).reshape(5, 2), np.arange(5)
    for shuffle in (False, True):
        loader = d2l.ArrayLoader((X, y), batch_size=2, shuffle=shuffle)
        batches = list(loader)
        assert len(loader) == len(batches) == 3 and len(batches[-1][0]) == 1
        rows = np.concatenate([b[1] for b in batches])
        assert sorted(rows) == list(range(5))
        assert all((bX[:, 0] // 2 == by).all() for bX, by in batches)  # X and y rows stay paired
    assert [b[1].tolist() for b in d2l.ArrayLoader((X, y), 2)] == [[0, 1], [2, 3], [4]]
    dropped = d2l.ArrayLoader((X, y), 2, drop_last=True)
    assert len(dropped) == 2 and [b[1].tolist() for b in dropped] == [[0, 1], [2, 3]]


def test_save_hyperparameters():
    class Data(d2l.DataModule):
        def __init__(self, a, b=2, _hidden=3):
            super().__init__()
            self.save_hyperparameters()

    data = Data(1)
    assert (data.a, data.b) == (1, 2) and not hasattr(data, '_hidden')
    assert data.hparams == {'a': 1, 'b': 2}


def test_add_to_class():
    class A:
        pass

    @d2l.add_to_class(A)
    def double(self, x):
        return 2 * x

    assert A().double(3) == 6


def test_data_module_dataloaders():
    data = d2l.DataModule()
    assert data.hparams == {'root': '../data', 'num_workers': 4}
    with pytest.raises(NotImplementedError):
        data.train_dataloader()

    class Data(d2l.DataModule):
        def get_dataloader(self, train):
            return 'train' if train else 'val'

    assert (Data().train_dataloader(), Data().val_dataloader()) == ('train', 'val')


def test_check_shape():
    d2l.check_shape(np.ones((2, 3)), (2, 3))
    with pytest.raises(AssertionError, match=r'\(2, 3\) != expected shape \(3, 2\)'):
        d2l.check_shape(np.ones((2, 3)), (3, 2))


def test_show_heatmaps():
    from matplotlib import pyplot as plt
    d2l.show_heatmaps(np.random.rand(2, 3, 4, 5), xlabel='x', ylabel='y', titles=['a', 'b', 'c'])
    fig = plt.gcf()
    axes = fig.axes[:-1]  # the last one is the colorbar
    assert len(axes) == 6 and len(fig.axes) == 7
    assert [ax.get_xlabel() for ax in axes] == ['', '', '', 'x', 'x', 'x']
    assert [ax.get_ylabel() for ax in axes] == ['y', '', '', 'y', '', '']
    assert [ax.get_title() for ax in axes] == ['a', 'b', 'c'] * 2
    plt.close(fig)


def test_softmax():
    X = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
    Y = nn.softmax(X)
    assert np.allclose(Y.sum(axis=-1), 1) and np.allclose(Y[1], 1 / 3)  # no overflow for large inputs
    assert np.allclose(Y[0], np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum())
    assert np.allclose(nn.softmax(X, axis=0).sum(axis=0), 1)


def test_module_train_and_eval_set_submodules():
    class Outer(nn.Module):
        def __init__(self):
            self.inner, self.dropout = nn.Module(), nn.Dropout(0.5)

        def forward(self, X):
            return self.dropout(X)

    model = Outer()
    assert model.training and model.inner.training and model.dropout.training
    assert model.eval() is model and not (model.training or model.inner.training or model.dropout.training)
    model.train()
    assert model.training and model.inner.training and model.dropout.training
    with pytest.raises(NotImplementedError):
        nn.Module()(1)


def test_dropout():
    X = np.ones((100, 100))
    dropout = nn.Dropout(0.5)
    Y = dropout(X)
    assert set(np.unique(Y)) <= {0.0, 2.0} and 0.4 < (Y == 0).mean() < 0.6
    assert dropout.eval()(X) is X
    assert nn.Dropout(0)(X) is X
    assert (nn.Dropout(1)(X) == 0).all()
    with pytest.raises(AssertionError):
        nn.Dropout(1.5)


def test_lazy_linear():
    layer = nn.LazyLinear(3)
    assert layer.weight is None
    X = np.random.rand(2, 5, 4)
    Y = layer(X)
    assert Y.shape == (2, 5, 3) and layer.weight.shape == (3, 4) and layer.bias.shape == (3,)
    assert np.abs(layer.weight).max() <= 1 / 2 and np.abs(layer.bias).max() <= 1 / 2  # 1/sqrt(4)
    assert np.allclose(Y, X @ layer.weight.T + layer.bias)
    weight = layer.weight
    layer(X)
    assert layer.weight is weight  # set on the first call only
    no_bias = nn.LazyLinear(1, bias=False)
    assert np.allclose(no_bias(X), X @ no_bias.weight.T) and no_bias.bias is None


def test_plt_is_pyplot_and_unknown_attributes_raise():
    from matplotlib import pyplot
    assert d2l.plt is pyplot
    with pytest.raises(AttributeError, match='no attribute'):
        d2l.no_such_name


def test_set_figsize():
    d2l.set_figsize((4.5, 3.5))
    assert list(d2l.plt.rcParams['figure.figsize']) == [4.5, 3.5]
    d2l.set_figsize()
    assert list(d2l.plt.rcParams['figure.figsize']) == [3.5, 2.5]


def test_plot():
    plt = d2l.plt
    x = np.arange(0, 3, 0.1)
    # Several curves against one x, with labels, legend, limits and scales
    d2l.plot(x, [x ** 2, 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'tangent'],
             xlim=[0, 3], ylim=[-3, 9], yscale='linear')
    axes = plt.gca()
    lines = axes.get_lines()
    assert len(lines) == 2 and np.allclose(lines[0].get_xdata(), x) and np.allclose(lines[1].get_ydata(), 2 * x - 3)
    assert lines[1].get_linestyle() == '--' and lines[1].get_color() == 'm'
    assert (axes.get_xlabel(), axes.get_ylabel()) == ('x', 'f(x)')
    assert [t.get_text() for t in axes.get_legend().get_texts()] == ['f(x)', 'tangent']
    assert axes.get_xlim() == (0, 3) and axes.get_ylim() == (-3, 9)
    # One curve given as Y only: plotted against its indices; the axes are cleared first
    d2l.plot([1.0, 4.0, 9.0], axes=axes)
    lines = axes.get_lines()
    assert len(lines) == 1 and list(lines[0].get_xdata()) == [0, 1, 2] and list(lines[0].get_ydata()) == [1, 4, 9]
    # One x for two curves, as a 1-axis Y
    d2l.plot(x, x ** 2, axes=axes)
    assert len(axes.get_lines()) == 1
    plt.close('all')


def test_module_parameters_and_lazy_linear_initializers():
    class Net(nn.Module):
        def __init__(self):
            self.w, self.count, self.layer = np.ones(2), np.arange(3), nn.LazyLinear(2, bias_init=np.zeros)

    net = Net()
    assert list(net.modules()) == [net, net.layer]
    assert [p is net.w for p in net.parameters()] == [True]  # integer arrays and uninitialized layers are not parameters
    net.layer(np.ones((1, 3)))
    assert [p.shape for p in net.parameters()] == [(2,), (2, 3), (2,)] and (net.layer.bias == 0).all()
    layer = nn.LazyLinear(1, weight_init=np.ones)
    assert np.allclose(layer(np.ones((1, 4))), 4 + layer.bias)


def test_mse_loss():
    assert nn.MSELoss()(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == 2.0


def test_param_grads_restores_the_parameters():
    model = d2l.LinearRegressionScratch(num_inputs=2, lr=0.1)
    model.w[:] = [[1.0], [2.0]]
    w, b = model.w, model.b
    X, y = np.array([[1.0, 1.0], [2.0, 0.0]]), np.array([[0.0], [1.0]])
    grad_w, grad_b = d2l.param_grads(lambda: model.loss(model(X), y), model, [w, b])
    residual = X @ w + b - y  # loss = mean(residual ** 2) / 2
    assert np.allclose(grad_w, X.T @ residual / 2) and np.allclose(grad_b, residual.mean())
    assert model.w is w and model.b is b
    with pytest.raises(AssertionError, match='not an attribute'):
        d2l.param_grads(lambda: 0.0, model, [np.ones(1)])


def test_optim_sgd_weight_decay_and_parameter_groups():
    w, b = np.ones(2), np.ones(1)
    optim.SGD([w, b], lr=0.1).step([np.ones(2), np.ones(1)])
    assert np.allclose(w, 0.9) and np.allclose(b, 0.9)
    w, b = np.ones(2), np.ones(1)
    sgd = optim.SGD([{'params': w, 'weight_decay': 3}, {'params': [b]}], lr=0.1)
    assert sgd.params[0] is w and sgd.params[1] is b
    sgd.step([np.zeros(2), np.zeros(1)])
    assert np.allclose(w, 0.7) and np.allclose(b, 1)


@pytest.mark.parametrize('make_model', [lambda: d2l.LinearRegressionScratch(2, lr=0.03),
                                        lambda: d2l.LinearRegression(lr=0.03)], ids=['scratch', 'concise'])
def test_trainer_fits_linear_regression(make_model):
    np.random.seed(0)
    model = make_model()
    model.board.display = False
    data = d2l.SyntheticRegressionData(w=np.array([2, -3.4]), b=4.2)
    trainer = d2l.Trainer(max_epochs=3)
    trainer.fit(model, data)
    w, b = model.get_w_b() if hasattr(model, 'get_w_b') else (model.w, model.b)
    assert np.abs(w.reshape(2) - [2, -3.4]).max() < 0.3 and abs(b.item() - 4.2) < 0.3
    assert trainer.train_batch_idx == 3 * 32 and trainer.val_batch_idx == 3 * 32
    assert set(model.board.data) == {'train_loss', 'val_loss'} and len(model.board.data['val_loss']) == 3


def test_progress_board_averages_every_n_points_and_draws(capsys):
    board = d2l.ProgressBoard('x', xlim=[0, 4], ylim=[0.1, 9], yscale='log')
    for x in range(4):
        board.draw(x, x ** 2, 'square', every_n=2)
        board.draw(x, 1, 'one')
    assert board.data['square'] == [(0.5, 0.5), (2.5, 6.5)] and len(board.data['one']) == 4
    axes = board.fig.axes[0]
    assert [t.get_text() for t in axes.get_legend().get_texts()] == ['square', 'one']
    assert (axes.get_xlabel(), axes.get_yscale(), axes.get_xlim()) == ('x', 'log', (0, 4))
    d2l.plt.close('all')


def test_module_base_class():
    model = d2l.Module()
    assert model.hparams == {'plot_train_per_epoch': 2, 'plot_valid_per_epoch': 1}
    for method, args in ((model.loss, (0, 0)), (model.configure_optimizers, ())):
        with pytest.raises(NotImplementedError):
            method(*args)
    with pytest.raises(AssertionError, match='Neural network'):
        model(np.ones(1))
    model.net = lambda X: 2 * X
    assert model(3) == 6


def test_trainer_without_validation_data_and_with_gradient_clipping(monkeypatch):
    class Data(d2l.DataModule):
        def __init__(self):
            super().__init__()
            self.batch_size = 2

        def get_dataloader(self, train):
            return self.get_tensorloader((np.ones((4, 1)), np.ones((4, 1))), train) if train else None

    clipped = []

    monkeypatch.setattr(d2l.Trainer, 'clip_gradients', lambda self, grad_clip_val, grads: clipped.append(grad_clip_val) or grads)
    model = d2l.LinearRegressionScratch(1, lr=0.1)
    model.board.display = False
    trainer = d2l.Trainer(max_epochs=2, gradient_clip_val=1)
    trainer.fit(model, Data())
    assert trainer.num_val_batches == 0 and trainer.val_batch_idx == 0 and clipped == [1] * 4


def test_training_and_plotting_without_jupyter(monkeypatch):
    # Outside Jupyter, neither IPython nor matplotlib_inline may be installed: None in sys.modules makes imports fail.
    import sys
    monkeypatch.setitem(sys.modules, 'IPython', None)
    monkeypatch.setitem(sys.modules, 'matplotlib_inline', None)
    model = d2l.LinearRegression(lr=0.03)
    model.board.display = False
    d2l.Trainer(max_epochs=1).fit(model, d2l.SyntheticRegressionData(w=np.array([2, -3.4]), b=4.2))
    assert len(model.board.data['val_loss']) == 1
    d2l.plot([1.0, 2.0])
    d2l.plt.close('all')


def test_browser_epochs(monkeypatch):
    import sys
    assert d2l.browser_epochs(10) == 10  # CPython
    monkeypatch.setattr('sys.platform', 'emscripten')
    monkeypatch.setitem(sys.modules, 'pyjs', None)  # not importable: Pyodide
    assert d2l.browser_epochs(10) == 1 and d2l.browser_epochs(10, pyodide=4) == 4 and d2l.browser_epochs(2, pyodide=4) == 2
    monkeypatch.setitem(sys.modules, 'pyjs', type(sys)('pyjs'))  # importable: xeus-python
    assert d2l.browser_epochs(10) == 3
    assert d2l.browser_examples(60000, pyodide=2000, xeus=9000) == 9000
    monkeypatch.setitem(sys.modules, 'pyjs', None)
    assert d2l.browser_examples(60000, pyodide=2000, xeus=9000) == 2000


def test_download_checks_the_hash_and_reuses_the_file(tmp_path, monkeypatch):
    import hashlib
    from divelet_ml import data
    calls = []
    monkeypatch.setattr(data, '_read_url', lambda url: calls.append(url) or b'abc')
    sha1 = hashlib.sha1(b'abc').hexdigest()
    path = d2l.download('http://host/dir/file.csv', str(tmp_path / 'sub'), sha1_hash=sha1)
    assert path == str(tmp_path / 'sub' / 'file.csv') and open(path, 'rb').read() == b'abc'
    assert d2l.download('http://host/dir/file.csv', str(tmp_path / 'sub'), sha1_hash=sha1) == path
    assert len(calls) == 1  # the second call found the file
    assert sorted(f.name for f in (tmp_path / 'sub').iterdir()) == ['file.csv']  # no partial file left
    with pytest.raises(AssertionError, match='SHA-1'):
        d2l.download('http://host/dir/other.csv', str(tmp_path), sha1_hash='0' * 40)


def test_download_in_jupyterlite_looks_in_the_file_browser(tmp_path, monkeypatch):
    import hashlib
    from divelet_ml import data
    monkeypatch.setattr(data, '_site_url', lambda: 'https://site')  # a JupyterLite kernel
    monkeypatch.setattr(data, 'DRIVE', str(tmp_path))
    monkeypatch.setattr(data, '_read_url', lambda url: pytest.fail('a page cannot read other hosts'))
    url = 'https://host/dir/file.csv'
    with pytest.raises(FileNotFoundError, match='upload it'):
        d2l.download(url, '../data')
    (tmp_path / 'any' / 'folder').mkdir(parents=True)
    (tmp_path / 'any' / 'folder' / 'file.csv').write_bytes(b'abc')  # uploaded anywhere in the file browser
    sha1 = hashlib.sha1(b'abc').hexdigest()
    assert d2l.download(url, '../data', sha1_hash=sha1) == str(tmp_path / 'any' / 'folder' / 'file.csv')
    with pytest.raises(AssertionError, match='SHA-1'):
        d2l.download(url, '../data', sha1_hash='0' * 40)


def _fake_fashion_mnist(folder, n_train=40, n_val=10):
    """Fashion-MNIST-shaped idx files, gzipped: image i has the constant gray level 10 * (i % 10) and label i % 10."""
    import gzip
    import struct
    from divelet_ml import data
    folder.mkdir(parents=True, exist_ok=True)
    for (images, labels), n in zip(data.FASHION_MNIST_FILES.values(), (n_train, n_val)):
        y = np.arange(n) % 10
        pixels = np.repeat((10 * y).astype(np.uint8), 28 * 28)
        (folder / images).write_bytes(gzip.compress(struct.pack('>IIII', 2051, n, 28, 28) + pixels.tobytes()))
        (folder / labels).write_bytes(gzip.compress(struct.pack('>II', 2049, n) + y.astype(np.uint8).tobytes()))


@pytest.fixture
def fashion(tmp_path, monkeypatch):
    """The Fashion-MNIST files served from tmp_path/'source'; the data root is tmp_path/'data'."""
    from divelet_ml import data
    _fake_fashion_mnist(tmp_path / 'source')
    monkeypatch.setattr(data, 'FASHION_MNIST_URL', (tmp_path / 'source').as_uri() + '/')
    (tmp_path / 'work').mkdir()
    monkeypatch.chdir(tmp_path / 'work')  # the default root is ../data
    return tmp_path


def test_fashion_mnist(fashion, monkeypatch):
    import matplotlib
    from divelet_ml import data
    matplotlib.use('Agg')
    fm = d2l.FashionMNIST(batch_size=16)
    assert (len(fm.train), len(fm.val)) == (40, 10)
    X, y = fm.train[3]
    assert X.shape == (1, 28, 28) and X.dtype == np.float32 and y.dtype == np.int64 and y == 3
    assert np.allclose(X, 30 / 255)  # no resizing: the pixels divided by 255
    assert fm.text_labels([0, 9]) == ['t-shirt', 'ankle boot']
    batches = list(fm.train_dataloader())
    assert len(batches) == 3 and batches[0][0].shape == (16, 1, 28, 28) and batches[-1][1].shape == (8,)
    assert sorted(np.concatenate([b[1] for b in fm.val_dataloader()])) == sorted(np.arange(10))
    fm.visualize(next(iter(fm.val_dataloader())), nrows=2, ncols=5)
    assert len(matplotlib.pyplot.gcf().axes) == 10
    matplotlib.pyplot.close('all')
    # The files were downloaded once to the root and are read from there afterwards.
    monkeypatch.setattr(data, 'FASHION_MNIST_URL', 'file:///nonexistent/')
    assert len(d2l.FashionMNIST().train) == 40
    small = d2l.FashionMNIST(num_train=7)
    assert len(small.train) == 7 and len(small.val) == 10 and small.train[3][1] == 3
    assert d2l.FashionMNIST(resize=(32, 32)).train[0][0].shape == (1, 32, 32)
    assert sorted(p.name for p in (fashion / 'data' / 'fashion-mnist').iterdir()) == sorted(
        n for names in data.FASHION_MNIST_FILES.values() for n in names)


def test_resize_is_bilinear_with_half_pixel_centers():
    from divelet_ml.data import _resize
    X = np.full((2, 28, 28), 0.25, np.float32)
    assert np.allclose(_resize(X, (32, 14)), 0.25) and _resize(X, (32, 14)).shape == (2, 32, 14)
    ramp = np.tile(np.arange(4, dtype=np.float32), (1, 4, 1))  # columns 0, 1, 2, 3
    assert np.allclose(_resize(ramp, (4, 2))[0, 0], [0.5, 2.5])  # halving averages neighbors
    assert np.allclose(_resize(ramp, (4, 8))[0, 0], [0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3])  # doubling interpolates


def test_read_url_outside_and_inside_jupyterlite(tmp_path, monkeypatch):
    import sys
    import types
    from divelet_ml import data
    (tmp_path / 'f.bin').write_bytes(b'xyz')
    assert data._site_url() is None and data._read_url((tmp_path / 'f.bin').as_uri()) == b'xyz'

    class Request:  # the browser's XMLHttpRequest, served from a dict
        files = {'https://host/site/data/fashion-mnist/f.bin': b'abc'}

        def open(self, method, url, asynchronous):
            assert (method, asynchronous) == ('GET', False)
            self.url = url

        def send(self):
            self.status = 200 if self.url in self.files else 404
            self.response = self.files.get(self.url)

    js = types.SimpleNamespace(
        location=types.SimpleNamespace(href='https://host/site/xeus/extensions/@jupyterlite/x/static/1.js?v=2'),
        XMLHttpRequest=types.SimpleNamespace(new=Request),
        Uint8Array=types.SimpleNamespace(new=lambda b: types.SimpleNamespace(to_py=lambda: memoryview(b))))
    monkeypatch.setitem(sys.modules, 'js', js)
    assert data._site_url() == 'https://host/site' and data._site_data_url() == 'https://host/site/data/'
    assert data._read_url('https://host/site/data/fashion-mnist/f.bin') == b'abc'
    assert data._fashion_mnist_bytes('f.bin', '../data') == b'abc'
    with pytest.raises(AssertionError, match='HTTP 404'):
        data._read_url('https://host/site/data/missing')


def test_cross_entropy_is_stable_and_differentiable():
    import autograd
    logits = np.array([[2.0, 0.5, -1.0], [1000.0, 0.0, -1000.0]])
    y = np.array([0, 2])
    expected = np.array([np.log(np.exp(logits[0]).sum()) - 2.0, 2000.0])
    assert np.allclose(nn.cross_entropy(logits, y, reduction='none'), expected)  # no overflow at 1000
    assert np.isclose(nn.cross_entropy(logits, y), expected.mean())
    grad = autograd.grad(lambda z: nn.cross_entropy(z, y))(logits)
    onehot = np.eye(3)[y]
    assert np.allclose(grad, (nn.softmax(logits) - onehot) / 2)


def test_sequential_flatten_relu():
    net = nn.Sequential(nn.Flatten(), nn.LazyLinear(5), nn.ReLU(), nn.LazyLinear(2))
    X = np.random.randn(4, 3, 2, 2).astype(np.float32)
    assert net(X).shape == (4, 2) and net(X).dtype == np.float32  # the parameters take the input's dtype
    assert (nn.ReLU()(np.array([-1.0, 0.0, 2.0])) == [0, 0, 2]).all()
    assert [p.shape for p in net.parameters()] == [(5, 12), (5,), (2, 5), (2,)]


def test_classifier_accuracy_loss_and_softmax_regression_training():
    class Data(d2l.DataModule):
        def __init__(self):
            super().__init__()
            self.batch_size = 8
            rng = np.random.default_rng(0)
            self.y = rng.integers(0, 3, 64)
            self.X = (np.eye(3)[self.y] * 3 + rng.normal(0, 0.3, (64, 3))).astype(np.float32).reshape(64, 1, 3)

        def get_dataloader(self, train):
            return self.get_tensorloader((self.X, self.y), train)

    clf = d2l.Classifier()
    Y_hat, Y = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]), np.array([1, 0, 0])
    assert np.isclose(clf.accuracy(Y_hat, Y), 2 / 3) and clf.accuracy(Y_hat, Y, averaged=False).tolist() == [1, 1, 0]
    assert clf.loss(Y_hat, Y, averaged=False).shape == (3,) and np.isclose(clf.loss(Y_hat, Y), clf.loss(Y_hat, Y, False).mean())
    model = d2l.SoftmaxRegression(num_outputs=3, lr=0.5)
    model.board.display = False
    data = Data()
    trainer = d2l.Trainer(max_epochs=3)
    trainer.fit(model, data)
    assert set(model.board.data) == {'train_loss', 'val_loss', 'val_acc'}
    assert model.board.data['val_acc'][-1][1] > 0.9 and isinstance(trainer.optim, optim.SGD)
    assert model.accuracy(model(data.X), data.y) > 0.9


def test_show_images(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    axes = d2l.show_images(np.zeros((6, 4, 4)), 2, 3, titles=list('abcdef'))
    assert len(axes) == 6 and axes[4].get_title() == 'e'
    assert len(d2l.show_images(np.zeros((2, 4, 4)), 1, 2)) == 2
    matplotlib.pyplot.close('all')


def test_module_tree_methods_state_dict_and_apply():
    net = nn.Sequential(nn.LazyLinear(3), nn.ReLU(), nn.LazyLinear(2))
    assert [n for n, _ in net.named_parameters()] == []  # lazy: nothing yet
    net(np.random.rand(4, 5))
    assert [type(m) for m in net.children()] == [nn.Linear, nn.ReLU, nn.Linear]  # a lazy layer becomes a Linear
    assert [(n, p.shape) for n, p in net.named_parameters()] == [('0.weight', (3, 5)), ('0.bias', (3,)),
                                                                 ('2.weight', (2, 3)), ('2.bias', (2,))]
    assert net[0] is getattr(net, '0') and net[-1] is net[2]

    clone = nn.Sequential(nn.LazyLinear(3), nn.ReLU(), nn.LazyLinear(2))
    clone.load_state_dict(net.state_dict())
    X = np.random.rand(2, 5)
    assert np.allclose(clone(X), net(X)) and clone[0].weight is not net[0].weight  # a copy

    seen = []
    net.apply(lambda m: seen.append(type(m).__name__))
    assert seen == ['Linear', 'ReLU', 'Linear', 'Sequential']  # children first, then the module

    outer = nn.Module()
    outer.add_module('inner', net)
    assert outer.inner is net and [n for n, _ in outer.named_parameters()][0] == 'inner.0.weight'
    with pytest.raises(AssertionError, match='not a Module'):
        outer.add_module('x', np.ones(1))


def test_shared_layer_counts_once_and_gets_its_gradients_added():
    shared = nn.LazyLinear(2)
    net = nn.Sequential(shared, nn.ReLU(), shared)
    X = np.random.rand(3, 2)
    net(X)
    assert len(list(net.modules())) == 3 and len(net.parameters()) == 2  # the shared layer is one module
    assert len(list(net.named_parameters())) == 2 and len(net.state_dict()) == 4  # state_dict has both names
    loss = lambda: (net(X) ** 2).sum()
    grad_w, grad_b = d2l.param_grads(loss, net, net.parameters())
    w, eps = shared.weight, 1e-6
    w[0, 0] += eps
    up = loss()
    w[0, 0] -= 2 * eps
    down = loss()
    w[0, 0] += eps
    numeric = (up - down) / (2 * eps)
    assert np.isclose(grad_w[0, 0], numeric, rtol=1e-4)  # both uses of the layer count


def test_linear_relu_and_initializers():
    layer = nn.Linear(4, 3)
    assert layer.weight.shape == (3, 4) and layer.weight.dtype == np.float32 and type(layer) is nn.Linear
    assert np.allclose(layer(np.ones((2, 4))), np.ones((2, 4)) @ layer.weight.T + layer.bias)
    assert (nn.relu(np.array([-1.0, 2.0])) == [0, 2]).all()
    w = np.zeros((6, 4))
    assert nn.init.normal_(w, 0, 0.01) is w and 0 < np.abs(w).max() < 0.1
    nn.init.uniform_(w, 5, 6)
    assert w.min() >= 5 and w.max() <= 6
    nn.init.constant_(w, 7)
    assert (w == 7).all()
    nn.init.zeros_(w)
    assert (w == 0).all()
    nn.init.ones_(w)
    assert (w == 1).all()
    nn.init.xavier_uniform_(w)
    assert np.abs(w).max() <= np.sqrt(6 / 10)
    kernel = np.zeros((6, 4, 3, 3))
    nn.init.xavier_uniform_(kernel)
    assert np.abs(kernel).max() <= np.sqrt(6 / (10 * 9))  # the window size is in both fans
    with pytest.raises(AssertionError, match='two dimensions'):
        nn.init.xavier_uniform_(np.zeros(3))


def test_device_helpers_without_a_gpu():
    assert (d2l.cpu(), d2l.gpu(1), d2l.num_gpus()) == ('cpu', 'cuda:1', 0)
    assert (d2l.try_gpu(), d2l.try_gpu(10), d2l.try_all_gpus()) == ('cpu', 'cpu', ['cpu'])


def finite_difference(fn, array, index, eps=1e-6):
    original = array[index]
    array[index] = original + eps
    up = fn()
    array[index] = original - eps
    down = fn()
    array[index] = original
    return (up - down) / (2 * eps)


def test_conv2d_values_and_gradients():
    import autograd
    X = np.arange(9.0).reshape(1, 1, 3, 3)
    K = np.array([[0.0, 1.0], [2.0, 3.0]]).reshape(1, 1, 2, 2)
    assert (nn.conv2d(X, K)[0, 0] == [[19, 25], [37, 43]]).all()  # the book's figure
    assert nn.conv2d(X, K, bias=np.array([1.0]))[0, 0, 0, 0] == 20
    assert nn.conv2d(X, K, padding=1).shape == (1, 1, 4, 4) and nn.conv2d(X, K, stride=(2, 1)).shape == (1, 1, 1, 2)
    rng = np.random.default_rng(0)
    X, W, b = rng.random((2, 2, 7, 6)), rng.random((3, 2, 3, 2)), rng.random(3)
    loss = lambda X, W, b: (nn.conv2d(X, W, b, stride=(2, 1), padding=(1, 0)) ** 2).sum()
    gX, gW, gb = autograd.grad(loss, (0, 1, 2))(X, W, b)
    assert np.isclose(gX[1, 0, 3, 2], finite_difference(lambda: loss(X, W, b), X, (1, 0, 3, 2)), rtol=1e-5)
    assert np.isclose(gW[2, 1, 1, 0], finite_difference(lambda: loss(X, W, b), W, (2, 1, 1, 0)), rtol=1e-5)
    assert np.isclose(gb[1], finite_difference(lambda: loss(X, W, b), b, (1,)), rtol=1e-5)


@pytest.mark.parametrize('pool', [nn.max_pool2d, nn.avg_pool2d])
def test_pooling_values_and_gradients(pool):
    import autograd
    X = np.arange(16.0).reshape(1, 1, 4, 4)
    expected = {'max_pool2d': [[5, 7], [13, 15]], 'avg_pool2d': [[2.5, 4.5], [10.5, 12.5]]}[pool.__name__]
    assert (pool(X, 2)[0, 0] == expected).all()
    assert pool(X, 3, stride=2, padding=1).shape == (1, 1, 2, 2) and pool(X, (2, 3), (2, 3), (0, 1)).shape == (1, 1, 2, 2)
    X = np.random.default_rng(1).random((2, 2, 7, 6))
    loss = lambda X: (pool(X, (3, 2), (2, 1), (1, 0)) ** 2).sum()
    g = autograd.grad(loss)(X)
    assert np.isclose(g[1, 0, 3, 2], finite_difference(lambda: loss(X), X, (1, 0, 3, 2)), rtol=1e-5)
    assert nn.max_pool2d(np.full((1, 1, 2, 2), -5.0), 3, padding=1)[0, 0, 0, 0] == -5  # padding is minus infinity
    assert nn.avg_pool2d(np.ones((1, 1, 2, 2)), 2, padding=1)[0, 0, 0, 0] == 0.25  # padding is zeros, counted


def test_avg_pool2d_tiled_gradient_ignores_the_cropped_border():
    import autograd
    X = np.random.default_rng(2).random((2, 2, 7, 6))  # the odd height drops its last row
    loss = lambda X: (nn.avg_pool2d(X, 2) ** 2).sum()
    g = autograd.grad(loss)(X)
    assert np.isclose(g[1, 0, 3, 2], finite_difference(lambda: loss(X), X, (1, 0, 3, 2)), rtol=1e-5)
    assert (g[:, :, 6] == 0).all()


def test_conv_and_pool_layers_sigmoid_and_sequential_iteration():
    net = nn.Sequential(nn.LazyConv2d(4, kernel_size=3, padding=1), nn.Sigmoid(), nn.MaxPool2d(2), nn.AvgPool2d(2),
                        nn.Flatten(), nn.LazyLinear(2))
    X = np.random.rand(3, 2, 8, 8).astype(np.float32)
    assert net(X).shape == (3, 2) and net(X).dtype == np.float32
    conv = net[0]
    assert type(conv) is nn.Conv2d and conv.weight.shape == (4, 2, 3, 3) and conv.bias.shape == (4,)
    assert np.abs(conv.weight).max() <= 1 / np.sqrt(18) and len(net) == 6
    assert [type(layer).__name__ for layer in net][:3] == ['Conv2d', 'Sigmoid', 'MaxPool2d']
    assert np.allclose(nn.sigmoid(np.array([0.0])), 0.5)
    eager = nn.Conv2d(2, 3, (1, 2), bias=False)
    assert eager.weight.shape == (3, 2, 1, 2) and eager.bias is None and eager(X).shape == (3, 3, 8, 7)
    lazy = nn.LazyConv2d(1, 3, bias=False)
    assert lazy(np.random.rand(1, 5, 4, 4)).dtype == np.float64 and lazy.weight.dtype == np.float64


def test_module_apply_init_creates_lazy_parameters_then_initializes():
    class Model(d2l.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.LazyLinear(3), nn.LazyLinear(1))

    model = Model()
    model.apply_init([np.ones((2, 4))])
    assert model.net[0].weight.shape == (3, 4)
    model.apply_init([np.ones((2, 4))], lambda m: type(m) is nn.Linear and nn.init.constant_(m.weight, 2))
    assert (model.net[0].weight == 2).all() and (model.net[1].weight == 2).all()


def test_corr2d_loop_matches_conv2d():
    X, K = np.arange(9.0).reshape(3, 3), np.array([[0.0, 1.0], [2.0, 3.0]])
    assert (d2l.corr2d(X, K) == [[19, 25], [37, 43]]).all()
    X = np.random.rand(6, 8)
    assert np.allclose(d2l.corr2d(X, K), nn.conv2d(X[None, None], K[None, None])[0, 0])


def test_clip_gradients_scales_to_the_norm_and_leaves_small_gradients_alone():
    trainer = d2l.Trainer(max_epochs=1, gradient_clip_val=1)
    grads = [np.array([3.0, 0.0]), np.array([[0.0], [4.0]])]  # joint norm 5
    clipped = trainer.clip_gradients(1, grads)
    assert np.isclose(np.sqrt(sum((g ** 2).sum() for g in clipped)), 1) and np.allclose(clipped[0], [0.6, 0.0])
    assert trainer.clip_gradients(10, grads) is grads


def test_vocab_orders_tokens_and_maps_unknown_ones_to_unk():
    vocab = d2l.Vocab(list('banana'))
    assert vocab.idx_to_token == ['<unk>', 'a', 'b', 'n'] and len(vocab) == 4
    assert vocab['n'] == 3 and vocab['z'] == vocab.unk == 0 and vocab[['a', 'b', 'z']] == [1, 2, 0]
    assert vocab.to_tokens([1, 2]) == ['a', 'b'] and vocab.to_tokens(3) == 'n'
    assert d2l.Vocab([list('ab'), list('ba')]).idx_to_token == ['<unk>', 'a', 'b']  # a list of lines is flattened
    assert d2l.Vocab(list('aab'), min_freq=2).idx_to_token == ['<unk>', 'a']


def test_time_machine_windows_are_shifted_by_one_and_split_into_train_and_validation(monkeypatch):
    monkeypatch.setattr(d2l.TimeMachine, '_download', lambda self: 'The Time Traveller (for so it will be) ' * 4)
    data = d2l.TimeMachine(batch_size=4, num_steps=5, num_train=20, num_val=10)
    corpus = [data.vocab[t] for t in data._tokenize(data._preprocess('The Time Traveller (for so it will be) ' * 4))]
    assert data.X.shape == data.Y.shape == (len(corpus) - 5, 5) and not data.X.flags.owndata  # views, not copies
    assert (data.X[:, 1:] == data.Y[:, :-1]).all() and data.X[3].tolist() == corpus[3:8] and data.Y[3].tolist() == corpus[4:9]
    train, val = list(data.train_dataloader()), list(data.val_dataloader())
    assert sum(len(b[0]) for b in train) == 20 and sum(len(b[0]) for b in val) == 10
    assert sorted(np.concatenate([b[0] for b in val])[:, 0].tolist()) == sorted(data.X[20:30, 0].tolist())


def test_time_machine_text_comes_from_the_site_in_jupyterlite_and_is_checked(monkeypatch):
    from divelet_ml import data
    text = b'the time machine'
    monkeypatch.setattr(data, '_site_url', lambda: 'https://site')
    monkeypatch.setattr(data, '_read_url', lambda url: text if url == 'https://site/data/timemachine/timemachine.txt' else pytest.fail(url))
    monkeypatch.setattr(data, 'TIME_MACHINE_SHA1', hashlib.sha1(text).hexdigest())
    assert data._time_machine_text('unused') == 'the time machine'
    monkeypatch.setattr(data, 'TIME_MACHINE_SHA1', '0' * 40)
    with pytest.raises(AssertionError, match='SHA-1'):
        data._time_machine_text('unused')


def test_time_machine_text_is_downloaded_outside_jupyterlite(tmp_path, monkeypatch):
    from divelet_ml import data
    monkeypatch.setattr(data, '_site_url', lambda: None)
    monkeypatch.setattr(data, 'download', lambda url, folder, sha1_hash: calls.append((url, folder, sha1_hash)) or str(tmp_path / 'timemachine.txt'))
    (tmp_path / 'timemachine.txt').write_text('abc def')
    calls = []
    assert data.TimeMachine(batch_size=1, num_steps=2, num_train=2, num_val=2).X.shape == (5, 2)  # 'abc def': 7 characters
    assert calls == [(d2l.DATA_URL + 'timemachine.txt', '../data', data.TIME_MACHINE_SHA1)]


def test_nn_rnn_matches_a_loop_and_is_differentiable():
    rnn = nn.RNN(3, 4)
    assert all(p.dtype == np.float32 for p in rnn.parameters()) and len(rnn.parameters()) == 4
    X = np.random.rand(5, 2, 3).astype(np.float32)
    outputs, state = rnn(X)
    h = np.zeros((2, 4), np.float32)
    for t in range(5):
        h = np.tanh(X[t] @ rnn.weight_ih.T + rnn.bias_ih + h @ rnn.weight_hh.T + rnn.bias_hh)
        np.testing.assert_allclose(outputs[t], h, rtol=1e-5, atol=1e-6)
    assert outputs.shape == (5, 2, 4) and state.shape == (1, 2, 4)
    _, middle = rnn(X[:3])  # continuing from a state is the same as running all the steps at once
    np.testing.assert_allclose(rnn(X[3:], middle)[0], outputs[3:], rtol=1e-5, atol=1e-6)
    grads = d2l.param_grads(lambda: (rnn(X)[0] ** 2).sum(), rnn, rnn.parameters())
    assert [g.shape for g in grads] == [p.shape for p in rnn.parameters()] and all(np.abs(g).sum() > 0 for g in grads)


@pytest.mark.parametrize('concise', [False, True])
def test_rnn_language_model_trains_clips_and_predicts(concise):
    text = 'the time traveller for so it will be convenient to speak of him ' * 40
    class Data(d2l.TimeMachine):
        def _download(self):
            return text
    data = Data(batch_size=64, num_steps=8, num_train=300, num_val=100)
    V = len(data.vocab)
    model = (d2l.RNNLM(d2l.RNN(V, 16), V, lr=1) if concise else d2l.RNNLMScratch(d2l.RNNScratch(V, 16), V, lr=1))
    model.board.display = False
    X, Y = next(iter(data.train_dataloader()))
    assert model(X).shape == (64, 8, V)
    before = float(model.loss(model(X), Y))
    d2l.Trainer(max_epochs=25, gradient_clip_val=1).fit(model, data)
    assert float(model.loss(model(X), Y)) < before - 0.5
    out = model.predict('time ', 12, data.vocab)
    assert len(out) == len('time ') + 12 and out.startswith('time ') and set(out) <= set(data.vocab.idx_to_token)


def test_check_len():
    d2l.check_len([1, 2], 2)
    with pytest.raises(AssertionError, match='expected length 3'):
        d2l.check_len([1, 2], 3)
