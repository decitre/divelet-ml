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


def test_trainer_without_validation_data_and_with_gradient_clipping():
    class Data(d2l.DataModule):
        def __init__(self):
            super().__init__()
            self.batch_size = 2

        def get_dataloader(self, train):
            return self.get_tensorloader((np.ones((4, 1)), np.ones((4, 1))), train) if train else None

    clipped = []

    @d2l.add_to_class(d2l.Trainer)
    def clip_gradients(self, grad_clip_val, grads):
        clipped.append(grad_clip_val)
        return grads

    model = d2l.LinearRegressionScratch(1, lr=0.1)
    model.board.display = False
    trainer = d2l.Trainer(max_epochs=2, gradient_clip_val=1)
    trainer.fit(model, Data())
    del d2l.Trainer.clip_gradients
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
