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
    assert data.time_machine_text('unused') == 'the time machine'
    monkeypatch.setattr(data, 'TIME_MACHINE_SHA1', '0' * 40)
    with pytest.raises(AssertionError, match='SHA-1'):
        data.time_machine_text('unused')


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


def test_sequential_add_module_appends_under_a_name_and_keeps_the_order():
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
    net.add_module('last', nn.Linear(3, 1))
    assert len(net) == net.num_layers == 3 and [type(m).__name__ for m in net] == ['Linear', 'ReLU', 'Linear']
    assert net[2] is net.last and net[-1] is net.last and net.forward(np.ones((4, 2), np.float32)).shape == (4, 1)
    assert [n for n, _ in net.named_parameters()] == ['0.weight', '0.bias', 'last.weight', 'last.bias']
    with pytest.raises(AssertionError, match='already'):
        net.add_module('last', nn.ReLU())


def test_grouped_convolution_matches_one_convolution_per_group_and_has_gradients():
    X, W = np.random.randn(2, 4, 5, 5), np.random.randn(6, 2, 3, 3)
    Y = nn.conv2d(X, W, padding=1, groups=2)
    expected = np.concatenate([nn.conv2d(X[:, :2], W[:3], padding=1), nn.conv2d(X[:, 2:], W[3:], padding=1)], axis=1)
    np.testing.assert_allclose(Y, expected)
    assert Y.shape == (2, 6, 5, 5) and nn.conv2d(X, W, bias=np.arange(6.0), padding=1, groups=2).shape == Y.shape
    from autograd import grad
    g = grad(lambda W: (nn.conv2d(X, W, padding=1, groups=2) ** 2).sum())(W)
    for index in [(0, 0, 0, 0), (4, 1, 2, 1)]:  # one weight of each group, against a central difference
        h = 1e-5
        up, down = W.copy(), W.copy()
        up[index] += h
        down[index] -= h
        numeric = ((nn.conv2d(X, up, padding=1, groups=2) ** 2).sum() - (nn.conv2d(X, down, padding=1, groups=2) ** 2).sum()) / (2 * h)
        assert np.isclose(g[index], numeric, rtol=1e-4)
    layer = nn.LazyConv2d(6, 3, padding=1, groups=2)
    assert layer(X.astype(np.float32)).shape == (2, 6, 5, 5) and layer.weight.shape == (6, 2, 3, 3) and nn.Conv2d(4, 6, 3, groups=2).weight.shape == (6, 2, 3, 3)
    with pytest.raises(AssertionError, match='not divisible'):
        nn.Conv2d(4, 6, 3, groups=4)
    with pytest.raises(AssertionError, match='not divisible'):
        nn.conv2d(X, W[:5], groups=2)


def test_adaptive_average_pooling_averages_each_channel_over_the_image():
    X = np.random.rand(2, 3, 4, 5)
    np.testing.assert_allclose(nn.AdaptiveAvgPool2d((1, 1))(X), X.mean(axis=(2, 3), keepdims=True))
    nn.AdaptiveAvgPool2d(1)
    with pytest.raises(AssertionError, match='not 1 x 1'):
        nn.AdaptiveAvgPool2d((2, 2))


@pytest.mark.parametrize('layer, shape, axes', [(nn.BatchNorm2d, (6, 3, 4, 4), (0, 2, 3)), (nn.BatchNorm1d, (8, 3), (0,)), (nn.BatchNorm1d, (8, 3, 5), (0, 2))])
def test_batch_norm_normalizes_in_training_and_uses_running_statistics_in_evaluation(layer, shape, axes):
    X = (np.random.randn(*shape) * 3 + 2).astype(np.float32)
    bn = layer(3)
    Y = bn(X)
    np.testing.assert_allclose(Y.mean(axis=axes), 0, atol=1e-5)
    np.testing.assert_allclose(Y.var(axis=axes), 1, rtol=1e-3)
    n = X.size // 3
    np.testing.assert_allclose(bn.running_mean, 0.1 * X.mean(axis=axes), rtol=1e-4)
    np.testing.assert_allclose(bn.running_var, 0.9 + 0.1 * X.var(axis=axes) * n / (n - 1), rtol=1e-4)
    assert bn.running_mean.dtype == bn.running_var.dtype == np.float32
    bn.eval()
    expected = (X - bn.running_mean.reshape((1, -1) + (1,) * (X.ndim - 2))) / np.sqrt(bn.running_var.reshape((1, -1) + (1,) * (X.ndim - 2)) + 1e-5)
    np.testing.assert_allclose(bn(X), expected, rtol=1e-4, atol=1e-5)
    before = bn.running_mean.copy()
    bn(X)
    assert (bn.running_mean == before).all()  # evaluation leaves the statistics alone
    with pytest.raises(AssertionError, match='dimensions'):
        nn.BatchNorm2d(3)(np.ones((2, 3)))


def test_batch_norm_buffers_are_not_parameters_and_its_gradients_are_right():
    bn = nn.BatchNorm2d(3)
    assert [n for n, _ in bn.named_parameters()] == ['weight', 'bias'] and len(bn.parameters()) == 2
    assert list(bn.state_dict()) == ['weight', 'bias', 'running_mean', 'running_var']
    X = np.random.randn(4, 3, 3, 3)
    bn.weight[:], bn.bias[:] = [1.5, 0.5, 2.0], [0.1, -0.2, 0.3]
    loss = lambda: (bn(X) ** 2 * np.arange(1, 4).reshape(1, 3, 1, 1)).sum()
    dw, db = d2l.param_grads(loss, bn, bn.parameters())
    assert (bn.running_mean != 0).all()  # the running statistics moved during the traced call, and stay moved
    bn.eval()
    for array, g in ((bn.weight, dw), (bn.bias, db)):
        bn.train()
        for k in range(3):
            h, saved = 1e-3, array[k]
            array[k] = saved + h
            up = float(loss())
            array[k] = saved - h
            down = float(loss())
            array[k] = saved
            np.testing.assert_allclose(g[k], (up - down) / (2 * h), rtol=1e-2)
    from autograd import grad
    Z, W = np.random.randn(3, 2, 2, 2), np.random.randn(3, 2, 2, 2)  # the gradient with respect to the input, through the batch statistics
    f = lambda Z: (nn.BatchNorm2d(2)(Z) * W).sum()
    gZ, h = grad(f)(Z), 1e-5
    for index in [(0, 0, 0, 0), (2, 1, 1, 0)]:
        up, down = Z.copy(), Z.copy()
        up[index] += h
        down[index] -= h
        np.testing.assert_allclose(gZ[index], (f(up) - f(down)) / (2 * h), rtol=1e-4)


def test_lazy_batch_norm_takes_its_size_from_the_first_input_and_works_in_a_sequential():
    net = nn.Sequential(nn.LazyConv2d(4, 3), nn.LazyBatchNorm2d(), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.LazyLinear(2), nn.LazyBatchNorm1d())
    X = np.random.rand(5, 1, 6, 6).astype(np.float32)
    assert net.forward(X).shape == (5, 2)
    assert type(net[1]) is nn.BatchNorm2d and type(net[6]) is nn.BatchNorm1d and net[1].weight.shape == (4,) and net[1].weight.dtype == np.float32
    assert len(net.parameters()) == 2 + 2 + 2 + 2
    net.eval()
    assert not net[1].training and net.forward(X).shape == (5, 2)
    lazy = nn.LazyBatchNorm1d(eps=1e-3, momentum=0.5)
    lazy(np.arange(6).reshape(3, 2))  # integers: the layer is float64
    assert lazy.weight.dtype == np.float64 and (lazy.eps, lazy.momentum) == (1e-3, 0.5)


def test_a_module_can_list_its_own_buffers():
    class Counter(nn.Module):
        buffers = ('seen',)

        def __init__(self):
            self.scale, self.seen = np.ones(2, np.float32), np.zeros(2, np.float32)
    layer = Counter()
    assert len(layer.parameters()) == 1 and [n for n, _ in layer.named_parameters()] == ['scale'] and set(layer.state_dict()) == {'scale', 'seen'}


def test_layer_summary_prints_each_layer_and_creates_the_lazy_parameters(capsys):
    model = d2l.Classifier()
    model.net = nn.Sequential(nn.LazyConv2d(2, 3), nn.ReLU(), nn.Flatten(), nn.LazyLinear(3))
    model.layer_summary((1, 1, 6, 6))
    assert capsys.readouterr().out.splitlines() == ['Conv2d output shape:\t (1, 2, 4, 4)', 'ReLU output shape:\t (1, 2, 4, 4)',
                                                     'Flatten output shape:\t (1, 32)', 'Linear output shape:\t (1, 3)']
    assert model.net[0].weight.dtype == np.float32


def test_init_cnn_sets_xavier_weights_of_linear_and_convolutional_layers_only():
    conv, linear, bn = nn.Conv2d(2, 3, 3), nn.Linear(4, 5), nn.BatchNorm2d(3)
    before = bn.weight.copy()
    for module in (conv, linear, bn, nn.ReLU()):
        d2l.init_cnn(module)
    assert np.abs(conv.weight).max() <= np.sqrt(6 / (2 * 9 + 3 * 9)) and np.abs(linear.weight).max() <= np.sqrt(6 / (4 + 5))
    assert (bn.weight == before).all()


def test_resnext_block_keeps_the_shape_or_halves_it_with_a_strided_projection():
    X = np.random.rand(2, 8, 6, 6).astype(np.float32)
    block = d2l.ResNeXtBlock(8, groups=2, bot_mul=1)
    assert block(X).shape == (2, 8, 6, 6) and block.conv4 is None
    assert block.conv2.groups == 4 and block.conv2.weight.shape == (8, 2, 3, 3)  # bot_channels // groups groups, each seeing 2 channels
    projected = d2l.ResNeXtBlock(12, groups=2, bot_mul=0.5, use_1x1conv=True, strides=2)
    assert projected(X).shape == (2, 12, 3, 3) and projected.conv2.weight.shape == (6, 2, 3, 3)
    grads = d2l.param_grads(lambda: (projected(X) ** 2).sum(), projected, projected.parameters())
    assert all(g.shape == p.shape for g, p in zip(grads, projected.parameters())) and np.abs(grads[0]).sum() > 0


def test_optimizer_groups_state_and_the_learning_rate_property():
    w, b = np.zeros(2, np.float32), np.zeros(1, np.float32)
    opt = optim.SGD([{'params': w, 'lr': 0.5}, {'params': [b]}], lr=0.1, momentum=0.9, weight_decay=0.1)
    assert [g['lr'] for g in opt.param_groups] == [0.5, 0.1] and opt.lr == 0.5 and opt.params == [w, b] and opt.state == [{}, {}]
    opt.lr = 0.2
    assert [g['lr'] for g in opt.param_groups] == [0.2, 0.2]
    with pytest.raises(NotImplementedError):
        type('Bad', (optim.Optimizer,), {})([w], weight_decay=0).step([np.ones(2)])


def test_sgd_with_momentum_uses_a_velocity_that_starts_as_the_first_gradient():
    w = np.array([1.0, -1.0], np.float32)
    opt = optim.SGD([w], lr=0.1, momentum=0.9)
    g1, g2 = np.array([1.0, 2.0], np.float32), np.array([0.5, -1.0], np.float32)
    opt.step([g1])
    np.testing.assert_allclose(w, [0.9, -1.2], rtol=1e-6)  # w - lr * g1
    opt.step([g2])
    np.testing.assert_allclose(w, np.array([0.9, -1.2]) - 0.1 * (0.9 * g1 + g2), rtol=1e-6)  # w - lr * (momentum * v + g2)


def _reference_steps(update, grads, w0):
    """Apply `update(w, g, state)` for each gradient, with plain NumPy float64."""
    w, state = np.array(w0, np.float64), {}
    for g in grads:
        update(w, np.asarray(g, np.float64), state)
    return w


def test_adaptive_optimizers_follow_pytorchs_formulas():
    grads = [np.array([0.3, -0.7], np.float32), np.array([0.1, 0.4], np.float32), np.array([-0.2, 0.2], np.float32)]
    w0 = np.array([1.0, -2.0])

    def adagrad(w, g, s):
        s['sum'] = s.get('sum', 0) + g ** 2
        w -= 0.1 * g / (np.sqrt(s['sum']) + 1e-10)

    def rmsprop(w, g, s):
        s['v'] = 0.9 * s.get('v', 0) + 0.1 * g ** 2
        w -= 0.01 * g / (np.sqrt(s['v']) + 1e-8)

    def adadelta(w, g, s):
        s['v'] = 0.9 * s.get('v', 0) + 0.1 * g ** 2
        delta = np.sqrt(s.get('d', 0) + 1e-6) / np.sqrt(s['v'] + 1e-6) * g
        s['d'] = 0.9 * s.get('d', 0) + 0.1 * delta ** 2
        w -= 1.0 * delta

    def adam(w, g, s):
        s['t'] = s.get('t', 0) + 1
        s['m'] = 0.9 * s.get('m', 0) + 0.1 * g
        s['v'] = 0.999 * s.get('v', 0) + 0.001 * g ** 2
        w -= 0.01 / (1 - 0.9 ** s['t']) * s['m'] / (np.sqrt(s['v']) / np.sqrt(1 - 0.999 ** s['t']) + 1e-8)

    for cls, kwargs, reference in [(optim.Adagrad, {'lr': 0.1}, adagrad), (optim.RMSprop, {'lr': 0.01, 'alpha': 0.9}, rmsprop),
                                   (optim.Adadelta, {'rho': 0.9}, adadelta), (optim.Adam, {'lr': 0.01}, adam)]:
        w = w0.astype(np.float32)
        opt = cls([w], **kwargs)
        for g in grads:
            opt.step([g])
        np.testing.assert_allclose(w, _reference_steps(reference, grads, w0), rtol=1e-4, err_msg=cls.__name__)
        assert w.dtype == np.float32
    w = w0.astype(np.float32)
    optim.Adam([w], lr=0.01, weight_decay=0.5).step([np.zeros(2, np.float32)])  # weight decay adds weight_decay * w to the gradient
    assert (np.abs(w - w0) > 0).all()


def test_multi_step_lr_multiplies_at_each_milestone():
    opt = optim.SGD([np.zeros(1, np.float32)], lr=0.5)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[2, 4, 4], gamma=0.5)
    seen = []
    for _ in range(5):
        seen.append(scheduler.get_last_lr()[0])
        scheduler.step()
    assert seen == [0.5, 0.5, 0.25, 0.25, 0.0625] and scheduler.get_last_lr() == [0.0625]  # the milestone 4 is listed twice


def test_mse_loss_reductions():
    a, b = np.array([1.0, 3.0]), np.array([0.0, 1.0])
    assert nn.MSELoss()(a, b) == 2.5 and nn.MSELoss('sum')(a, b) == 5.0 and nn.MSELoss(reduction='none')(a, b).tolist() == [1.0, 4.0]
    with pytest.raises(AssertionError, match='unknown reduction'):
        nn.MSELoss('max')


def test_timer_and_accumulator():
    timer = d2l.Timer()
    first = timer.stop()
    timer.start()
    second = timer.stop()
    assert timer.times == [first, second] and timer.sum() == first + second and timer.avg() == (first + second) / 2
    assert timer.cumsum() == [first, first + second]
    acc = d2l.Accumulator(2)
    acc.add(1, np.float32(2.5))
    acc.add(3, 4)
    assert (acc[0], acc[1]) == (4.0, 6.5)
    acc.reset()
    assert acc.data == [0.0, 0.0]


def test_linreg_squared_loss_evaluate_loss_and_accuracy():
    X, w, b = np.arange(6.0).reshape(3, 2), np.array([[1.0], [2.0]]), np.array([0.5])
    y_hat = d2l.linreg(X, w, b)
    assert y_hat.tolist() == [[2.5], [8.5], [14.5]]
    np.testing.assert_allclose(d2l.squared_loss(y_hat, np.array([2.5, 6.5, 14.5])), [[0.0], [2.0], [0.0]])
    assert d2l.evaluate_loss(lambda X: d2l.linreg(X, w, b), [(X, np.array([2.5, 6.5, 14.5]))], d2l.squared_loss) == 2 / 3
    assert d2l.accuracy(np.array([[0.1, 0.9], [0.8, 0.2]]), np.array([1, 1])) == 1.0 and d2l.accuracy(np.array([1, 0]), np.array([1, 1])) == 1.0


def test_evaluate_accuracy_puts_the_model_in_evaluation_mode():
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    model[1].weight[:], model[1].bias[:] = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32), 0
    batches = [(np.eye(4, dtype=np.float32)[[0, 1]], np.array([0, 1])), (np.eye(4, dtype=np.float32)[[1, 0]], np.array([0, 1]))]
    assert d2l.evaluate_accuracy_gpu(model, batches) == 0.5 and not model.training
    assert d2l.evaluate_accuracy_gpu(lambda X: np.array([[1.0, 0.0]] * len(X)), [(np.zeros((3, 1)), np.array([0, 0, 1]))]) == 2 / 3


def test_annotate_and_the_animator_plot_points_and_show_them_only_in_jupyter(monkeypatch):
    import sys
    from divelet_ml import plotting
    d2l.plt.figure()
    d2l.annotate('here', (0.5, 0.5), (0.1, 0.1))
    assert [t.get_text() for t in d2l.plt.gca().texts] == ['here']
    shown = []
    monkeypatch.setattr(plotting, '_show', shown.append)
    animator = d2l.Animator(xlabel='epoch', legend=['a', 'b'], xlim=[0, 3], ylim=[0, 3])
    animator.add(1, (1.0, None))
    animator.add([2, 2], [2.0, 3.0])
    animator.add(3, 2.5)  # a scalar is one line
    assert animator.X == [[1, 2, 3], [2]] and animator.Y == [[1.0, 2.0, 2.5], [3.0]] and len(shown) == 3
    d2l.plt.close('all')
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, 'IPython', None)  # no IPython: nothing is shown
    plotting._show(d2l.plt.figure())
    monkeypatch.undo()
    import IPython
    monkeypatch.setattr(IPython, 'get_ipython', lambda: None)  # IPython but no shell
    plotting._show(d2l.plt.figure())
    calls = []
    monkeypatch.setattr(IPython, 'get_ipython', lambda: object())
    monkeypatch.setattr(IPython.display, 'display', lambda fig: calls.append('display'))
    monkeypatch.setattr(IPython.display, 'clear_output', lambda wait: calls.append('clear'))
    plotting._show(d2l.plt.figure())
    assert calls == ['display', 'clear']
    d2l.plt.close('all')


def test_airfoil_data_comes_from_the_site_in_jupyterlite_and_is_checked(tmp_path, monkeypatch):
    from divelet_ml import data
    text = b'800\t0\t0.3048\t71.3\t0.0026\t126.2\n1000\t0\t0.3048\t71.3\t0.0026\t125.2\n'
    monkeypatch.setattr(data, 'AIRFOIL_SHA1', hashlib.sha1(text).hexdigest())
    monkeypatch.setattr(data, '_site_url', lambda: 'https://site')
    monkeypatch.setattr(data, '_read_url', lambda url: text if url == 'https://site/data/airfoil/airfoil_self_noise.dat' else pytest.fail(url))
    assert data.airfoil_self_noise() == text
    monkeypatch.setattr(data, 'AIRFOIL_SHA1', '0' * 40)
    with pytest.raises(AssertionError, match='SHA-1'):
        data.airfoil_self_noise()
    monkeypatch.setattr(data, '_site_url', lambda: None)  # outside JupyterLite: downloaded to `root`, with the book's hash
    calls = []
    (tmp_path / 'airfoil_self_noise.dat').write_bytes(text)
    monkeypatch.setattr(data, 'download', lambda url, folder, sha1_hash: calls.append((url, folder, sha1_hash)) or str(tmp_path / 'airfoil_self_noise.dat'))
    assert data.airfoil_self_noise(str(tmp_path)) == text and calls == [(d2l.DATA_URL + 'airfoil_self_noise.dat', str(tmp_path), '0' * 40)]


def test_get_data_ch11_standardizes_and_load_array_batches(monkeypatch):
    from divelet_ml import data
    rows = np.random.rand(30, 6) * 100
    monkeypatch.setattr(data, 'airfoil_self_noise', lambda root='../data': '\n'.join('\t'.join(str(v) for v in r) for r in rows).encode())
    data_iter, features = d2l.get_data_ch11(batch_size=8, n=20)
    batches = list(data_iter)
    assert features == 5 and len(data_iter) == 3 and batches[0][0].shape == (8, 5) and batches[0][1].shape == (8,) and batches[0][0].dtype == np.float32
    allX = np.concatenate([X for X, _ in batches])
    expected = (rows - rows.mean(axis=0)) / rows.std(axis=0)  # the statistics of all 30 rows, as in the book
    np.testing.assert_allclose(sorted(allX[:, 0]), sorted(expected[:20, 0]), atol=1e-5)
    assert not d2l.load_array((np.arange(4), np.arange(4)), 2, is_train=False).shuffle


def test_load_data_fashion_mnist_gives_training_and_validation_loaders(monkeypatch):
    from divelet_ml import data
    sets = [data.ArrayDataset(np.zeros((5, 1, 4, 4), np.float32), np.arange(5)), data.ArrayDataset(np.ones((3, 1, 4, 4), np.float32), np.arange(3))]
    seen = []
    monkeypatch.setattr(data, 'load_fashion_mnist', lambda root, resize: seen.append((root, resize)) or sets)
    train, val = d2l.load_data_fashion_mnist(2, resize=(4, 4))
    assert seen == [('../data', (4, 4))] and len(train) == 3 and len(val) == 2 and train.shuffle and not val.shuffle
    d2l.load_data_fashion_mnist(2)
    assert seen[-1] == ('../data', (28, 28))
    train, val = d2l.load_data_fashion_mnist(2, num_train=3, num_val=1)
    assert len(train) == 2 and len(val) == 1 and train.arrays[0].shape[0] == 3 and val.arrays[1].shape == (1,)


def test_train_2d_and_show_trace_2d():
    d2l.plt.close('all')

    def gd(x1, x2, s1, s2):
        return x1 - 0.1 * x1, x2 - 0.1 * x2, 0, 0
    results = d2l.train_2d(gd, steps=3)
    assert results[0] == (-5, -2) and np.allclose(results[-1], (-5 * 0.9 ** 3, -2 * 0.9 ** 3))
    def with_gradient(x1, x2, s1, s2, f_grad):
        g1, g2 = f_grad(x1, x2)
        return x1 - g1, x2 - g2, 0, 0
    assert d2l.train_2d(with_gradient, steps=2, f_grad=lambda x1, x2: (0.5 * x1, 0.5 * x2))[-1] == (-5 * 0.25, -2 * 0.25)
    d2l.show_trace_2d(lambda x1, x2: x1 ** 2 + 2 * x2 ** 2, results)
    assert len(d2l.plt.gca().lines) == 1 and d2l.plt.gca().get_xlabel() == 'x1'
    d2l.plt.close('all')


def _quiet_animator(monkeypatch):
    from divelet_ml import plotting
    monkeypatch.setattr(plotting, '_show', lambda fig: None)


def test_train_ch11_updates_the_parameters_with_the_trainers_gradients(monkeypatch):
    _quiet_animator(monkeypatch)
    rng = np.random.RandomState(0)
    X = rng.randn(200, 5).astype(np.float32)
    y = (X @ np.array([1, -2, 0.5, 0, 3], np.float32)).astype(np.float32)
    data_iter = d2l.load_array((X, y), 10)
    seen = []

    def sgd(params, grads, states, hyperparams):
        seen.append((len(params), states, hyperparams))
        for p, g in zip(params, grads):
            p -= hyperparams['lr'] * g
    times, losses = d2l.train_ch11(sgd, 'state', {'lr': 0.1}, data_iter, 5, num_epochs=2)
    assert seen[0] == (2, 'state', {'lr': 0.1}) and len(seen) == 40 and len(times) == len(losses) == 2 and losses[-1] < losses[0]  # a point every 200 examples


@pytest.mark.parametrize('trainer_fn, hyperparams', [(optim.SGD, {'lr': 0.05, 'momentum': 0.5}), (optim.Adam, {'lr': 0.05})])
def test_train_concise_ch11_trains_a_linear_layer_with_an_optimizer(monkeypatch, trainer_fn, hyperparams):
    _quiet_animator(monkeypatch)
    rng = np.random.RandomState(0)
    X = rng.randn(200, 5).astype(np.float32)
    y = (X @ np.array([1, -2, 0.5, 0, 3], np.float32)).astype(np.float32)
    animators = []
    original = d2l.Animator
    monkeypatch.setattr('divelet_ml.optimization.Animator', lambda **kw: animators.append(original(**kw)) or animators[-1])
    d2l.train_concise_ch11(trainer_fn, hyperparams, d2l.load_array((X, y), 10), num_epochs=3)
    assert animators[0].Y[0][-1] < animators[0].Y[0][0]


# ---- Chapter 10: nn.Linear on sequences, nn.Embedding, nn.LSTM, nn.GRU, the data and the encoder--decoder

def test_linear_flattens_leading_dimensions_and_keeps_its_gradients():
    linear = nn.Linear(3, 2)
    X = np.random.rand(4, 5, 3).astype(np.float32)
    np.testing.assert_allclose(linear(X), X @ linear.weight.T + linear.bias, rtol=1e-5, atol=1e-6)
    assert linear(X).shape == (4, 5, 2) and linear(X[0, 0]).shape == (2,) and linear(X[0]).shape == (5, 2)
    grads = d2l.param_grads(lambda: (linear(X) ** 2).sum(), linear, linear.parameters())
    Y = X @ linear.weight.T + linear.bias
    np.testing.assert_allclose(grads[0], 2 * np.einsum('bto,bti->oi', Y, X), rtol=1e-4)
    np.testing.assert_allclose(grads[1], 2 * Y.sum((0, 1)), rtol=1e-4)
    assert nn.Linear(3, 2, bias=False)(X).shape == (4, 5, 2)


def test_embedding_looks_up_rows_and_has_a_dense_gradient():
    emb = nn.Embedding(6, 4)
    assert emb.weight.dtype == np.float32 and emb.weight.shape == (6, 4) and len(emb.parameters()) == 1
    X = np.array([[0, 2, 2], [5, 0, 1]])
    out = emb(X)
    assert out.shape == (2, 3, 4) and np.array_equal(out[0, 1], emb.weight[2])
    grad, = d2l.param_grads(lambda: emb(X).sum(), emb, emb.parameters())
    counts = np.bincount(X.reshape(-1), minlength=6)
    np.testing.assert_allclose(grad, np.repeat(counts[:, None], 4, 1))  # each row's gradient is how often it was used
    assert emb(X.astype(np.int32)).shape == (2, 3, 4)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_nn_lstm_matches_a_loop_with_pytorchs_gates_and_state_shapes():
    lstm = nn.LSTM(3, 4, num_layers=2)
    assert len(lstm.parameters()) == 8 and all(p.dtype == np.float32 for p in lstm.parameters())
    assert lstm.weight_ih_l0.shape == (16, 3) and lstm.weight_ih_l1.shape == (16, 4)
    X = np.random.rand(5, 2, 3).astype(np.float32)
    outputs, (H, C) = lstm(X)
    assert outputs.shape == (5, 2, 4) and H.shape == C.shape == (2, 2, 4)
    x = X
    for k in range(2):
        W_ih, W_hh, b_ih, b_hh = (getattr(lstm, f'{n}_l{k}') for n in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'))
        h = c = np.zeros((2, 4), np.float32)
        out = []
        for t in range(5):
            g = x[t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh
            i, f, gg, o = _sigmoid(g[:, :4]), _sigmoid(g[:, 4:8]), np.tanh(g[:, 8:12]), _sigmoid(g[:, 12:])
            c = f * c + i * gg
            h = o * np.tanh(c)
            out.append(h)
        x = np.stack(out)
        np.testing.assert_allclose(H[k], h, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(C[k], c, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(outputs, x, rtol=1e-5, atol=1e-6)
    _, middle = lstm(X[:3])  # continuing from a state is the same as running all the steps at once
    np.testing.assert_allclose(lstm(X[3:], middle)[0], outputs[3:], rtol=1e-5, atol=1e-6)


def test_nn_gru_matches_a_loop_with_pytorchs_gates_and_state_shapes():
    gru = nn.GRU(3, 4, num_layers=2)
    assert len(gru.parameters()) == 8 and gru.weight_ih_l0.shape == (12, 3) and gru.weight_hh_l1.shape == (12, 4)
    X = np.random.rand(5, 2, 3).astype(np.float32)
    outputs, H = gru(X)
    assert outputs.shape == (5, 2, 4) and H.shape == (2, 2, 4)
    x = X
    for k in range(2):
        W_ih, W_hh, b_ih, b_hh = (getattr(gru, f'{n}_l{k}') for n in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'))
        h = np.zeros((2, 4), np.float32)
        out = []
        for t in range(5):
            gi, gh = x[t] @ W_ih.T + b_ih, h @ W_hh.T + b_hh
            r, z = _sigmoid(gi[:, :4] + gh[:, :4]), _sigmoid(gi[:, 4:8] + gh[:, 4:8])
            n = np.tanh(gi[:, 8:] + r * gh[:, 8:])  # the reset gate multiplies the hidden part only, as in PyTorch
            h = (1 - z) * n + z * h
            out.append(h)
        x = np.stack(out)
        np.testing.assert_allclose(H[k], h, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(outputs, x, rtol=1e-5, atol=1e-6)
    _, middle = gru(X[:3])
    np.testing.assert_allclose(gru(X[3:], middle)[0], outputs[3:], rtol=1e-5, atol=1e-6)


def test_nn_gru_and_lstm_gradients_match_finite_differences():
    for rnn in (nn.GRU(2, 3, num_layers=2), nn.LSTM(2, 3, num_layers=2)):
        rnn.weight_hh_l0, rnn.weight_ih_l1 = rnn.weight_hh_l0.astype(np.float64), rnn.weight_ih_l1.astype(np.float64)
        for name in rnn._flat_weights_names:
            setattr(rnn, name, getattr(rnn, name).astype(np.float64))
        X = np.random.rand(4, 2, 2)
        f = lambda: (rnn(X)[0] ** 2).sum()
        grads = d2l.param_grads(f, rnn, rnn.parameters())
        for g, p in zip(grads, rnn.parameters()):
            i = (0,) * p.ndim
            old = p[i]
            p[i] = old + 1e-6
            up = f()
            p[i] = old - 1e-6
            down = f()
            p[i] = old
            np.testing.assert_allclose(g[i], (up - down) / 2e-6, rtol=1e-4, atol=1e-8)


def test_nn_gru_dropout_acts_between_layers_in_training_mode_only():
    gru = nn.GRU(3, 4, num_layers=2, dropout=1.0)
    X = np.random.rand(5, 2, 3).astype(np.float32)
    gru.eval()
    assert np.abs(gru(X)[0]).sum() > 0 and np.array_equal(gru(X)[0], gru(X)[0])
    reference, _ = nn.GRU._layer(gru, 1, np.zeros((5, 2, 4), np.float32), None)
    gru.train()
    np.testing.assert_allclose(gru(X)[0], reference, rtol=1e-6)  # dropout of 1 gives the second layer a zero input
    assert not np.allclose(gru.eval()(X)[0], reference)
    one = nn.GRU(3, 4, num_layers=1, dropout=1.0)
    np.testing.assert_allclose(one.train()(X)[0], one.eval()(X)[0])  # not after the last layer


def test_extract_unpacks_zip_and_tar_files(tmp_path):
    import tarfile
    import zipfile
    (tmp_path / 'a.txt').write_text('hello')
    with zipfile.ZipFile(tmp_path / 'a.zip', 'w') as z:
        z.write(tmp_path / 'a.txt', 'folder/a.txt')
    d2l.extract(str(tmp_path / 'a.zip'))
    assert (tmp_path / 'folder' / 'a.txt').read_text() == 'hello'
    with tarfile.open(tmp_path / 'b.tar', 'w') as t:
        t.add(tmp_path / 'a.txt', 'b/a.txt')
    out = tmp_path / 'out'
    d2l.extract(str(tmp_path / 'b.tar'), str(out))
    assert (out / 'b' / 'a.txt').read_text() == 'hello'
    with pytest.raises(AssertionError, match='zip/tar'):
        d2l.extract(str(tmp_path / 'a.txt'))


def _fra_zip(text):
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        z.writestr('fra-eng/fra.txt', text)
    return buffer.getvalue()


def test_fra_eng_text_comes_from_the_site_in_jupyterlite_and_is_checked(monkeypatch):
    from divelet_ml import data
    blob = _fra_zip('Go.\tVa !\n')
    monkeypatch.setattr(data, '_site_url', lambda: 'https://site')
    monkeypatch.setattr(data, '_read_url', lambda url: blob if url == 'https://site/data/fra-eng/fra-eng.zip' else pytest.fail(url))
    monkeypatch.setattr(data, 'FRA_ENG_SHA1', hashlib.sha1(blob).hexdigest())
    assert data.fra_eng_text('unused') == 'Go.\tVa !\n'
    monkeypatch.setattr(data, 'FRA_ENG_SHA1', '0' * 40)
    with pytest.raises(AssertionError, match='SHA-1'):
        data.fra_eng_text('unused')


def test_fra_eng_text_is_downloaded_and_extracted_outside_jupyterlite(tmp_path, monkeypatch):
    from divelet_ml import data
    (tmp_path / 'fra-eng.zip').write_bytes(_fra_zip('Hi.\tSalut !\n'))
    calls = []
    monkeypatch.setattr(data, '_site_url', lambda: None)
    monkeypatch.setattr(data, 'download', lambda url, folder, sha1_hash: calls.append((url, folder, sha1_hash)) or str(tmp_path / 'fra-eng.zip'))
    assert data.fra_eng_text(str(tmp_path)) == 'Hi.\tSalut !\n'
    assert calls == [(d2l.DATA_URL + 'fra-eng.zip', str(tmp_path), data.FRA_ENG_SHA1)]


FRA = 'Go.\tVa !\nGo.\tAllez !\nHi.\tSalut !\nRun!\tCours !\nRun!\tCourez !\nWho?\tQui ?\nWow!\tÇa alors !\nFire!\tAu feu !\n'


def test_mtfraeng_arrays_vocabularies_and_batches(monkeypatch):
    from divelet_ml import data
    monkeypatch.setattr(data, 'fra_eng_text', lambda root: FRA)
    mt = d2l.MTFraEng(batch_size=2, num_steps=4, num_train=4, num_val=2)
    src, tgt, src_valid_len, label = mt.arrays
    assert src.shape == tgt.shape == label.shape == (7, 4) and src_valid_len.shape == (7,)
    assert mt._preprocess('Go. now!').startswith('go . now !') and mt._tokenize('a b\tc d\n\ne', None)[0] == [['a', 'b', '<eos>']]
    tgt_tokens = mt.tgt_vocab.to_tokens(tgt[0])
    assert tgt_tokens[0] == '<bos>' and mt.tgt_vocab.to_tokens(label[0])[-1] == '<pad>'  # the decoder's input starts with <bos>, the label is shifted
    assert np.array_equal(label[0][:-1], tgt[0][1:])
    assert src_valid_len[0] == 3 and mt.src_vocab.to_tokens(src[0])[:3] == ['go', '.', '<eos>'] and mt.tgt_vocab.to_tokens(tgt[0])[1] == '<unk>'  # 'va' occurs once
    batches = list(mt.train_dataloader())
    assert len(batches) == 2 and len(batches[0]) == 4 and batches[0][0].shape == (2, 4) and len(mt.val_dataloader()) == 2
    built = mt.build(['go .', 'zzz .'], ['va !', 'qui ?'])
    assert built[0].shape == (2, 4) and built[0][1][0] == mt.src_vocab['<unk>'] and built[2].tolist() == [3, 3]


def test_bleu_scores_overlap_with_a_brevity_penalty():
    assert d2l.bleu('il est calme', 'il est calme', k=2) == pytest.approx(1.0)
    assert d2l.bleu('a b', 'a b c d', k=2) == pytest.approx(np.exp(-1))  # all n-grams match, the prediction is half as long
    assert d2l.bleu('x y', 'a b', k=2) == 0


def test_init_seq2seq_initializes_linear_and_gru_weights_not_biases():
    gru = nn.GRU(3, 4, num_layers=2)
    biases = [getattr(gru, n).copy() for n in gru._flat_weights_names if 'bias' in n]
    d2l.init_seq2seq(gru)
    for name in gru._flat_weights_names:
        a = getattr(gru, name)
        if 'weight' in name:
            assert np.abs(a).max() <= np.sqrt(6 / (a.shape[0] + a.shape[1])) + 1e-6
    assert all(np.array_equal(b, getattr(gru, n)) for b, n in zip(biases, [n for n in gru._flat_weights_names if 'bias' in n]))
    linear = nn.Linear(3, 4)
    d2l.init_seq2seq(linear)
    assert np.abs(linear.weight).max() <= np.sqrt(6 / 7) + 1e-6
    d2l.init_seq2seq(nn.LazyLinear(4))  # a lazy layer has no weight yet and is left alone


class _Seq2SeqDecoder(d2l.Decoder):
    """The book's decoder of section 10.7."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        import autograd.numpy as anp
        self.anp = anp
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, X, state):
        anp = self.anp
        embs = self.embedding(X.T.astype(np.int32))
        enc_output, hidden_state = state
        context = anp.tile(enc_output[-1], (embs.shape[0], 1, 1))
        outputs, hidden_state = self.rnn(anp.concatenate((embs, context), -1), hidden_state)
        return anp.swapaxes(self.dense(outputs), 0, 1), [enc_output, hidden_state]


def test_seq2seq_shapes_loss_mask_training_and_greedy_translation():
    np.random.seed(0)
    vocab, embed, hidden, layers, batch, steps = 10, 8, 16, 2, 4, 9
    encoder = d2l.Seq2SeqEncoder(vocab, embed, hidden, layers)
    outputs, state = encoder(np.zeros((batch, steps)))
    assert outputs.shape == (steps, batch, hidden) and state.shape == (layers, batch, hidden)
    decoder = _Seq2SeqDecoder(vocab, embed, hidden, layers)
    dec_outputs, dec_state = decoder(np.zeros((batch, steps)), decoder.init_state(encoder(np.zeros((batch, steps)))))
    assert dec_outputs.shape == (batch, steps, vocab) and dec_state[1].shape == (layers, batch, hidden)
    # A copy task: the label is the source, which the model can learn in a few steps; the padding does not count in the loss.
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=0, lr=0.01)
    src = np.random.randint(1, vocab, (16, 5))
    tgt_in = np.concatenate([np.full((16, 1), 1), src[:, :-1]], 1)
    batch_ = (src, tgt_in, np.full(16, 5), src)
    model(*batch_[:-1])
    Y_hat = model(*batch_[:-1])
    padded = src.copy()
    padded[:, 3:] = 0
    l = model.loss(Y_hat, padded)
    ce = d2l.Classifier.loss(model, Y_hat, padded, averaged=False)
    np.testing.assert_allclose(l, (ce * (padded.reshape(-1) != 0)).sum() / (padded != 0).sum(), rtol=1e-5)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, d2l.optim.Adam) and len(optimizer.params) == len(model.parameters())
    first = None
    for _ in range(60):
        grads = d2l.param_grads(lambda: model.loss(model(*batch_[:-1]), batch_[-1]), model, optimizer.params)
        first = first if first is not None else float(model.loss(model(*batch_[:-1]), batch_[-1]))
        optimizer.step(grads)
    model.eval()
    assert float(model.loss(model(*batch_[:-1]), batch_[-1])) < first / 2
    preds, weights = model.predict_step(batch_, 5)
    assert preds.shape == (16, 5) and weights == []
    assert (preds == src).mean() > 0.5  # greedy decoding from <bos> reproduces most of the copy


def test_show_list_len_pair_hist_draws_two_histograms():
    d2l.plt.close('all')
    d2l.show_list_len_pair_hist(['source', 'target'], 'tokens', 'count', [[1], [1, 2]], [[1, 2, 3], [1]])
    assert d2l.plt.gca().get_xlabel() == 'tokens' and len(d2l.plt.gca().patches) > 0


def test_d2l_gru_wraps_a_multilayer_nn_gru():
    gru = d2l.GRU(3, 4, 2, dropout=0.1)
    outputs, state = gru(np.random.rand(5, 2, 3).astype(np.float32))
    assert outputs.shape == (5, 2, 4) and state.shape == (2, 2, 4) and gru.num_layers == 2 and gru.rnn.dropout.p == 0.1
