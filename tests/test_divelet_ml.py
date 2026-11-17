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
