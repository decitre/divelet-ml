import numpy as np
import pytest

import divelet_ml as d2l
from divelet_ml import nn


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
