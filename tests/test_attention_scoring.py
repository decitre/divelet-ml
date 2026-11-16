"""Mathematical properties of the attention scoring functions defined in the 11.3 notebook.

The notebook's code cells run in this process, so the tests use its `masked_softmax`,
`DotProductAttention` and `AdditiveAttention` as written there.
"""

from pathlib import Path

import matplotlib
import nbformat
import numpy as np
import pytest

NOTEBOOK = (Path(__file__).parent.parent / "content" / "11_attention-mechanisms-and-transformers"
            / "11.03_attention-scoring-functions.ipynb")


@pytest.fixture(scope="module")
def ns():
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    namespace = {}
    for cell in nbformat.read(NOTEBOOK, as_version=4).cells:
        if cell.cell_type == "code":
            exec(cell.source, namespace)
    plt.close("all")
    return namespace


def test_masked_softmax(ns):
    X = np.random.rand(2, 3, 5)
    original = X.copy()
    for valid_lens, lengths in ((np.array([2, 4]), [[2] * 3, [4] * 3]),
                                (np.array([[1, 3, 5], [2, 4, 5]]), [[1, 3, 5], [2, 4, 5]])):
        W = ns["masked_softmax"](X, valid_lens)
        assert np.allclose(W.sum(axis=-1), 1)
        for b in range(2):
            for q, n in enumerate(lengths[b]):
                assert (W[b, q, n:] == 0).all() and (W[b, q, :n] > 0).all()
                assert np.allclose(W[b, q, :n], np.exp(X[b, q, :n]) / np.exp(X[b, q, :n]).sum())
    assert (X == original).all()  # the input is not modified
    assert np.allclose(ns["masked_softmax"](X, None).sum(axis=-1), 1)


def test_dot_product_attention(ns):
    queries = np.random.normal(0, 1, (2, 1, 3))
    keys = np.random.normal(0, 1, (2, 10, 3))
    values = np.random.normal(0, 1, (2, 10, 4))
    valid_lens = np.array([2, 6])
    attention = ns["DotProductAttention"](dropout=0.5).eval()
    out = attention(queries, keys, values, valid_lens)
    scores = queries @ keys.swapaxes(1, 2) / np.sqrt(3)  # (11.3.2)
    expected = ns["masked_softmax"](scores, valid_lens)
    assert np.allclose(attention.attention_weights, expected)
    assert np.allclose(out, expected @ values)  # (11.3.6): a weighted average of the values
    # Identical keys get identical weights: the output is the mean of the valid values.
    same_keys = np.repeat(keys[:, :1], 10, axis=1)
    out = attention(queries, same_keys, values, valid_lens)
    assert np.allclose(out[0, 0], values[0, :2].mean(axis=0)) and np.allclose(out[1, 0], values[1, :6].mean(axis=0))


def test_dot_product_attention_scaling(ns):
    # Scaling by 1/sqrt(d) keeps the variance of the scores near 1, whatever d is.
    d = 256
    q = np.random.normal(0, 1, (1, 2000, d))
    k = np.random.normal(0, 1, (1, 2000, d))
    scores = (q * k).sum(axis=-1) / np.sqrt(d)
    assert 0.85 < scores.var() < 1.15


def test_additive_attention(ns):
    queries = np.random.normal(0, 1, (2, 1, 20))
    keys = np.random.normal(0, 1, (2, 10, 2))
    values = np.random.normal(0, 1, (2, 10, 4))
    valid_lens = np.array([2, 6])
    attention = ns["AdditiveAttention"](num_hiddens=8, dropout=0.1).eval()
    out = attention(queries, keys, values, valid_lens)
    W_q, W_k, w_v = attention.W_q.weight, attention.W_k.weight, attention.w_v.weight
    assert (W_q.shape, W_k.shape, w_v.shape) == ((8, 20), (8, 2), (1, 8))
    # (11.3.7): a(q, k) = w_v^T tanh(W_q q + W_k k), for every query and key
    scores = np.array([[[(w_v @ np.tanh(W_q @ queries[b, i] + W_k @ keys[b, j]))[0]
                         for j in range(10)] for i in range(1)] for b in range(2)])
    expected = ns["masked_softmax"](scores, valid_lens)
    assert np.allclose(attention.attention_weights, expected)
    assert np.allclose(out, expected @ values)


def test_dropout_only_in_training(ns):
    queries, keys, values = np.ones((1, 1, 2)), np.ones((1, 4, 2)), np.ones((1, 4, 3))
    attention = ns["DotProductAttention"](dropout=0.5)
    assert np.allclose(attention.eval()(queries, keys, values), 1)
    outs = {round(float(attention.train()(queries, keys, values)[0, 0, 0]), 6) for _ in range(50)}
    assert len(outs) > 1  # dropout zeroes random weights and rescales the others
