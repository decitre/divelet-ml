"""Spike: time one epoch of the book's character language model (9.5, 10.1 to 10.3) with a tanh RNN, an LSTM, a GRU and a two-layer GRU.
28 characters (one-hot), 32 hidden units, minibatches of 1024 sequences of 32 steps; 10 training minibatches (forward and backward) and 5
validation minibatches (forward) per epoch, as in `spikes/rnn-epoch`. Plain NumPy and autograd, so it runs in CPython and in JupyterLite."""
import sys, time
import numpy as np
import autograd, autograd.numpy as anp

rng = np.random.RandomState(0)
f32 = np.float32
V, H, B, T = 28, 32, 1024, 32
w = lambda *s: (rng.randn(*s) * 0.01).astype(f32)
sig = lambda x: 1 / (1 + anp.exp(-x))

def layer(kind, x, P, H):  # x: (T, B, d); returns outputs (T, B, H)
    Bn = x.shape[1]; h = anp.zeros((Bn, H), f32); c = anp.zeros((Bn, H), f32); outs = []
    for t in range(x.shape[0]):
        gx, gh = anp.dot(x[t], P[0]) + P[2], anp.dot(h, P[1])
        if kind == 'rnn':
            h = anp.tanh(gx + gh)
        elif kind == 'gru':
            z = sig(gx[:, :H] + gh[:, :H]); r = sig(gx[:, H:2 * H] + gh[:, H:2 * H])
            h = z * h + (1 - z) * anp.tanh(gx[:, 2 * H:] + r * gh[:, 2 * H:])
        else:
            g = gx + gh; i, f, o, cc = sig(g[:, :H]), sig(g[:, H:2 * H]), sig(g[:, 2 * H:3 * H]), anp.tanh(g[:, 3 * H:])
            c = f * c + i * cc; h = o * anp.tanh(c)
        outs.append(h)
    return anp.stack(outs)

def make(kind, L):
    k = {'rnn': 1, 'gru': 3, 'lstm': 4}[kind]
    P = [[w(V if l == 0 else H, k * H), w(H, k * H), np.zeros(k * H, f32)] for l in range(L)]
    return {'layers': P, 'out': w(H, V)}

def loss(P, kind, X, Y):
    x = anp.transpose(X, (1, 0, 2))
    for lp in P['layers']:
        x = layer(kind, x, lp, H)
    logits = anp.dot(anp.transpose(x, (1, 0, 2)), P['out'])
    z = logits - anp.max(logits, -1, keepdims=True); lse = anp.log(anp.sum(anp.exp(z), -1))
    return anp.mean(lse - anp.sum(z * Y, -1))

if __name__ == '__main__':
    where = 'browser' if sys.platform == 'emscripten' else 'CPython'
    print(f'{where}, NumPy {np.__version__}')
    X = np.eye(V, dtype=f32)[rng.randint(0, V, (B, T))]; Y = np.eye(V, dtype=f32)[rng.randint(0, V, (B, T))]
    for kind, L in (('rnn', 1), ('lstm', 1), ('gru', 1), ('gru', 2)):
        P = make(kind, L); g = autograd.grad(lambda P: loss(P, kind, X, Y)); g(P)
        t = time.perf_counter()
        for _ in range(10): g(P)
        for _ in range(5): loss(P, kind, X, Y)
        print(f'{kind:5s} {L} layer(s): {time.perf_counter() - t:5.2f} s per epoch', flush=True)
