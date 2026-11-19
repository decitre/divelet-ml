"""Spike for the Chapter 9 plan: how long does one epoch of the book's 9.5 recurrent language model take in NumPy + autograd?

Sizes are the book's: a vocabulary of 28 characters, 32 hidden units, minibatches of 1024 sequences of 32 steps, 10,000 training and
5,000 validation sequences (10 and 5 minibatches), SGD with learning rate 1 and gradient clipping at 1. The token ids are random:
the cost does not depend on their values. Runs in CPython, or as the one code cell of bench.ipynb in either JupyterLite kernel.
"""
import time
import numpy as onp
import autograd.numpy as np
from autograd import grad

V, H, B, T = 28, 32, 1024, 32
NTRAIN, NVAL = 10000, 5000
rng = onp.random.RandomState(0)
f32 = lambda a: a.astype('float32')
params = [f32(rng.randn(V, H) * .01), f32(rng.randn(H, H) * .01), onp.zeros(H, 'float32'), f32(rng.randn(H, V) * .01), onp.zeros(V, 'float32')]
ids = rng.randint(0, V, (NTRAIN + NVAL, T + 1))
X, Y = ids[:, :-1], ids[:, 1:]

def forward(params, X, Y):
    Wxh, Whh, bh, Whq, bq = params
    onehot = f32(X.T[:, :, None] == onp.arange(V))                  # (T, B, V)
    h, outs = np.zeros((X.shape[0], H), 'float32'), []
    for t in range(T):
        h = np.tanh(np.dot(onehot[t], Wxh) + np.dot(h, Whh) + bh)
        outs.append(h)
    z = np.dot(np.concatenate(outs, axis=0), Whq) + bq              # (T*B, V)
    z = z - np.max(z, axis=1, keepdims=True)
    logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    return -np.mean(logp[onp.arange(z.shape[0]), Y.T.reshape(-1)])

grads = grad(forward)

def train_step(lr=1.0, clip=1.0):
    i = rng.randint(0, NTRAIN - B)
    g = grads(params, X[i:i + B], Y[i:i + B])
    norm = onp.sqrt(sum((gi ** 2).sum() for gi in g))
    for p, gi in zip(params, g): p -= lr * gi * min(1.0, clip / norm)

def epoch():
    t = time.perf_counter()
    for _ in range(-(-NTRAIN // B)): train_step()
    train = time.perf_counter() - t; t = time.perf_counter()
    for k in range(-(-NVAL // B)): forward(params, X[NTRAIN + k * B:NTRAIN + (k + 1) * B], Y[NTRAIN + k * B:NTRAIN + (k + 1) * B])
    return train, time.perf_counter() - t

train_step(); epoch()                                                # warm up
runs = [epoch() for _ in range(2)]
train, val = min(r[0] for r in runs), min(r[1] for r in runs)
t = time.perf_counter(); grads(params, X[:B], Y[:B]); step = time.perf_counter() - t
print(f'one training step (forward and backward, 1024 x 32): {step:.2f} s')
print(f'one epoch: training {train:.1f} s + validation {val:.1f} s = {train + val:.1f} s; the book trains 100 epochs: {100 * (train + val) / 60:.1f} min')
