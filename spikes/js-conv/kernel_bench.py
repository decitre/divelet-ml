"""Run inside the Pyodide worker of kernel_worker.js: one training step of Chapter 5's MLP and of 7.6's LeNet, op by op,
in NumPy and through TensorFlow.js (WebGL), copies included. Random data of the notebooks' shapes."""
import sys, time
sys.path.insert(0, '/pkg')
import numpy as onp
import js
from pyodide.ffi import to_js
from autograd import grad
from divelet_ml import nn

T = time.perf_counter
def timeit(f, n=3):
    f(); t = T()
    for _ in range(n): f()
    return (T() - t) / n * 1e3

def f32(*shape): return onp.random.rand(*shape).astype('float32') - .5

def jmm(a, b):
    r = js.mm(to_js(a.ravel()), to_js(a.shape), to_js(b.ravel()), to_js(b.shape)).to_py()
    return onp.asarray(r).reshape(a.shape[0], b.shape[1])

def jconv(x, w, pad):
    r = onp.asarray(js.convfb(to_js(x.ravel()), to_js(x.shape), to_js(w.ravel()), to_js(w.shape), pad > 0).to_py())
    n, o = x.shape[0], w.shape[0]; ho = x.shape[2] + 2 * pad - w.shape[2] + 1; wo = x.shape[3] + 2 * pad - w.shape[3] + 1
    sy, sx = n * o * ho * wo, x.size
    return r[:sy].reshape(n, o, ho, wo), r[sy:sy + sx].reshape(x.shape), r[sy + sx:].reshape(w.shape)

def np_conv(x, w, pad):
    return grad(lambda a, b: nn.conv2d(a, b, padding=pad).sum(), argnum=(0, 1))(x, w)

def row(name, np_ms, js_ms):
    print(f'  {name:34s} NumPy {np_ms:9.2f} ms   TF.js WebGL {js_ms:8.2f} ms   x{np_ms / js_ms:5.1f}')
    return np_ms, js_ms

print('overhead of one call (10x10 matmul):', f'NumPy {timeit(lambda: f32(10, 10) @ f32(10, 10), 20):.3f} ms, TF.js {timeit(lambda: jmm(f32(10, 10), f32(10, 10)), 20):.3f} ms')

def step(title, matmuls, convs):
    print(title); tot = [0.0, 0.0]
    for name, (m, k, n) in matmuls:
        a, b = f32(m, k), f32(k, n)
        r = row(f'matmul {name} ({m}x{k} @ {k}x{n})', timeit(lambda: a @ b), timeit(lambda: jmm(a, b)))
        tot[0] += r[0]; tot[1] += r[1]
    for name, xs, ws, pad in convs:
        x, w = f32(*xs), f32(*ws)
        r = row(f'conv {name}, forward+backward', timeit(lambda: np_conv(x, w, pad)), timeit(lambda: jconv(x, w, pad)))
        tot[0] += r[0]; tot[1] += r[1]
    row('sum of the ops (the rest stays NumPy)', *tot)

# Chapter 5's MLP, batch 256: forward X@W1, H@W2; backward X.T@dZ1, H.T@G, G@W2.T
step('5.2 MLP step', [('X@W1', (256, 784, 256)), ('H@W2', (256, 256, 10)), ('X.T@dZ1', (784, 256, 256)), ('H.T@G', (256, 256, 10)), ('G@W2.T', (256, 10, 256))], [])
# 7.6's LeNet, batch 128: two convolutions, then three linear layers (forward, and the two backward products each)
lin = [(f'{n} {i}', s) for n, (a, b) in [('fc1', (400, 120)), ('fc2', (120, 84)), ('fc3', (84, 10))] for i, s in
       enumerate([(128, a, b), (128, b, a), (a, 128, b)])]
step('7.6 LeNet step', lin, [('1 (1->6, 5x5, pad 2, 28x28)', (128, 1, 28, 28), (6, 1, 5, 5), 2), ('2 (6->16, 5x5, 14x14)', (128, 6, 14, 14), (16, 6, 5, 5), 0)])
