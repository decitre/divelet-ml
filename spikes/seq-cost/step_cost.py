"""Spike: time one training step (forward and backward with autograd, plain NumPy) of the book's two translation models at the
book's sizes and at smaller ones. Self-contained (NumPy and autograd), so it runs the same in CPython and in both JupyterLite kernels.

    python step_cost.py            # prints one line per configuration
"""
import sys, time
import numpy as np
import autograd, autograd.numpy as anp

rng = np.random.RandomState(0)
f32 = np.float32
def w(*shape): return (rng.randn(*shape) * 0.05).astype(f32)

def softmax(x):
    e = anp.exp(x - anp.max(x, axis=-1, keepdims=True)); return e / anp.sum(e, axis=-1, keepdims=True)
def sigmoid(x): return 1 / (1 + anp.exp(-x))
def xent(logits, y):  # mean cross entropy over all positions
    z = logits - anp.max(logits, axis=-1, keepdims=True)
    lse = anp.log(anp.sum(anp.exp(z), axis=-1))
    n = y.size
    return anp.mean(lse - z.reshape(n, -1)[np.arange(n), y.reshape(-1)].reshape(y.shape))

# ---- GRU encoder-decoder (10.7): embedding, a stack of GRU layers, the decoder gets the encoder's last state as context
def gru_params(e, h, L, Vs, Vt):
    P = {'Es': w(Vs, e), 'Et': w(Vt, e), 'out': w(h, Vt)}
    for name, in_dims in (('enc', [e] + [h] * (L - 1)), ('dec', [e + h] + [h] * (L - 1))):
        for l, d in enumerate(in_dims):
            P[f'{name}{l}'] = [w(d, 3 * h), w(h, 3 * h), np.zeros(3 * h, f32)]  # gates z, r and candidate, fused
    return P

def gru_layer(x, h0, Wx, Wh, b, H):  # x: (T, B, d)
    h, outs = h0, []
    for t in range(x.shape[0]):
        gx, gh = anp.matmul(x[t], Wx) + b, anp.matmul(h, Wh)
        z = sigmoid(gx[:, :H] + gh[:, :H]); r = sigmoid(gx[:, H:2 * H] + gh[:, H:2 * H])
        c = anp.tanh(gx[:, 2 * H:] + r * gh[:, 2 * H:])
        h = z * h + (1 - z) * c
        outs.append(h)
    return anp.stack(outs), h

def gru_loss(P, src, tgt_in, tgt_out, L, H):
    B = src.shape[0]
    x = anp.transpose(P['Es'][src], (1, 0, 2))
    states = []
    for l in range(L):
        x, h = gru_layer(x, anp.zeros((B, H), f32), *P[f'enc{l}'], H); states.append(h)
    ctx = states[-1]
    y = anp.transpose(P['Et'][tgt_in], (1, 0, 2))
    y = anp.concatenate([y, anp.repeat(ctx[None], y.shape[0], axis=0)], axis=2)
    for l in range(L):
        y, _ = gru_layer(y, states[l], *P[f'dec{l}'], H)
    logits = anp.matmul(anp.transpose(y, (1, 0, 2)), P['out'])
    return xent(logits, tgt_out)

# ---- Transformer encoder-decoder (11.7): multi-head attention, position-wise network, add and layer norm
def ln(x): m = anp.mean(x, -1, keepdims=True); v = anp.mean((x - m) ** 2, -1, keepdims=True); return (x - m) / anp.sqrt(v + f32(1e-5))

def mha(q, kv, W, heads, causal):
    B, Tq, H = q.shape; Tk = kv.shape[1]; d = H // heads
    Q = anp.transpose(anp.matmul(q, W[0]).reshape(B, Tq, heads, d), (0, 2, 1, 3))
    K = anp.transpose(anp.matmul(kv, W[1]).reshape(B, Tk, heads, d), (0, 2, 1, 3))
    V = anp.transpose(anp.matmul(kv, W[2]).reshape(B, Tk, heads, d), (0, 2, 1, 3))
    s = anp.matmul(Q, anp.transpose(K, (0, 1, 3, 2))) / float(np.sqrt(d))
    if causal: s = s - f32(1e9) * np.triu(np.ones((Tq, Tk), f32), 1)
    o = anp.transpose(anp.matmul(softmax(s), V), (0, 2, 1, 3)).reshape(B, Tq, H)
    return anp.matmul(o, W[3])

def tf_params(H, ffn, blks, Vs, Vt):
    P = {'Es': w(Vs, H), 'Et': w(Vt, H), 'out': w(H, Vt), 'enc': [], 'dec': []}
    for _ in range(blks):
        P['enc'].append({'att': [w(H, H) for _ in range(4)], 'f1': w(H, ffn), 'f2': w(ffn, H)})
        P['dec'].append({'att1': [w(H, H) for _ in range(4)], 'att2': [w(H, H) for _ in range(4)], 'f1': w(H, ffn), 'f2': w(ffn, H)})
    return P

def tf_loss(P, src, tgt_in, tgt_out, H, heads):
    pe = lambda T: anp.array(np.sin(np.arange(T)[:, None] / 10000 ** (np.arange(H)[None] / H)), f32)
    x = P['Es'][src] * float(np.sqrt(H)) + pe(src.shape[1])
    for b in P['enc']:
        x = ln(x + mha(x, x, b['att'], heads, False)); x = ln(x + anp.matmul(anp.maximum(anp.matmul(x, b['f1']), 0), b['f2']))
    y = P['Et'][tgt_in] * float(np.sqrt(H)) + pe(tgt_in.shape[1])
    for b in P['dec']:
        y = ln(y + mha(y, y, b['att1'], heads, True)); y = ln(y + mha(y, x, b['att2'], heads, False))
        y = ln(y + anp.matmul(anp.maximum(anp.matmul(y, b['f1']), 0), b['f2']))
    return xent(anp.matmul(y, P['out']), tgt_out)

def time_step(kind, B, T, Vs, Vt, **cfg):
    src = rng.randint(0, Vs, (B, T)); tin = rng.randint(0, Vt, (B, T)); tout = rng.randint(0, Vt, (B, T))
    if kind == 'gru':
        P = gru_params(cfg['e'], cfg['h'], cfg['L'], Vs, Vt); f = lambda P: gru_loss(P, src, tin, tout, cfg['L'], cfg['h'])
    else:
        P = tf_params(cfg['h'], cfg['ffn'], cfg['blks'], Vs, Vt); f = lambda P: tf_loss(P, src, tin, tout, cfg['h'], cfg['heads'])
    g = autograd.grad(f); g(P)  # warm up
    best = 1e9
    for _ in range(2):
        t = time.perf_counter(); g(P); best = min(best, time.perf_counter() - t)
    return best

CONFIGS = [  # kind, batch, steps, source vocab, target vocab, sizes. The book's MTFraEng keeps 512 training and 128 validation pairs and
    # builds its vocabularies from them (min_freq 2): 195 and 213 tokens. Each kind: the book's sizes, then smaller ones.
    ('gru', 128, 9, 195, 213, dict(e=256, h=256, L=2)),
    ('gru', 128, 9, 195, 213, dict(e=128, h=128, L=2)),
    ('gru', 128, 9, 195, 213, dict(e=64, h=64, L=2)),
    ('transformer', 128, 9, 195, 213, dict(h=256, ffn=64, blks=2, heads=4)),
    ('transformer', 128, 9, 195, 213, dict(h=128, ffn=64, blks=2, heads=4)),
    ('transformer', 128, 9, 195, 213, dict(h=64, ffn=64, blks=2, heads=4)),
    # What the whole data set would cost (167,130 pairs; vocabularies of 10,012 and 17,851 tokens with min_freq 2).
    ('gru', 128, 9, 10012, 17851, dict(e=256, h=256, L=2)),
    ('transformer', 128, 9, 10012, 17851, dict(h=256, ffn=64, blks=2, heads=4)),
]
if __name__ == '__main__':
    where = 'browser' if sys.platform == 'emscripten' else 'CPython'
    print(f'{where}, NumPy {np.__version__}')
    for kind, B, T, Vs, Vt, cfg in CONFIGS:
        if sys.platform == 'emscripten' and Vt > 1000:
            continue  # the whole data set's output layer is too big for WebAssembly's 32-bit memory
        t = time_step(kind, B, T, Vs, Vt, **cfg)
        print(f'{kind:12s} batch {B:3d} vocab {Vs:5d}/{Vt:5d} {cfg}: {t * 1000:7.0f} ms per step', flush=True)
