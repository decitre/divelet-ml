"""Run by profile_pyodide.mjs inside Pyodide: where the time of one MLP epoch goes."""
import sys, time, cProfile, pstats, io
sys.path.insert(0, '/pkg')
import matplotlib; matplotlib.use('Agg')
import autograd.numpy as np
import numpy as onp
import divelet_ml as d2l

T = time.perf_counter
def timeit(f, n):
    f(); t = T()
    for _ in range(n): f()
    return (T() - t) / n

# Fashion-MNIST shapes with random values: dense compute does not depend on the values.
class Fake(d2l.DataModule):
    def __init__(self):
        super().__init__(); self.batch_size = 256
    def get_dataloader(self, train):
        n = 60000 if train else 10000
        X = onp.random.rand(n, 1, 28, 28).astype('float32'); y = onp.random.randint(0, 10, n)
        return d2l.ArrayLoader((X, y), 256, shuffle=train)

class MLPScratch(d2l.Classifier):
    def __init__(self, lr=0.1, sigma=0.01):
        super().__init__(); self.save_hyperparameters()
        self.W1 = (np.random.randn(784, 256) * sigma).astype(np.float32); self.b1 = np.zeros(256, dtype=np.float32)
        self.W2 = (np.random.randn(256, 10) * sigma).astype(np.float32); self.b2 = np.zeros(10, dtype=np.float32)
    def forward(self, X):
        X = X.reshape((-1, 784)); H = np.maximum(np.matmul(X, self.W1) + self.b1, 0)
        return np.matmul(H, self.W2) + self.b2

NB_TRAIN, NB_VAL = 235, 40  # 60000/256 and 10000/256, rounded up
X = onp.random.rand(256, 784).astype('float32'); W1 = onp.random.rand(784, 256).astype('float32')
H = onp.random.rand(256, 256).astype('float32'); W2 = onp.random.rand(256, 10).astype('float32'); G = onp.random.rand(256, 10).astype('float32')
dH = onp.random.rand(256, 256).astype('float32')
# one training step multiplies: fwd X@W1, H@W2; bwd X.T@dZ1, H.T@G, G@W2.T
mm = {'X@W1': lambda: X @ W1, 'H@W2': lambda: H @ W2, 'X.T@dZ1': lambda: X.T @ dH, 'H.T@G': lambda: H.T @ G, 'G@W2.T': lambda: G @ W2.T}
mm_t = {k: timeit(f, 10) for k, f in mm.items()}
for k, v in mm_t.items(): print(f'matmul {k:8s} {v*1e3:8.1f} ms')
mm_step = sum(mm_t.values()); mm_fwd = mm_t['X@W1'] + mm_t['H@W2']
print(f'matmuls per train step {mm_step*1e3:.1f} ms -> epoch (235 steps) {mm_step*NB_TRAIN:.1f} s')

model = MLPScratch(); data = Fake()
trainer = d2l.Trainer(max_epochs=1); trainer.prepare_data(data); trainer.prepare_model(model)
model.board.display = False  # plotting measured separately
trainer.optim = model.configure_optimizers(); trainer.epoch = 0; trainer.train_batch_idx = 0; trainer.val_batch_idx = 0
batch = next(iter(trainer.train_dataloader))
def step():
    g = d2l.param_grads(lambda: model.training_step(trainer.prepare_batch(batch)), model, trainer.optim.params)
    trainer.optim.step(g)
step_t = timeit(step, 10)
print(f'full train step {step_t*1e3:.1f} ms -> {step_t*NB_TRAIN:.1f} s per epoch; matmul share {mm_step/step_t*100:.0f}%')
val_t = timeit(lambda: model.validation_step(trainer.prepare_batch(batch)), 10)
print(f'val step {val_t*1e3:.1f} ms -> {val_t*NB_VAL:.1f} s; matmul share {mm_fwd/val_t*100:.0f}%')

t = T(); trainer2 = d2l.Trainer(max_epochs=1); m2 = MLPScratch(); m2.board.display = False; trainer2.fit(m2, data)
print(f'real epoch via Trainer.fit, no plotting (incl. data shuffling/batching): {T()-t:.1f} s')

pr = cProfile.Profile(); pr.enable(); step(); step(); step(); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(8); print(s.getvalue()[:2500])

import matplotlib.pyplot as plt
def plot3():
    fig = plt.figure(figsize=(3.5, 2.5))
    for _ in range(3):
        plt.plot([0, 1, 2], [1, .5, .3]); b = io.StringIO(); fig.savefig(b, format='svg')
    plt.close(fig)
print(f'3 plot redraws (figure + svg render): {timeit(plot3, 3):.2f} s')
