// Spike 2: Pyodide in a web worker (as a JupyterLite kernel runs) calling TensorFlow.js on WebGL, synchronously, from Python.
// Serve the repository root after ./build.sh and open /spikes/js-conv/kernel.html:  python3 -m http.server 8000
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
const PY = '/workspace/public/pyodide/static/pyodide/';
importScripts(PY + 'pyodide.js');

// The ops a JS-backed `nn` would hand to the GPU. Data crosses as flat Float32Arrays; dataSync blocks, so Python stays synchronous.
self.mm = (a, as, b, bs) => tf.tidy(() => tf.matMul(tf.tensor(a, as), tf.tensor(b, bs)).dataSync());
self.convfb = (x, xs, w, ws, same) => tf.tidy(() => {  // convolution forward and its gradients, NCHW in and out
  const xt = tf.tensor(x, xs).transpose([0, 2, 3, 1]), wt = tf.tensor(w, ws).transpose([2, 3, 1, 0]);
  const f = (a, b) => tf.conv2d(a, b, 1, same ? 'same' : 'valid');
  const [n, , h, wd] = xs, [o, , kh, kw] = ws, ho = same ? h : h - kh + 1, wo = same ? wd : wd - kw + 1;
  const { value, grads } = tf.valueAndGrads(f)([xt, wt], tf.ones([n, ho, wo, o]));
  return tf.concat([value.transpose([0, 3, 1, 2]).flatten(), grads[0].transpose([0, 3, 1, 2]).flatten(), grads[1].transpose([3, 2, 0, 1]).flatten()]).dataSync();
});

onmessage = async () => {
  try {
    await tf.setBackend('webgl'); await tf.ready();
    const py = await loadPyodide({ indexURL: PY });
    await py.loadPackage(['numpy', 'autograd', 'matplotlib', 'pandas', 'ipython']);
    py.FS.mkdirTree('/pkg/divelet_ml');
    const listing = await (await fetch('/src/divelet_ml/')).text();
    for (const f of [...listing.matchAll(/href="([^"]+\.py)"/g)].map(m => m[1]))
      py.FS.writeFile('/pkg/divelet_ml/' + f, new Uint8Array(await (await fetch('/src/divelet_ml/' + f)).arrayBuffer()));
    let text = '';
    py.setStdout({ batched: s => text += s + '\n' });
    await py.runPythonAsync(await (await fetch('/spikes/js-conv/kernel_bench.py')).text());
    postMessage({ ok: true, text: `${navigator.userAgent}\nTF.js backend: ${tf.getBackend()}\n${text}` });
  } catch (e) { postMessage({ ok: false, text: String(e.message || e).slice(-1500) }); }
};
