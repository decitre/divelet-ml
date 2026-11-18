// Times one epoch of Chapter 5's MLP in the Pyodide of the built site (needs ./build.sh), in Node:
//   node tests/profile_pyodide.mjs
// Prints each matmul, a training step, a validation step and a plot redraw. Random data of Fashion-MNIST's shapes.
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const PY = join(ROOT, 'workspace/public/pyodide/static/pyodide/');
const SRC = join(ROOT, 'src/divelet_ml/');
const { loadPyodide } = await import(PY + 'pyodide.mjs');
const py = await loadPyodide({ indexURL: PY });
await py.loadPackage(['numpy', 'autograd', 'matplotlib', 'pandas', 'ipython']);
py.FS.mkdirTree('/pkg/divelet_ml');
for (const f of readdirSync(SRC).filter(f => f.endsWith('.py')))
  py.FS.writeFile('/pkg/divelet_ml/' + f, readFileSync(SRC + f));
try {
  await py.runPythonAsync(readFileSync(join(ROOT, 'tests/profile_pyodide.py'), 'utf8'));
} catch (e) {
  console.error(String(e.message).slice(-1500));  // the Python traceback, not Pyodide's minified source
  process.exit(1);
}
