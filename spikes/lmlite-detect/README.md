# Spike: detect the dog and the cat of section 14.3 with lmLite's `OnnxModel`

Question (ROADMAP, Phase 6): can a Chapter 14 section use a pretrained model through lmLite, in both kernels and natively, with a
model small enough for the site?

`make_notebook.py` writes `detect.ipynb` (and `detect_pyodide.ipynb`, with the install line that jl-forge adds for Pyodide). It runs
`Xenova/yolos-tiny` (Apache-2.0, `onnx/model_quantized.onnx`, 9.7 MB) as a bare ONNX model: NumPy preprocessing, `OnnxModel.run`, softmax and
the book's `box_center_to_corner` on the outputs, then the intersection over union with the book's hand-made boxes of 14.3.
`detect_native.py` is the same natively; `browser_run.py` uploads the notebook to lmLite's hosted JupyterLite site and runs it in Chrome.
`model/` (the model and its config) and `catdog.jpg` are downloaded, not committed: the URLs are in `detect.ipynb`.

## Result (2026-11-22, one Mac, Chrome)

| | native (onnxruntime 1.32) | xeus-python (WASM) | Pyodide (WASM) |
|---|---|---|---|
| load | 0.2 s | 4.0 s | 6.5 s |
| one image (512 x 664) | 0.10 s | 0.96 s | 1.00 s |
| dog, score; box | 0.996; [58, 43, 489, 510] | 0.997; [57, 45, 491, 512] | 0.997; [57, 45, 491, 512] |
| cat, score; box | 0.957; [392, 114, 675, 467] | 0.977; [391, 116, 674, 470] | 0.977; [391, 116, 674, 470] |
| IoU with the book's boxes, dog and cat | 0.73, 0.84 | 0.73, 0.85 | 0.73, 0.85 |

Both kernels run it with no stderr output. The two WASM kernels agree exactly; native differs by a few pixels and 0.01 to 0.02 of score, the
drift of quantized weights that lmLite's NATIVE.md describes (the browser stays the reference).

## Findings

- **`OnnxModel` is not in a release.** It is on lmLite's `main` (the hosted site's wheel is called 0.12.1 but has it); PyPI's 0.12.1 does not.
  divelet-ml installs released packages only (conda-forge for xeus-python, PyPI wheels for Pyodide), so no divelet-ml notebook can use it
  until lmLite publishes a release that has it. Decision: wait for that release.
- **`LLM.pipeline("object-detection")` is no route for tests**: lmLite's native backend does not run pipelines (`NotImplementedError`), so a
  CPython notebook test could not run such a notebook. The bare `OnnxModel` runs natively, which is what makes `tests/test_notebooks.py` possible.
- **Pyodide has no network in `urllib`**: the notebook fetches with `pyodide.http.pyfetch` there (lmLite's own notebooks use `pyodide_http`).
- **Hosting**: the model (9.7 MB) and `config.json` (4 KB, for the 91 class names) would be copied by `build.sh` to the site, as the data sets are,
  with Apache-2.0 credit in `NOTICE.md`; the smoke test rejects other hosts, so this spike's Hugging Face URL is not an option for a notebook.
  Site size: about 290 MB of 1 GB, so one small model is affordable, not many.
- **Fit with 14.3**: the section teaches box formats and drawing by hand. The detector's boxes use the book's `box_center_to_corner`, so a
  tablet-marked section after the book's code ("What a pretrained detector finds") fits without replacing it.

## Not checked

The model hosted by a divelet-ml build (the self-hosted path, `lmlite` as a dependency of the xeus and Pyodide builds), iPad/WebKit, WebGPU,
other models (`detr-resnet-50` is much larger), and the 14.3 notebook itself.
