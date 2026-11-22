import json
cells = [
("markdown", "# Spike: detect the dog and the cat of section 14.3 with a bare ONNX model\n\n`Xenova/yolos-tiny` (Apache-2.0, 9.7 MB quantized) run with lmlite's `OnnxModel`; NumPy in, NumPy out. The same notebook runs natively and in both JupyterLite kernels."),
("code", '''import io, json, sys, time, urllib.request
import numpy as np
from PIL import Image

try:
    import pyodide_http
    pyodide_http.patch_all()  # urllib goes through the browser's fetch
except ImportError:
    pass

from lmlite import OnnxModel

IN_BROWSER = sys.platform == "emscripten"
HF = "https://huggingface.co/Xenova/yolos-tiny/resolve/main/"
IMAGE_URL = "https://raw.githubusercontent.com/d2l-ai/d2l-en/23d7a5aecceee57d1292c56e90cce307f183bb0a/img/catdog.jpg"

async def fetch(url):
    try:
        from pyodide.http import pyfetch  # the Pyodide kernel; urllib has no network there
        return await (await pyfetch(url)).bytes()
    except (ImportError, AttributeError):
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "divelet-spike"})) as r:
            return r.read()

image = Image.open(io.BytesIO(await fetch(IMAGE_URL))).convert("RGB")
labels = json.loads(await fetch(HF + "config.json"))["id2label"] if IN_BROWSER else json.load(open("model/config.json"))["id2label"]
print(image.size, len(labels), "labels")'''),
("code", '''t = time.time()
if IN_BROWSER:
    model = await OnnxModel.load(HF + "onnx/model_quantized.onnx", backend="wasm", licence="Apache-2.0")
else:
    model = await OnnxModel.load("model/model_quantized.onnx", backend="native", licence="Apache-2.0")
print(f"load {time.time() - t:.1f} s", model.model_info()["backend"], model.model_info()["size"], "bytes")'''),
("code", '''MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

def preprocess(image, shortest=512):
    w, h = image.size
    scale = shortest / min(w, h)
    pixels = np.asarray(image.resize((round(w * scale), round(h * scale)), Image.BILINEAR), np.float32) / 255
    return ((pixels - MEAN) / STD).transpose(2, 0, 1)[None]

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def center_to_corner(boxes):  # the book's `box_center_to_corner`
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), axis=-1)

x = preprocess(image)
t = time.time()
out = await model.run({"pixel_values": x})
print(f"input {x.shape}, run {time.time() - t:.2f} s")
probs = softmax(out["logits"][0])[:, :-1]
cls, score = probs.argmax(1), probs.max(1)
W, H = image.size
found = {}
for i in np.flatnonzero(score > 0.7):
    box = center_to_corner(out["pred_boxes"][0, i]) * [W, H, W, H]
    found[labels[str(cls[i])]] = box
    print(labels[str(cls[i])], round(float(score[i]), 3), np.round(box).astype(int).tolist())'''),
("code", '''# The book's hand-made boxes of section 14.3, and how well the detector's agree with them (intersection over union)
book = {"dog": np.array([60.0, 45.0, 378.0, 516.0]), "cat": np.array([400.0, 112.0, 655.0, 493.0])}
def iou(a, b):
    inter = np.clip(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0, None).prod()
    return inter / ((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - inter)
for name in book:
    print(name, "IoU with the book's box:", round(float(iou(found[name], book[name])), 2))
await model.dispose()'''),
]
nb = {"cells": [{"cell_type": k, "metadata": {}, "source": s.splitlines(keepends=True), **({"outputs": [], "execution_count": None} if k == "code" else {})} for k, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python", "language": "python", "name": "python"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("detect.ipynb", "w"), indent=1)

# The Pyodide kernel installs lmlite from the wheel that the site serves, with the line that jl-forge adds to its notebooks.
nb["cells"][1]["source"][0:0] = ['import sys; sys.path.append("/drive"); from install import _; await _()\n']
json.dump(nb, open("detect_pyodide.ipynb", "w"), indent=1)
