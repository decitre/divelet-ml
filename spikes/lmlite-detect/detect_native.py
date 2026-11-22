"""Spike: yolos-tiny as a bare ONNX model, run natively with lmlite's OnnxModel, on the photo of the book's section 14.3."""
import asyncio, json, time
import numpy as np
from PIL import Image
from lmlite import OnnxModel

LABELS = json.load(open("model/config.json"))["id2label"]
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(image, shortest=512):
    w, h = image.size
    scale = shortest / min(w, h)
    size = (round(w * scale), round(h * scale))
    pixels = np.asarray(image.convert("RGB").resize(size, Image.BILINEAR), np.float32) / 255
    return ((pixels - MEAN) / STD).transpose(2, 0, 1)[None]


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


async def main():
    model = await OnnxModel.load("model/model_quantized.onnx", backend="native", licence="Apache-2.0")
    print(model.inputs, model.outputs)
    image = Image.open("catdog.jpg")
    t = time.time()
    out = await model.run({model.inputs[0]["name"] if isinstance(model.inputs[0], dict) else model.inputs[0]: preprocess(image)})
    print("run", round(time.time() - t, 2), "s", {k: v.shape for k, v in out.items()})
    probs = softmax(out["logits"][0])[:, :-1]  # the last class is "no object"
    cls, score = probs.argmax(1), probs.max(1)
    for i in np.flatnonzero(score > 0.7):
        cx, cy, w, h = out["pred_boxes"][0, i]
        W, H = image.size
        print(LABELS[str(cls[i])], round(float(score[i]), 3), "box (x1,y1,x2,y2):", [round(v) for v in ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)])
    await model.dispose()

asyncio.run(main())
