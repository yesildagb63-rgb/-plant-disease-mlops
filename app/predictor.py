import json

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Predictor:
    def __init__(self, config="config.yaml", treatments="app/treatments.json"):
        cfg = yaml.safe_load(open(config, encoding="utf-8"))
        self.size = cfg["img_size"]
        self.threshold = cfg["confidence_threshold"]
        self.classes = json.load(open(cfg["classes_path"], encoding="utf-8"))
        self.info = json.load(open(treatments, encoding="utf-8"))
        self.sess = ort.InferenceSession(cfg["onnx_path"])

    def preprocess(self, img):
        img = img.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def predict(self, img):
        logits = self.sess.run(None, {"input": self.preprocess(img)})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(p)[::-1]
        best = self.classes[order[0]]
        conf = float(p[order[0]])
        confident = conf >= self.threshold
        return {
            "class": best,
            "name": self.info[best]["name"],
            "confidence": round(conf, 4),
            "confident": confident,
            "treatment": self.info[best]["treatment"] if confident else [
                "Model bu fotoğraftan emin değil. Yaprağı yakından, gün ışığında ve tek başına çekip tekrar deneyin."],
            "top3": [{"class": self.classes[i], "name": self.info[self.classes[i]]["name"],
                      "confidence": round(float(p[i]), 4)} for i in order[:3]],
        }