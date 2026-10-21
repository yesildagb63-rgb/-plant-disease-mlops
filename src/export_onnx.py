import json
import os

import numpy as np
import onnxruntime as ort
import torch
import yaml

cfg = yaml.safe_load(open("config.yaml"))
size = cfg["img_size"]

model = torch.load("models/model.pt", weights_only=False)
model.eval()
classes = sorted(os.listdir(f"{cfg['processed_dir']}/train"))

dummy = torch.randn(1, 3, size, size)
kwargs = dict(input_names=["input"], output_names=["logits"],
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
try:
    torch.onnx.export(model, dummy, cfg["onnx_path"], dynamo=False, **kwargs)
except TypeError:
    torch.onnx.export(model, dummy, cfg["onnx_path"], **kwargs)

json.dump(classes, open(cfg["classes_path"], "w"), indent=2)

sess = ort.InferenceSession(cfg["onnx_path"])
x = np.random.rand(2, 3, size, size).astype(np.float32)
onnx_out = sess.run(None, {"input": x})[0]
with torch.no_grad():
    torch_out = model(torch.from_numpy(x)).numpy()
diff = float(np.abs(onnx_out - torch_out).max())
print(f"ONNX kaydedildi: {cfg['onnx_path']} ({os.path.getsize(cfg['onnx_path']) / 1e6:.1f} MB)")
print(f"Siniflar: {classes}")
print(f"PyTorch ile en buyuk fark: {diff:.2e}", "OK" if diff < 1e-3 else "FARK BUYUK")