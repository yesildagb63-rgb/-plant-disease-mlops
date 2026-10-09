import random
import shutil
from pathlib import Path

import yaml

cfg = yaml.safe_load(open("config.yaml"))
src, out = Path(cfg["data_dir"]), Path(cfg["processed_dir"])
tr, va, _ = cfg["split"]
rng = random.Random(cfg["seed"])
dist = {}

for cls in cfg["classes"]:
    files = sorted((src / cls).iterdir())
    rng.shuffle(files)
    n = len(files)
    a, b = int(n * tr), int(n * (tr + va))
    parts = {"train": files[:a], "val": files[a:b], "test": files[b:]}
    for split, fs in parts.items():
        d = out / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for f in fs:
            shutil.copy2(f, d / f.name)
    dist[cls] = {s: len(fs) for s, fs in parts.items()}

print(f"{'class':32}{'train':>7}{'val':>7}{'test':>7}")
for cls, c in dist.items():
    print(f"{cls:32}{c['train']:>7}{c['val']:>7}{c['test']:>7}")
