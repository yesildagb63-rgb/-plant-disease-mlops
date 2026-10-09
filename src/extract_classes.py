import re
import sys
import zipfile
from pathlib import Path

import yaml

cfg = yaml.safe_load(open("config.yaml"))
zip_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Hp\Downloads\archive.zip"
out = Path(cfg["data_dir"])
norm = lambda s: re.sub("_+", "_", s)
classes = {norm(c): c for c in cfg["classes"]}
count = dict.fromkeys(cfg["classes"], 0)

with zipfile.ZipFile(zip_path) as z:
    for info in z.infolist():
        if info.is_dir():
            continue
        parts = Path(info.filename).parts
        cls = next((classes[norm(p)] for p in parts[:-1] if norm(p) in classes), None)
        if cls is None:
            continue
        dst = out / cls / parts[-1]
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as s, open(dst, "wb") as d:
            d.write(s.read())
        count[cls] += 1

print(count)
