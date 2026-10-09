import hashlib
import json
import os
import sys
from pathlib import Path

import great_expectations as gx
import pandas as pd
import yaml
from PIL import Image

cfg = yaml.safe_load(open("config.yaml"))

rows = []
for cls_dir in sorted(Path(cfg["data_dir"]).iterdir()):
    if not cls_dir.is_dir():
        continue
    for f in cls_dir.iterdir():
        row = {"label": cls_dir.name, "ext": f.suffix.lower(), "readable": True,
               "width": 0, "height": 0, "md5": hashlib.md5(f.read_bytes()).hexdigest()}
        try:
            with Image.open(f) as im:
                im.verify()
            with Image.open(f) as im:
                row["width"], row["height"] = im.size
        except Exception:
            row["readable"] = False
        rows.append(row)
df = pd.DataFrame(rows)

ctx = gx.get_context(mode="ephemeral")
suite = ctx.suites.add(gx.ExpectationSuite(name="plantvillage"))

E = gx.expectations
for e in [
    E.ExpectTableRowCountToBeBetween(min_value=1000),
    E.ExpectColumnDistinctValuesToEqualSet(column="label", value_set=cfg["classes"]),
    E.ExpectColumnValuesToBeInSet(column="readable", value_set=[True]),
    E.ExpectColumnValuesToBeInSet(column="ext", value_set=[".jpg", ".jpeg", ".png"]),
    E.ExpectColumnValuesToBeBetween(column="width", min_value=64, max_value=4096),
    E.ExpectColumnValuesToBeBetween(column="height", min_value=64, max_value=4096),
    E.ExpectColumnValuesToBeUnique(column="md5", mostly=0.99),
]:
    suite.add_expectation(e)

batch = (ctx.data_sources.add_pandas("pd")
         .add_dataframe_asset("images")
         .add_batch_definition_whole_dataframe("all")
         .get_batch(batch_parameters={"dataframe": df}))
res = batch.validate(suite)

checks = {}
for r in res.results:
    c = r.expectation_config
    name = getattr(c, "type", None) or getattr(c, "expectation_type", "")
    checks[f"{name}:{c.kwargs.get('column', 'table')}"] = r.success

report = {"success": res.success, "rows": len(df),
          "per_class": df["label"].value_counts().to_dict(), "checks": checks}
os.makedirs("reports", exist_ok=True)
json.dump(report, open("reports/data_validation.json", "w"), indent=2)

for k, v in checks.items():
    print(("OK   " if v else "FAIL ") + k)
print("VALIDATION", "PASSED" if res.success else "FAILED")
sys.exit(0 if res.success else 1)
