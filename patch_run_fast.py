import json, pathlib, sys

# Usage: python patch_run_fast.py False    (or True)
target = sys.argv[1] if len(sys.argv) > 1 else "False"
p = pathlib.Path("sepsis_prediction.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
patched = False
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    if "RUN_FAST" in src and "= True" in src or "= False" in src:
        new = src
        new = new.replace("RUN_FAST        = True", f"RUN_FAST        = {target}")
        new = new.replace("RUN_FAST        = False", f"RUN_FAST        = {target}")
        if new != src:
            c["source"] = new
            patched = True
p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"RUN_FAST gezet op {target} (patched={patched})")
