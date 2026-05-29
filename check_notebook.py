import json, ast
nb = json.loads(open("sepsis_prediction.ipynb", encoding="utf-8").read())
errors = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors += 1
        print(f"Cell {i}: line {e.lineno}: {e.msg}")
        lines = src.split("\n")
        for n in range(max(0, e.lineno - 2), min(len(lines), e.lineno + 1)):
            print(f"  {n+1:3} | {lines[n]}")
        print()
print("Totaal syntax errors:", errors)
