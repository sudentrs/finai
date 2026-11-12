import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


notebook_path = Path(r"C:\Users\sudet\Downloads\ai_model.ipynb")
excel_path = Path(r"C:\Users\sudet\Desktop\finai\old finai\combined_sentences (3).xlsx")

print("NOTEBOOK")
nb = json.loads(notebook_path.read_text(encoding="utf-8"))
print("cells", len(nb.get("cells", [])))
print("kernelspec", nb.get("metadata", {}).get("kernelspec"))
for i, cell in enumerate(nb.get("cells", [])[:40]):
    src = "".join(cell.get("source", []))
    print(f"\n--- CELL {i} {cell.get('cell_type')} ---")
    print(src[:1600])

print("\nEXCEL")
xl = pd.ExcelFile(excel_path)
print("sheets", xl.sheet_names)
for sheet in xl.sheet_names:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    print(f"\n--- {sheet} ---")
    print("shape", df.shape)
    print("columns", list(df.columns))
    print(df.head(5).to_string())
    print("dtypes")
    print(df.dtypes.to_string())
