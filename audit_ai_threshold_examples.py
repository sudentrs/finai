import ast
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


NOTEBOOK = Path(r"C:\Users\sudet\Downloads\ai_model.ipynb")
INPUT_XLSX = Path(r"C:\Users\sudet\Desktop\finai\old finai\combined_sentences (3).xlsx")
OUT_DIR = Path(r"C:\Users\sudet\Downloads\finai\ai_threshold_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def extract_list_assignments(notebook_path, names):
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    extracted = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names:
                        try:
                            extracted[target.id] = ast.literal_eval(node.value)
                        except Exception:
                            pass
    return extracted


df = pd.read_excel(INPUT_XLSX)
df["sentence_clean"] = df["sentence"].map(clean_text)
df = df[df["sentence_clean"].str.len() > 0].copy()
y = df["is_ai_related"].astype(int)

custom_stopwords = [
    "infrastructure",
    "applications",
    "technology",
    "technologies",
    "tools",
    "solutions",
    "systems",
    "system",
    "platform",
    "digital",
    "data",
    "models",
    "development",
    "deployment",
    "automation",
    "cloud",
    "network",
    "capabilities",
    "features",
    "process",
    "processes",
    "optimize",
    "optimization",
    "advanced",
    "support",
    "insights",
    "offerings",
    "services",
    "integration",
    "implementation",
    "architecture",
    "enhanced",
    "improved",
    "large",
    "capability",
    "enabled",
]

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words=custom_stopwords,
    sublinear_tf=True,
)
lr = LogisticRegression(
    solver="saga",
    C=3.0,
    max_iter=10000,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42,
)

X = tfidf.fit_transform(df["sentence_clean"])
lr.fit(X, y)

lists = extract_list_assignments(NOTEBOOK, {"new_sentences", "test_sentences"})
rows = []
for source_name, sentences in lists.items():
    for idx, sentence in enumerate(sentences, start=1):
        rows.append({"source_list": source_name, "example_id": idx, "sentence": clean_text(sentence)})

examples = pd.DataFrame(rows).drop_duplicates(subset=["sentence"]).reset_index(drop=True)
examples["prob_ai"] = lr.predict_proba(tfidf.transform(examples["sentence"]))[:, 1]
examples["pred_060"] = (examples["prob_ai"] >= 0.60).astype(int)
examples["classification_060"] = examples["pred_060"].map({1: "AI-related", 0: "Not AI"})
examples.to_csv(OUT_DIR / "notebook_example_sentence_predictions.csv", index=False)

print(examples.to_string(max_colwidth=120))
