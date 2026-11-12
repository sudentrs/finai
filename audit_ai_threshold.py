from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split


INPUT_XLSX = Path(r"C:\Users\sudet\Desktop\finai\old finai\combined_sentences (3).xlsx")
OUT_DIR = Path(r"C:\Users\sudet\Downloads\finai\ai_threshold_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def metrics_at_threshold(y_true, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive": int(pred.sum()),
        "predicted_positive_rate": float(pred.mean()),
    }


def threshold_table(y_true, score, thresholds):
    return pd.DataFrame([metrics_at_threshold(y_true, score, th) for th in thresholds])


df = pd.read_excel(INPUT_XLSX)
df["sentence_clean"] = df["sentence"].map(clean_text)
df = df[df["sentence_clean"].str.len() > 0].copy()
df["label"] = df["is_ai_related"].astype(int)

thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)

summary_lines = []
summary_lines.append("AI threshold audit")
summary_lines.append(f"Input: {INPUT_XLSX}")
summary_lines.append(f"Rows with non-empty sentence: {len(df):,}")
summary_lines.append(f"Manual AI positives: {df['label'].sum():,}")
summary_lines.append(f"Manual non-AI negatives: {(1 - df['label']).sum():,}")
summary_lines.append(f"Positive share: {df['label'].mean():.4%}")

if "ai_score" in df.columns:
    ai_score = df["ai_score"].astype(float).to_numpy()
    score_table = threshold_table(df["label"].to_numpy(), ai_score, thresholds)
    score_table.to_csv(OUT_DIR / "ai_score_threshold_metrics.csv", index=False)
    best_f1_row = score_table.loc[score_table["f1"].idxmax()]
    th_060_row = score_table.loc[np.isclose(score_table["threshold"], 0.60)].iloc[0]
    th_054_row = score_table.loc[np.isclose(score_table["threshold"], 0.54)].iloc[0]
    summary_lines.append("")
    summary_lines.append("Existing ai_score threshold scan")
    summary_lines.append(f"AUC: {roc_auc_score(df['label'], ai_score):.4f}")
    summary_lines.append(
        "Best-F1 threshold: "
        f"{best_f1_row['threshold']:.2f}, F1={best_f1_row['f1']:.4f}, "
        f"precision={best_f1_row['precision']:.4f}, recall={best_f1_row['recall']:.4f}, "
        f"FP={int(best_f1_row['fp'])}, FN={int(best_f1_row['fn'])}"
    )
    summary_lines.append(
        "At threshold 0.54: "
        f"F1={th_054_row['f1']:.4f}, precision={th_054_row['precision']:.4f}, "
        f"recall={th_054_row['recall']:.4f}, FP={int(th_054_row['fp'])}, FN={int(th_054_row['fn'])}"
    )
    summary_lines.append(
        "At threshold 0.60: "
        f"F1={th_060_row['f1']:.4f}, precision={th_060_row['precision']:.4f}, "
        f"recall={th_060_row['recall']:.4f}, FP={int(th_060_row['fp'])}, FN={int(th_060_row['fn'])}"
    )


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

x = df["sentence_clean"]
y = df["label"].to_numpy()

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
    max_iter=5000,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42,
)

# Cleaner validation than the notebook: no upsampling before the split.
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)
lr.fit(Xtr, y_train)
holdout_proba = lr.predict_proba(Xte)[:, 1]
holdout_table = threshold_table(y_test, holdout_proba, thresholds)
holdout_table.to_csv(OUT_DIR / "logreg_holdout_threshold_metrics.csv", index=False)

hold_best = holdout_table.loc[holdout_table["f1"].idxmax()]
hold_060 = holdout_table.loc[np.isclose(holdout_table["threshold"], 0.60)].iloc[0]
hold_054 = holdout_table.loc[np.isclose(holdout_table["threshold"], 0.54)].iloc[0]

summary_lines.append("")
summary_lines.append("TF-IDF + logistic regression holdout validation")
summary_lines.append(f"Train size: {len(X_train):,}; test size: {len(X_test):,}")
summary_lines.append(f"Holdout AUC: {roc_auc_score(y_test, holdout_proba):.4f}")
summary_lines.append(
    "Best-F1 threshold: "
    f"{hold_best['threshold']:.2f}, F1={hold_best['f1']:.4f}, "
    f"precision={hold_best['precision']:.4f}, recall={hold_best['recall']:.4f}, "
    f"FP={int(hold_best['fp'])}, FN={int(hold_best['fn'])}"
)
summary_lines.append(
    "At threshold 0.54: "
    f"F1={hold_054['f1']:.4f}, precision={hold_054['precision']:.4f}, "
    f"recall={hold_054['recall']:.4f}, FP={int(hold_054['fp'])}, FN={int(hold_054['fn'])}"
)
summary_lines.append(
    "At threshold 0.60: "
    f"F1={hold_060['f1']:.4f}, precision={hold_060['precision']:.4f}, "
    f"recall={hold_060['recall']:.4f}, FP={int(hold_060['fp'])}, FN={int(hold_060['fn'])}"
)

# 5-fold out-of-fold probabilities for a more stable threshold audit.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    stop_words=custom_stopwords,
    sublinear_tf=True,
)
cv_lr = LogisticRegression(
    solver="saga",
    C=3.0,
    max_iter=5000,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42,
)

from sklearn.pipeline import make_pipeline

pipe = make_pipeline(cv_tfidf, cv_lr)
cv_proba = cross_val_predict(pipe, x, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
cv_table = threshold_table(y, cv_proba, thresholds)
cv_table.to_csv(OUT_DIR / "logreg_5fold_oof_threshold_metrics.csv", index=False)
cv_best = cv_table.loc[cv_table["f1"].idxmax()]
cv_060 = cv_table.loc[np.isclose(cv_table["threshold"], 0.60)].iloc[0]
cv_054 = cv_table.loc[np.isclose(cv_table["threshold"], 0.54)].iloc[0]

summary_lines.append("")
summary_lines.append("TF-IDF + logistic regression 5-fold out-of-fold validation")
summary_lines.append(f"OOF AUC: {roc_auc_score(y, cv_proba):.4f}")
summary_lines.append(
    "Best-F1 threshold: "
    f"{cv_best['threshold']:.2f}, F1={cv_best['f1']:.4f}, "
    f"precision={cv_best['precision']:.4f}, recall={cv_best['recall']:.4f}, "
    f"FP={int(cv_best['fp'])}, FN={int(cv_best['fn'])}"
)
summary_lines.append(
    "At threshold 0.54: "
    f"F1={cv_054['f1']:.4f}, precision={cv_054['precision']:.4f}, "
    f"recall={cv_054['recall']:.4f}, FP={int(cv_054['fp'])}, FN={int(cv_054['fn'])}"
)
summary_lines.append(
    "At threshold 0.60: "
    f"F1={cv_060['f1']:.4f}, precision={cv_060['precision']:.4f}, "
    f"recall={cv_060['recall']:.4f}, FP={int(cv_060['fp'])}, FN={int(cv_060['fn'])}"
)

diagnostics = df[["sentence_clean", "label"]].copy()
diagnostics["cv_prob_ai"] = cv_proba
diagnostics["cv_pred_060"] = (cv_proba >= 0.60).astype(int)
diagnostics["error_060"] = diagnostics["label"] != diagnostics["cv_pred_060"]
diagnostics.sort_values(["error_060", "cv_prob_ai"], ascending=[False, False]).to_csv(
    OUT_DIR / "logreg_5fold_sentence_predictions.csv", index=False
)

(OUT_DIR / "threshold_audit_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
print("\n".join(summary_lines))
