from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parent
SOURCE_NDJSON = ROOT / "results_sections.ndjson"
OUT_DIR = ROOT / "analysis" / "item1_ai_exposure"
RANDOM_SEED = 42
N_TOPICS = 10

EXPLICIT_AI_PATTERN = re.compile(
    r"\b(ai|artificial intelligence|machine learning|generative ai|large language model|llm|"
    r"deep learning|neural network|predictive analytics|computer vision|natural language)\b",
    re.IGNORECASE,
)

STOP_TERMS = {
    "ai",
    "artificial",
    "intelligence",
    "machine",
    "learning",
    "technology",
    "technologies",
    "use",
    "uses",
    "using",
    "used",
    "company",
    "business",
    "product",
    "products",
    "service",
    "services",
    "solutions",
    "solution",
    "platform",
    "platforms",
    "data",
    "including",
    "include",
    "includes",
    "customers",
    "customer",
    "software",
}

# Manual interpretation of the stable NMF topics generated below.
# Keep topic 1 and 5 out of exposure/expertise because they are mostly
# "learning/development/training" false positives rather than AI capability.
TOPIC_INTERPRETATION = {
    0: ("AI products and LLM interfaces", "expertise", True, True),
    1: ("Learning and development false positives", "exclude_noise", False, False),
    2: ("Generic real-time systems false positives", "exclude_noise", False, False),
    3: ("Core ML algorithms and AI capabilities", "expertise", True, True),
    4: ("Workforce training false positives", "exclude_noise", False, False),
    5: ("Cloud, infrastructure, and AI hardware demand", "infrastructure_exposure", True, True),
    6: ("Analytics and predictive intelligence", "expertise", True, True),
    7: ("Digital strategy, automation, and market demand", "market_exposure", False, True),
    8: ("Operational decision support", "operational_exposure", False, True),
    9: ("Generic regulation false positives", "exclude_noise", False, False),
}


def clean_text(text: object) -> str:
    value = str(text or "").replace("\n", " ").replace("\r", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def year_from_record(record: dict) -> str:
    period_end = str(record.get("period_end") or "")
    filed_date = str(record.get("filed_date") or "")
    if len(period_end) >= 4 and period_end[:4].isdigit():
        return period_end[:4]
    if len(filed_date) >= 4 and filed_date[:4].isdigit():
        return filed_date[:4]
    return ""


def extract_item1_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    filing_rows = []
    sentence_rows = []

    with SOURCE_NDJSON.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            section = (record.get("sections") or {}).get("ITEM_1")
            if not section:
                continue

            cik = str(record.get("cik") or "").lstrip("0")
            year = year_from_record(record)
            total = int(section.get("total", 0) or 0)
            ai_count = int(section.get("ai_count", 0) or 0)
            risk_count = int(section.get("risk_count", 0) or 0)
            ai_risk_count = int(section.get("ai_risk_count", 0) or 0)

            filing_rows.append(
                {
                    "company_name": record.get("company_name", ""),
                    "cik": cik,
                    "year": year,
                    "period_end": record.get("period_end", ""),
                    "filed_date": record.get("filed_date", ""),
                    "accession": record.get("accession", ""),
                    "filename": record.get("filename", ""),
                    "sic": record.get("sic", ""),
                    "state": record.get("state", ""),
                    "item1_total_sentences": total,
                    "item1_ai_sentences_count": ai_count,
                    "item1_risk_sentences_count": risk_count,
                    "item1_ai_risk_sentences_count": ai_risk_count,
                    "item1_ai_exposure_ratio": safe_div(ai_count, total),
                }
            )

            for sentence in section.get("ai_sentences") or []:
                text = clean_text(sentence.get("sentence", ""))
                if len(text) < 25:
                    continue
                sentence_rows.append(
                    {
                        "company_name": record.get("company_name", ""),
                        "cik": cik,
                        "year": year,
                        "period_end": record.get("period_end", ""),
                        "filed_date": record.get("filed_date", ""),
                        "accession": record.get("accession", ""),
                        "filename": record.get("filename", ""),
                        "sic": record.get("sic", ""),
                        "state": record.get("state", ""),
                        "item1_total_sentences": total,
                        "sentence_id": sentence.get("sentence_id"),
                        "ai_prob": sentence.get("ai_prob"),
                        "risk_prob": sentence.get("risk_prob"),
                        "explicit_ai_term": bool(EXPLICIT_AI_PATTERN.search(text)),
                        "sentence": text,
                    }
                )

    filing_df = pd.DataFrame(filing_rows)
    sentence_df = (
        pd.DataFrame(sentence_rows)
        .drop_duplicates(subset=["cik", "period_end", "accession", "sentence"])
        .reset_index(drop=True)
    )
    return filing_df, sentence_df


def topic_label(feature_names: np.ndarray, weights: np.ndarray, top_n: int = 4) -> str:
    top_idx = np.argsort(weights)[::-1][:top_n]
    return " / ".join(feature_names[i] for i in top_idx)


def fit_topics(sentence_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=8,
        max_df=0.55,
        ngram_range=(1, 2),
        max_features=4000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vectorizer.fit_transform(sentence_df["sentence"])
    feature_names = np.array(vectorizer.get_feature_names_out())
    keep_mask = np.array([term not in STOP_TERMS and not term.endswith(" ai") for term in feature_names])
    X = X[:, keep_mask]
    feature_names = feature_names[keep_mask]

    nmf = NMF(n_components=N_TOPICS, random_state=RANDOM_SEED, init="nndsvda", max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_

    assignments = sentence_df.copy()
    assignments["topic_id"] = W.argmax(axis=1)
    assignments["topic_weight"] = W.max(axis=1)

    topic_rows = []
    example_rows = []
    for topic_id in range(N_TOPICS):
        manual_label, dimension, expertise_flag, exposure_flag = TOPIC_INTERPRETATION[topic_id]
        top_term_idx = np.argsort(H[topic_id])[::-1][:15]
        terms = [feature_names[i] for i in top_term_idx]
        topic_slice = assignments[assignments["topic_id"] == topic_id].sort_values("topic_weight", ascending=False)

        topic_rows.append(
            {
                "topic_id": topic_id,
                "model_label": topic_label(feature_names, H[topic_id]),
                "manual_label": manual_label,
                "dimension": dimension,
                "counts_as_expertise": expertise_flag,
                "counts_as_exposure": exposure_flag,
                "sentence_count": int(len(topic_slice)),
                "explicit_ai_sentence_count": int(topic_slice["explicit_ai_term"].sum()),
                "explicit_ai_share": safe_div(int(topic_slice["explicit_ai_term"].sum()), len(topic_slice)),
                "top_terms": ", ".join(terms),
            }
        )

        for _, row in topic_slice.head(12).iterrows():
            example_rows.append(
                {
                    "topic_id": topic_id,
                    "model_label": topic_label(feature_names, H[topic_id]),
                    "manual_label": manual_label,
                    "dimension": dimension,
                    "topic_weight": row["topic_weight"],
                    "year": row["year"],
                    "cik": row["cik"],
                    "company_name": row["company_name"],
                    "sentence": row["sentence"],
                }
            )

    topic_df = pd.DataFrame(topic_rows)
    example_df = pd.DataFrame(example_rows)

    meta = topic_df[["topic_id", "model_label", "manual_label", "dimension", "counts_as_expertise", "counts_as_exposure"]]
    assignments = assignments.merge(meta, on="topic_id", how="left")
    return assignments, topic_df, example_df


def build_company_year_indices(filing_df: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = (
        filing_df.groupby(["cik", "company_name", "year"], dropna=False)
        .agg(
            filings=("accession", "nunique"),
            item1_total_sentences=("item1_total_sentences", "sum"),
            item1_ai_sentences_count=("item1_ai_sentences_count", "sum"),
            item1_risk_sentences_count=("item1_risk_sentences_count", "sum"),
            item1_ai_risk_sentences_count=("item1_ai_risk_sentences_count", "sum"),
        )
        .reset_index()
    )

    if assignments.empty:
        return base, pd.DataFrame()

    sentence_agg = (
        assignments.groupby(["cik", "company_name", "year"], dropna=False)
        .agg(
            topic_model_ai_sentences=("sentence", "count"),
            explicit_ai_sentences=("explicit_ai_term", "sum"),
            expertise_ai_sentences=("counts_as_expertise", "sum"),
            exposure_ai_sentences=("counts_as_exposure", "sum"),
            topic_breadth=("topic_id", "nunique"),
            excluded_noise_sentences=("counts_as_exposure", lambda s: int((~s.astype(bool)).sum())),
        )
        .reset_index()
    )

    out = base.merge(sentence_agg, on=["cik", "company_name", "year"], how="left")
    count_cols = [
        "topic_model_ai_sentences",
        "explicit_ai_sentences",
        "expertise_ai_sentences",
        "exposure_ai_sentences",
        "topic_breadth",
        "excluded_noise_sentences",
    ]
    out[count_cols] = out[count_cols].fillna(0).astype(int)

    out["item1_ai_exposure_ratio"] = out.apply(
        lambda r: safe_div(r["item1_ai_sentences_count"], r["item1_total_sentences"]), axis=1
    )
    out["explicit_ai_exposure_ratio"] = out.apply(
        lambda r: safe_div(r["explicit_ai_sentences"], r["item1_total_sentences"]), axis=1
    )
    out["ai_expertise_ratio"] = out.apply(
        lambda r: safe_div(r["expertise_ai_sentences"], r["item1_total_sentences"]), axis=1
    )
    out["ai_expertise_share_of_ai"] = out.apply(
        lambda r: safe_div(r["expertise_ai_sentences"], r["topic_model_ai_sentences"]), axis=1
    )
    out["ai_exposure_clean_ratio"] = out.apply(
        lambda r: safe_div(r["exposure_ai_sentences"], r["item1_total_sentences"]), axis=1
    )
    out["noise_share_of_ai"] = out.apply(
        lambda r: safe_div(r["excluded_noise_sentences"], r["topic_model_ai_sentences"]), axis=1
    )

    topic_counts = (
        assignments.groupby(["cik", "company_name", "year", "topic_id", "manual_label", "dimension"], dropna=False)
        .size()
        .reset_index(name="topic_sentence_count")
    )
    denominators = base[["cik", "company_name", "year", "item1_total_sentences", "item1_ai_sentences_count"]]
    topic_counts = topic_counts.merge(denominators, on=["cik", "company_name", "year"], how="left")
    topic_counts["topic_ratio_item1"] = topic_counts.apply(
        lambda r: safe_div(r["topic_sentence_count"], r["item1_total_sentences"]), axis=1
    )
    topic_counts["topic_share_of_ai"] = topic_counts.apply(
        lambda r: safe_div(r["topic_sentence_count"], r["item1_ai_sentences_count"]), axis=1
    )

    return out.sort_values(["year", "cik"]), topic_counts.sort_values(["year", "cik", "topic_id"])


def build_yearly_topic_summary(assignments: pd.DataFrame, filing_df: pd.DataFrame) -> pd.DataFrame:
    denominators = (
        filing_df.groupby("year", dropna=False)
        .agg(
            filings=("accession", "nunique"),
            item1_total_sentences=("item1_total_sentences", "sum"),
            item1_ai_sentences_count=("item1_ai_sentences_count", "sum"),
        )
        .reset_index()
    )
    topic_year = (
        assignments.groupby(["year", "topic_id", "manual_label", "dimension"], dropna=False)
        .size()
        .reset_index(name="topic_sentence_count")
    )
    topic_year = topic_year.merge(denominators, on="year", how="left")
    topic_year["topic_ratio_item1"] = topic_year.apply(
        lambda r: safe_div(r["topic_sentence_count"], r["item1_total_sentences"]), axis=1
    )
    topic_year["topic_share_of_ai"] = topic_year.apply(
        lambda r: safe_div(r["topic_sentence_count"], r["item1_ai_sentences_count"]), axis=1
    )
    return topic_year.sort_values(["year", "topic_id"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filing_df, sentence_df = extract_item1_rows()
    assignments, topic_df, example_df = fit_topics(sentence_df)
    company_year_df, company_year_topic_df = build_company_year_indices(filing_df, assignments)
    yearly_topic_df = build_yearly_topic_summary(assignments, filing_df)

    filing_df.to_csv(OUT_DIR / "item1_filing_denominators.csv", index=False)
    sentence_df.to_csv(OUT_DIR / "item1_ai_sentences.csv", index=False)
    assignments.to_csv(OUT_DIR / "item1_ai_topic_assignments.csv", index=False)
    topic_df.to_csv(OUT_DIR / "item1_ai_topics.csv", index=False)
    example_df.to_csv(OUT_DIR / "item1_ai_topic_examples.csv", index=False)
    company_year_df.to_csv(OUT_DIR / "item1_company_year_exposure_expertise.csv", index=False)
    company_year_topic_df.to_csv(OUT_DIR / "item1_company_year_topic_exposure.csv", index=False)
    yearly_topic_df.to_csv(OUT_DIR / "item1_yearly_topic_summary.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "item1_filings", "value": len(filing_df)},
            {"metric": "item1_ai_sentence_rows_raw_deduped", "value": len(sentence_df)},
            {"metric": "topic_count", "value": len(topic_df)},
            {"metric": "company_year_rows", "value": len(company_year_df)},
            {"metric": "explicit_ai_sentence_rows", "value": int(sentence_df["explicit_ai_term"].sum())},
            {
                "metric": "explicit_ai_sentence_share",
                "value": safe_div(int(sentence_df["explicit_ai_term"].sum()), len(sentence_df)),
            },
            {
                "metric": "topic_sentence_counts",
                "value": json.dumps(dict(Counter(assignments["manual_label"]))),
            },
        ]
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"Wrote ITEM_1 AI exposure/topic outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
