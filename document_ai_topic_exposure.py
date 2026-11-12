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
DOCUMENT_LEVEL = ROOT / "analysis" / "results_sections_sample_style" / "document_level.csv"
OUT_DIR = ROOT / "analysis" / "document_ai_exposure"
RANDOM_SEED = 42
N_TOPICS = 12

EXPLICIT_AI_PATTERN = re.compile(
    r"\b(ai|artificial intelligence|machine learning|generative ai|large language model|llm|"
    r"deep learning|neural network|predictive analytics|computer vision|natural language)\b",
    re.IGNORECASE,
)

BOILERPLATE_PATTERN = re.compile(
    r"\b(xbrl|taxonomy|linkbase|schema document|calculation linkbase|definition linkbase|"
    r"label linkbase|presentation linkbase|inline xbrl|exhibit\s+\d|"
    r"101\.(sch|cal|def|lab|pre|ins)|104 cover page)\b",
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
    "companies",
    "business",
    "businesses",
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
    "systems",
    "system",
    "information",
    "may",
    "could",
    "new",
}

# Initial interpretation of the stable whole-document NMF topics. The script
# writes topic examples, so this can be audited and refined after each rerun.
TOPIC_INTERPRETATION = {
    0: ("AI applications, analytics, and digital products", "expertise", True, True),
    1: ("AI reputational harm and legal liability", "risk_legal_exposure", False, True),
    2: ("Generic AI risk-factor language", "risk_general_exposure", False, True),
    3: ("AI laws, privacy, and regulatory compliance", "risk_regulatory_exposure", False, True),
    4: ("Workforce learning and training false positives", "exclude_noise", False, False),
    5: ("AI intellectual property exposure", "risk_ip_exposure", False, True),
    6: ("Cybersecurity governance and incident response", "risk_cyber_exposure", False, True),
    7: ("Core AI and machine-learning capabilities", "expertise", True, True),
    8: ("AI-enabled cyber threats and attacks", "risk_cyber_exposure", False, True),
    9: ("EU AI Act and formal AI regulation", "risk_regulatory_exposure", False, True),
    10: ("AI governance statements and regulator scrutiny", "risk_regulatory_exposure", False, True),
    11: ("Generative AI models, tools, and LLMs", "expertise", True, True),
}


def clean_text(text: object) -> str:
    value = str(text or "").replace("\n", " ").replace("\r", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def normalize_cik(value: object) -> str:
    return str(value or "").replace(".0", "").lstrip("0")


def year_from_record(record: dict) -> str:
    period_end = str(record.get("period_end") or "")
    filed_date = str(record.get("filed_date") or "")
    if len(period_end) >= 4 and period_end[:4].isdigit():
        return period_end[:4]
    if len(filed_date) >= 4 and filed_date[:4].isdigit():
        return filed_date[:4]
    return ""


def read_document_denominators() -> pd.DataFrame:
    usecols = [
        "company_name",
        "cik",
        "fiscal_year_end",
        "form_type",
        "period_end",
        "filed_date",
        "quarter",
        "form_from_path",
        "accession",
        "ticker",
        "sic",
        "state",
        "total_sentences",
        "ai_related_sentences_count",
        "risk_related_sentences_count",
        "ai_risk_sentences_count",
        "crsp_match_accepted",
        "crsp_match_gap_days",
        "crsp_match_method",
    ]
    df = pd.read_csv(DOCUMENT_LEVEL, usecols=lambda c: c in usecols, dtype={"cik": str, "period_end": str})
    df["cik"] = df["cik"].map(normalize_cik)
    df["year"] = df["period_end"].astype(str).str.slice(0, 4)
    rename_map = {
        "total_sentences": "document_total_sentences",
        "ai_related_sentences_count": "document_ai_sentences_count",
        "risk_related_sentences_count": "document_risk_sentences_count",
        "ai_risk_sentences_count": "document_ai_risk_sentences_count",
    }
    df = df.rename(columns=rename_map)
    df["document_ai_exposure_ratio"] = df.apply(
        lambda r: safe_div(r["document_ai_sentences_count"], r["document_total_sentences"]), axis=1
    )
    return df


def extract_document_ai_sentences() -> pd.DataFrame:
    rows = []
    with SOURCE_NDJSON.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            sections = record.get("sections") or {}
            cik = normalize_cik(record.get("cik"))
            year = year_from_record(record)
            for section_name, section in sections.items():
                total = int(section.get("total", 0) or 0)
                for sentence in section.get("ai_sentences") or []:
                    text = clean_text(sentence.get("sentence", ""))
                    if len(text) < 25:
                        continue
                    rows.append(
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
                            "section": section_name,
                            "section_total_sentences": total,
                            "sentence_id": sentence.get("sentence_id"),
                            "ai_prob": sentence.get("ai_prob"),
                            "risk_prob": sentence.get("risk_prob"),
                            "explicit_ai_term": bool(EXPLICIT_AI_PATTERN.search(text)),
                            "boilerplate_noise": bool(BOILERPLATE_PATTERN.search(text)),
                            "sentence": text,
                        }
                    )

    sentence_df = pd.DataFrame(rows)
    if sentence_df.empty:
        return sentence_df
    return sentence_df.drop_duplicates(subset=["accession", "section", "sentence"]).reset_index(drop=True)


def topic_label(feature_names: np.ndarray, weights: np.ndarray, top_n: int = 4) -> str:
    top_idx = np.argsort(weights)[::-1][:top_n]
    return " / ".join(feature_names[i] for i in top_idx)


def fit_topics(sentence_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df = sentence_df.loc[~sentence_df["boilerplate_noise"]].copy().reset_index(drop=True)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=10,
        max_df=0.55,
        ngram_range=(1, 2),
        max_features=6000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vectorizer.fit_transform(model_df["sentence"])
    feature_names = np.array(vectorizer.get_feature_names_out())
    keep_mask = np.array([term not in STOP_TERMS and not term.endswith(" ai") for term in feature_names])
    X = X[:, keep_mask]
    feature_names = feature_names[keep_mask]

    nmf = NMF(n_components=N_TOPICS, random_state=RANDOM_SEED, init="nndsvda", max_iter=600)
    W = nmf.fit_transform(X)
    H = nmf.components_

    assignments = model_df.copy()
    assignments["topic_id"] = W.argmax(axis=1)
    assignments["topic_weight"] = W.max(axis=1)

    topic_rows = []
    example_rows = []
    for topic_id in range(N_TOPICS):
        manual_label, dimension, expertise_flag, exposure_flag = TOPIC_INTERPRETATION[topic_id]
        top_term_idx = np.argsort(H[topic_id])[::-1][:20]
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
                "top_sections": json.dumps(dict(Counter(topic_slice["section"]).most_common(8))),
            }
        )

        for _, row in topic_slice.head(15).iterrows():
            example_rows.append(
                {
                    "topic_id": topic_id,
                    "model_label": topic_label(feature_names, H[topic_id]),
                    "manual_label": manual_label,
                    "dimension": dimension,
                    "topic_weight": row["topic_weight"],
                    "year": row["year"],
                    "section": row["section"],
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


def build_company_year_indices(
    filing_df: pd.DataFrame, sentence_df: pd.DataFrame, assignments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = (
        filing_df.groupby(["cik", "company_name", "year"], dropna=False)
        .agg(
            filings=("accession", "nunique"),
            document_total_sentences=("document_total_sentences", "sum"),
            document_ai_sentences_count=("document_ai_sentences_count", "sum"),
            document_risk_sentences_count=("document_risk_sentences_count", "sum"),
            document_ai_risk_sentences_count=("document_ai_risk_sentences_count", "sum"),
            crsp_match_accepted=("crsp_match_accepted", "max"),
        )
        .reset_index()
    )

    if assignments.empty:
        return base, pd.DataFrame()

    raw_sentence_agg = (
        sentence_df.groupby(["cik", "company_name", "year"], dropna=False)
        .agg(
            raw_ai_sentence_rows=("sentence", "count"),
            boilerplate_ai_sentence_rows=("boilerplate_noise", "sum"),
        )
        .reset_index()
    )
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

    out = base.merge(raw_sentence_agg, on=["cik", "company_name", "year"], how="left")
    out = out.merge(sentence_agg, on=["cik", "company_name", "year"], how="left")
    count_cols = [
        "raw_ai_sentence_rows",
        "boilerplate_ai_sentence_rows",
        "topic_model_ai_sentences",
        "explicit_ai_sentences",
        "expertise_ai_sentences",
        "exposure_ai_sentences",
        "topic_breadth",
        "excluded_noise_sentences",
    ]
    out[count_cols] = out[count_cols].fillna(0).astype(int)

    out["document_ai_exposure_ratio"] = out.apply(
        lambda r: safe_div(r["document_ai_sentences_count"], r["document_total_sentences"]), axis=1
    )
    out["document_explicit_ai_exposure_ratio"] = out.apply(
        lambda r: safe_div(r["explicit_ai_sentences"], r["document_total_sentences"]), axis=1
    )
    out["document_ai_expertise_ratio"] = out.apply(
        lambda r: safe_div(r["expertise_ai_sentences"], r["document_total_sentences"]), axis=1
    )
    out["document_ai_expertise_share_of_ai"] = out.apply(
        lambda r: safe_div(r["expertise_ai_sentences"], r["topic_model_ai_sentences"]), axis=1
    )
    out["document_ai_exposure_clean_ratio"] = out.apply(
        lambda r: safe_div(r["exposure_ai_sentences"], r["document_total_sentences"]), axis=1
    )
    out["document_boilerplate_share_of_raw_ai"] = out.apply(
        lambda r: safe_div(r["boilerplate_ai_sentence_rows"], r["raw_ai_sentence_rows"]), axis=1
    )
    out["document_noise_share_of_modeled_ai"] = out.apply(
        lambda r: safe_div(r["excluded_noise_sentences"], r["topic_model_ai_sentences"]), axis=1
    )

    topic_counts = (
        assignments.groupby(["cik", "company_name", "year", "topic_id", "manual_label", "dimension"], dropna=False)
        .size()
        .reset_index(name="topic_sentence_count")
    )
    denominators = base[["cik", "company_name", "year", "document_total_sentences", "document_ai_sentences_count"]]
    topic_counts = topic_counts.merge(denominators, on=["cik", "company_name", "year"], how="left")
    topic_counts["topic_ratio_document"] = topic_counts.apply(
        lambda r: safe_div(r["topic_sentence_count"], r["document_total_sentences"]), axis=1
    )
    topic_counts["topic_share_of_ai"] = topic_counts.apply(
        lambda r: safe_div(r["topic_sentence_count"], r["document_ai_sentences_count"]), axis=1
    )
    return out.sort_values(["year", "cik"]), topic_counts.sort_values(["year", "cik", "topic_id"])


def build_yearly_topic_summary(assignments: pd.DataFrame, filing_df: pd.DataFrame) -> pd.DataFrame:
    denominators = (
        filing_df.groupby("year", dropna=False)
        .agg(
            filings=("accession", "nunique"),
            document_total_sentences=("document_total_sentences", "sum"),
            document_ai_sentences_count=("document_ai_sentences_count", "sum"),
        )
        .reset_index()
    )
    topic_year = (
        assignments.groupby(["year", "topic_id", "manual_label", "dimension"], dropna=False)
        .size()
        .reset_index(name="topic_sentence_count")
    )
    topic_year = topic_year.merge(denominators, on="year", how="left")
    topic_year["topic_ratio_document"] = topic_year.apply(
        lambda r: safe_div(r["topic_sentence_count"], r["document_total_sentences"]), axis=1
    )
    topic_year["topic_share_of_ai"] = topic_year.apply(
        lambda r: safe_div(r["topic_sentence_count"], r["document_ai_sentences_count"]), axis=1
    )
    return topic_year.sort_values(["year", "topic_id"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filing_df = read_document_denominators()
    sentence_df = extract_document_ai_sentences()
    assignments, topic_df, example_df = fit_topics(sentence_df)
    company_year_df, company_year_topic_df = build_company_year_indices(filing_df, sentence_df, assignments)
    yearly_topic_df = build_yearly_topic_summary(assignments, filing_df)

    filing_df.to_csv(OUT_DIR / "document_filing_denominators.csv", index=False)
    sentence_df.to_csv(OUT_DIR / "document_ai_sentences.csv", index=False)
    assignments.to_csv(OUT_DIR / "document_ai_topic_assignments.csv", index=False)
    topic_df.to_csv(OUT_DIR / "document_ai_topics.csv", index=False)
    example_df.to_csv(OUT_DIR / "document_ai_topic_examples.csv", index=False)
    company_year_df.to_csv(OUT_DIR / "document_company_year_exposure_expertise.csv", index=False)
    company_year_topic_df.to_csv(OUT_DIR / "document_company_year_topic_exposure.csv", index=False)
    yearly_topic_df.to_csv(OUT_DIR / "document_yearly_topic_summary.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "document_filings", "value": len(filing_df)},
            {"metric": "document_ai_sentence_rows_raw_deduped", "value": len(sentence_df)},
            {"metric": "boilerplate_ai_sentence_rows", "value": int(sentence_df["boilerplate_noise"].sum())},
            {
                "metric": "boilerplate_ai_sentence_share",
                "value": safe_div(int(sentence_df["boilerplate_noise"].sum()), len(sentence_df)),
            },
            {"metric": "modeled_ai_sentence_rows", "value": len(assignments)},
            {"metric": "topic_count", "value": len(topic_df)},
            {"metric": "company_year_rows", "value": len(company_year_df)},
            {"metric": "explicit_ai_sentence_rows", "value": int(sentence_df["explicit_ai_term"].sum())},
            {
                "metric": "explicit_ai_sentence_share",
                "value": safe_div(int(sentence_df["explicit_ai_term"].sum()), len(sentence_df)),
            },
            {"metric": "topic_sentence_counts", "value": json.dumps(dict(Counter(assignments["manual_label"])))},
        ]
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"Wrote document AI exposure/topic outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
