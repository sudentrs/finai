from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


RANDOM_SEED = 42

AI_PATTERN = re.compile(
    r"\b("
    r"ai|artificial intelligence|machine learning|generative ai|genai|large language model|"
    r"large language models|llm|llms|deep learning|neural network|neural networks|"
    r"predictive analytics|computer vision|natural language processing|nlp|chatgpt|"
    r"automation|automated decision|algorithmic|algorithm|algorithms"
    r")\b",
    re.IGNORECASE,
)

WEAK_AI_FALSE_POSITIVE_PATTERN = re.compile(
    r"\b(training and development|learning and development|distance learning|learning curve|"
    r"continuing education|automated teller|automation equipment)\b",
    re.IGNORECASE,
)

TRANSCRIPT_BOILERPLATE_PATTERN = re.compile(
    r"\b("
    r"safe harbor|forward-looking statements|non-gaap|operator instructions|"
    r"question-and-answer session|conference call|webcast|replay|transcript|"
    r"copyright|all rights reserved|seeking alpha|motley fool|edited transcript"
    r")\b",
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
    "customer",
    "customers",
    "product",
    "products",
    "service",
    "services",
    "solution",
    "solutions",
    "platform",
    "platforms",
    "data",
    "quarter",
    "year",
    "think",
    "going",
    "really",
    "great",
    "including",
}


def clean_text(value: object) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    out = []
    for piece in pieces:
        piece = clean_text(piece)
        if len(piece) >= 25 and len(piece.split()) >= 5:
            out.append(piece)
    return out


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def first_present(row: pd.Series, cols: list[str]) -> str:
    for col in cols:
        if col and col in row.index and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value:
                return value
    return ""


def topic_label(feature_names: np.ndarray, weights: np.ndarray, top_n: int = 4) -> str:
    top_idx = np.argsort(weights)[::-1][:top_n]
    return " / ".join(feature_names[i] for i in top_idx)


def heuristic_category(sentence: str, model_label: str, top_terms: str) -> tuple[str, str]:
    text = f"{sentence} {model_label} {top_terms}".lower()
    if re.search(r"\b(training|employee learning|learning and development|education)\b", text):
        return "noise", "Training/learning false positives"
    if re.search(
        r"\b(regulation|regulatory|compliance|privacy|liability|legal|laws?|governance|"
        r"risk|risks|cyber|security|copyright|intellectual property|ip|responsible ai|"
        r"model risk|data protection|eu ai act)\b",
        text,
    ):
        return "ai_risk_governance_exposure", "AI risk, regulation, governance, privacy, cyber, or legal exposure"
    if re.search(
        r"\b(model|models|llm|llms|generative|machine learning|analytics|algorithm|"
        r"automation|automate|automated|copilot|assistant|platform|product|tool|tools|"
        r"capability|capabilities|inference|training data|computer vision|nlp)\b",
        text,
    ):
        return "ai_expertise", "AI products, capabilities, tools, models, analytics, or automation"
    return "ai_general_exposure", "General AI exposure"


def extract_sentence_rows(
    df: pd.DataFrame,
    text_col: str,
    id_col: str | None,
    company_col: str | None,
    ticker_col: str | None,
    date_col: str | None,
    year_col: str | None,
    quarter_col: str | None,
    section_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transcript_rows = []
    sentence_rows = []

    for idx, row in df.iterrows():
        transcript_id = first_present(row, [id_col or ""]) or f"row_{idx}"
        company = first_present(row, [company_col or "company", "company_name", "conm"])
        ticker = first_present(row, [ticker_col or "ticker", "tic", "symbol"])
        call_date = first_present(row, [date_col or "call_date", "date", "event_date"])
        year = first_present(row, [year_col or "year", "fiscal_year"])
        quarter = first_present(row, [quarter_col or "quarter", "fiscal_quarter"])
        section = first_present(row, [section_col or "section", "call_section"]) or "FULL_CALL"

        full_text = clean_text(row[text_col])
        sentences = split_sentences(full_text)
        ai_sentences = []
        boilerplate_ai = 0
        weak_false_positive = 0

        for sentence_id, sentence in enumerate(sentences):
            is_ai = bool(AI_PATTERN.search(sentence))
            is_weak_noise = bool(WEAK_AI_FALSE_POSITIVE_PATTERN.search(sentence))
            is_boilerplate = bool(TRANSCRIPT_BOILERPLATE_PATTERN.search(sentence))
            if not is_ai:
                continue
            if is_boilerplate:
                boilerplate_ai += 1
            if is_weak_noise:
                weak_false_positive += 1
            ai_sentences.append(sentence)
            sentence_rows.append(
                {
                    "transcript_id": transcript_id,
                    "company_name": company,
                    "ticker": ticker,
                    "call_date": call_date,
                    "year": year,
                    "quarter": quarter,
                    "section": section,
                    "sentence_id": sentence_id,
                    "boilerplate_noise": is_boilerplate,
                    "weak_false_positive": is_weak_noise,
                    "sentence": sentence,
                }
            )

        transcript_rows.append(
            {
                "transcript_id": transcript_id,
                "company_name": company,
                "ticker": ticker,
                "call_date": call_date,
                "year": year,
                "quarter": quarter,
                "section": section,
                "total_sentences": len(sentences),
                "ai_sentences_count": len(ai_sentences),
                "boilerplate_ai_sentence_rows": boilerplate_ai,
                "weak_false_positive_ai_rows": weak_false_positive,
                "ai_ratio": safe_div(len(ai_sentences), len(sentences)),
            }
        )

    return pd.DataFrame(transcript_rows), pd.DataFrame(sentence_rows)


def fit_topics(sentence_df: pd.DataFrame, n_topics: int, min_df: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sentence_df.empty:
        raise ValueError("No AI sentences found. Check text column or AI_PATTERN.")

    model_df = sentence_df.loc[
        ~(sentence_df["boilerplate_noise"].astype(bool) | sentence_df["weak_false_positive"].astype(bool))
    ].copy()
    model_df = model_df.drop_duplicates(subset=["sentence"]).reset_index(drop=True)
    if len(model_df) < max(n_topics * 5, 30):
        raise ValueError(f"Only {len(model_df)} non-noise AI sentences found; too few for topic modeling.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=min_df,
        max_df=0.6,
        ngram_range=(1, 2),
        max_features=5000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vectorizer.fit_transform(model_df["sentence"])
    feature_names = np.array(vectorizer.get_feature_names_out())
    keep_mask = np.array([term not in STOP_TERMS and not term.endswith(" ai") for term in feature_names])
    X = X[:, keep_mask]
    feature_names = feature_names[keep_mask]

    nmf = NMF(n_components=n_topics, random_state=RANDOM_SEED, init="nndsvda", max_iter=600)
    W = nmf.fit_transform(X)
    H = nmf.components_

    dedup_assignments = model_df.copy()
    dedup_assignments["topic_id"] = W.argmax(axis=1)
    dedup_assignments["topic_weight"] = W.max(axis=1)

    topic_rows = []
    example_rows = []
    for topic_id in range(n_topics):
        top_idx = np.argsort(H[topic_id])[::-1][:20]
        terms = [feature_names[i] for i in top_idx]
        model_label = topic_label(feature_names, H[topic_id])
        topic_slice = dedup_assignments[dedup_assignments["topic_id"] == topic_id].sort_values(
            "topic_weight", ascending=False
        )
        example_sentence = str(topic_slice["sentence"].iloc[0]) if len(topic_slice) else ""
        category, category_reason = heuristic_category(example_sentence, model_label, " ".join(terms))

        topic_rows.append(
            {
                "topic_id": topic_id,
                "model_label": model_label,
                "manual_label": model_label,
                "mutually_exclusive_category": category,
                "category_reason": category_reason,
                "sentence_count_dedup_training": int(len(topic_slice)),
                "top_terms": ", ".join(terms),
                "top_sections": json.dumps(dict(Counter(topic_slice["section"]).most_common(8))),
            }
        )

        for _, row in topic_slice.head(15).iterrows():
            example_rows.append(
                {
                    "topic_id": topic_id,
                    "model_label": model_label,
                    "mutually_exclusive_category": category,
                    "topic_weight": row["topic_weight"],
                    "transcript_id": row["transcript_id"],
                    "company_name": row["company_name"],
                    "ticker": row["ticker"],
                    "call_date": row["call_date"],
                    "section": row["section"],
                    "sentence": row["sentence"],
                }
            )

    topic_df = pd.DataFrame(topic_rows)
    example_df = pd.DataFrame(example_rows)
    dedup_assignments = dedup_assignments.merge(
        topic_df[["topic_id", "model_label", "manual_label", "mutually_exclusive_category"]],
        on="topic_id",
        how="left",
    )
    return dedup_assignments, topic_df, example_df


def assign_topics_to_occurrences(sentence_df: pd.DataFrame, dedup_assignments: pd.DataFrame) -> pd.DataFrame:
    lookup = dedup_assignments[
        ["sentence", "topic_id", "topic_weight", "model_label", "manual_label", "mutually_exclusive_category"]
    ].drop_duplicates(subset=["sentence"])
    substantive = sentence_df.loc[
        ~(sentence_df["boilerplate_noise"].astype(bool) | sentence_df["weak_false_positive"].astype(bool))
    ].copy()
    assigned = substantive.merge(lookup, on="sentence", how="left")
    missing = assigned["mutually_exclusive_category"].isna()
    assigned.loc[missing, "mutually_exclusive_category"] = "noise"
    assigned.loc[missing, "manual_label"] = "Unmatched substantive occurrence"

    noise = sentence_df.loc[
        sentence_df["boilerplate_noise"].astype(bool) | sentence_df["weak_false_positive"].astype(bool)
    ].copy()
    noise["topic_id"] = np.nan
    noise["topic_weight"] = np.nan
    noise["model_label"] = "Noise"
    noise["manual_label"] = np.where(
        noise["boilerplate_noise"].astype(bool),
        "Transcript boilerplate",
        "Weak AI false positive",
    )
    noise["mutually_exclusive_category"] = "noise"
    return pd.concat([assigned, noise], ignore_index=True, sort=False)


def build_transcript_indexes(transcript_df: pd.DataFrame, occurrence_assignments: pd.DataFrame) -> pd.DataFrame:
    counts = (
        occurrence_assignments.groupby(["transcript_id", "mutually_exclusive_category"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["ai_expertise", "ai_risk_governance_exposure", "ai_general_exposure", "noise"]:
        if col not in counts.columns:
            counts[col] = 0
    counts["original_ai_sentence_occurrences"] = (
        counts["ai_expertise"] + counts["ai_risk_governance_exposure"] + counts["ai_general_exposure"] + counts["noise"]
    )

    topic_breadth = (
        occurrence_assignments.loc[occurrence_assignments["mutually_exclusive_category"] != "noise"]
        .groupby("transcript_id", dropna=False)["topic_id"]
        .nunique()
        .reset_index(name="topic_breadth")
    )

    out = transcript_df.merge(counts, on="transcript_id", how="left").merge(topic_breadth, on="transcript_id", how="left")
    fill_cols = [
        "ai_expertise",
        "ai_risk_governance_exposure",
        "ai_general_exposure",
        "noise",
        "original_ai_sentence_occurrences",
        "topic_breadth",
    ]
    out[fill_cols] = out[fill_cols].fillna(0).astype(int)

    out["ai_expertise_ratio"] = out.apply(lambda r: safe_div(r["ai_expertise"], r["total_sentences"]), axis=1)
    out["ai_risk_governance_ratio"] = out.apply(
        lambda r: safe_div(r["ai_risk_governance_exposure"], r["total_sentences"]), axis=1
    )
    out["ai_general_exposure_ratio"] = out.apply(lambda r: safe_div(r["ai_general_exposure"], r["total_sentences"]), axis=1)
    out["ai_noise_ratio"] = out.apply(lambda r: safe_div(r["noise"], r["total_sentences"]), axis=1)
    out["ai_expertise_share"] = out.apply(
        lambda r: safe_div(r["ai_expertise"], r["original_ai_sentence_occurrences"]), axis=1
    )
    out["ai_risk_governance_share"] = out.apply(
        lambda r: safe_div(r["ai_risk_governance_exposure"], r["original_ai_sentence_occurrences"]), axis=1
    )
    out["ai_general_exposure_share"] = out.apply(
        lambda r: safe_div(r["ai_general_exposure"], r["original_ai_sentence_occurrences"]), axis=1
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AI expertise/risk indexes from earnings call transcripts.")
    parser.add_argument("--input", required=True, help="Path to input CSV with one row per transcript or transcript section.")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV files.")
    parser.add_argument("--text-col", required=True, help="Column containing transcript text.")
    parser.add_argument("--id-col", default=None, help="Transcript id column. If omitted, row index is used.")
    parser.add_argument("--company-col", default=None, help="Company name column.")
    parser.add_argument("--ticker-col", default=None, help="Ticker column.")
    parser.add_argument("--date-col", default=None, help="Call date column.")
    parser.add_argument("--year-col", default=None, help="Fiscal/calendar year column.")
    parser.add_argument("--quarter-col", default=None, help="Quarter column.")
    parser.add_argument("--section-col", default=None, help="Optional section column, e.g. prepared_remarks or q_and_a.")
    parser.add_argument("--n-topics", type=int, default=10, help="Number of NMF AI topics.")
    parser.add_argument("--min-df", type=int, default=5, help="Minimum document frequency for TF-IDF terms.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    if args.text_col not in df.columns:
        raise ValueError(f"Text column {args.text_col!r} not found. Available columns: {list(df.columns)}")

    transcript_df, sentence_df = extract_sentence_rows(
        df=df,
        text_col=args.text_col,
        id_col=args.id_col,
        company_col=args.company_col,
        ticker_col=args.ticker_col,
        date_col=args.date_col,
        year_col=args.year_col,
        quarter_col=args.quarter_col,
        section_col=args.section_col,
    )
    dedup_assignments, topic_df, example_df = fit_topics(sentence_df, n_topics=args.n_topics, min_df=args.min_df)
    occurrence_assignments = assign_topics_to_occurrences(sentence_df, dedup_assignments)
    indexes = build_transcript_indexes(transcript_df, occurrence_assignments)

    transcript_df.to_csv(out_dir / "earnings_call_transcript_denominators.csv", index=False)
    sentence_df.to_csv(out_dir / "earnings_call_ai_sentences_raw.csv", index=False)
    dedup_assignments.to_csv(out_dir / "earnings_call_ai_topic_assignments_dedup_training.csv", index=False)
    topic_df.to_csv(out_dir / "earnings_call_ai_topic_dictionary.csv", index=False)
    example_df.to_csv(out_dir / "earnings_call_ai_topic_examples.csv", index=False)
    occurrence_assignments.to_csv(out_dir / "earnings_call_ai_sentence_occurrence_assignments.csv", index=False)
    indexes.to_csv(out_dir / "earnings_call_ai_indexes.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "transcript_rows", "value": len(transcript_df)},
            {"metric": "total_sentences", "value": int(transcript_df["total_sentences"].sum())},
            {"metric": "raw_ai_sentence_occurrences", "value": len(sentence_df)},
            {"metric": "dedup_training_ai_sentences", "value": len(dedup_assignments)},
            {"metric": "topic_count", "value": len(topic_df)},
            {"metric": "ai_expertise_occurrences", "value": int(indexes["ai_expertise"].sum())},
            {
                "metric": "ai_risk_governance_exposure_occurrences",
                "value": int(indexes["ai_risk_governance_exposure"].sum()),
            },
            {"metric": "ai_general_exposure_occurrences", "value": int(indexes["ai_general_exposure"].sum())},
            {"metric": "noise_occurrences", "value": int(indexes["noise"].sum())},
            {
                "metric": "topic_category_counts",
                "value": json.dumps(dict(Counter(topic_df["mutually_exclusive_category"]))),
            },
        ]
    )
    summary.to_csv(out_dir / "earnings_call_ai_index_summary.csv", index=False)
    print(f"Wrote earnings call AI index outputs to {out_dir}")


if __name__ == "__main__":
    main()
