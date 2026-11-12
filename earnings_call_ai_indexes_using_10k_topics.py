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


ROOT = Path(__file__).resolve().parent
DEFAULT_10K_SENTENCES = ROOT / "analysis" / "document_ai_exposure" / "document_ai_sentences.csv"
RANDOM_SEED = 42
N_TOPICS = 12

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

# This is the same 10-K topic interpretation used for the final document-level
# 10-K AI exposure/expertise work. The earnings-call script below does not fit
# new earnings-call topics; it projects EC sentences onto this 10-K topic space.
TOPIC_INTERPRETATION = {
    0: ("AI applications, analytics, and digital products", "expertise", "ai_expertise"),
    1: ("AI reputational harm and legal liability", "risk_legal_exposure", "ai_risk_governance_exposure"),
    2: ("Generic AI risk-factor language", "risk_general_exposure", "ai_risk_governance_exposure"),
    3: ("AI laws, privacy, and regulatory compliance", "risk_regulatory_exposure", "ai_risk_governance_exposure"),
    4: ("Workforce learning and training false positives", "exclude_noise", "noise"),
    5: ("AI intellectual property exposure", "risk_ip_exposure", "ai_risk_governance_exposure"),
    6: ("Cybersecurity governance and incident response", "risk_cyber_exposure", "ai_risk_governance_exposure"),
    7: ("Core AI and machine-learning capabilities", "expertise", "ai_expertise"),
    8: ("AI-enabled cyber threats and attacks", "risk_cyber_exposure", "ai_risk_governance_exposure"),
    9: ("EU AI Act and formal AI regulation", "risk_regulatory_exposure", "ai_risk_governance_exposure"),
    10: ("AI governance statements and regulator scrutiny", "risk_regulatory_exposure", "ai_risk_governance_exposure"),
    11: ("Generative AI models, tools, and LLMs", "expertise", "ai_expertise"),
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
    sentences = []
    for piece in pieces:
        piece = clean_text(piece)
        if len(piece) >= 25 and len(piece.split()) >= 5:
            sentences.append(piece)
    return sentences


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


def build_10k_topic_model(sentences_path: Path):
    training = pd.read_csv(sentences_path, usecols=["sentence", "boilerplate_noise"], low_memory=False)
    training["sentence"] = training["sentence"].map(clean_text)
    model_df = training.loc[~training["boilerplate_noise"].astype(bool)].copy().reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=10,
        max_df=0.55,
        ngram_range=(1, 2),
        max_features=6000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X_full = vectorizer.fit_transform(model_df["sentence"])
    full_feature_names = np.array(vectorizer.get_feature_names_out())
    keep_mask = np.array([term not in STOP_TERMS and not term.endswith(" ai") for term in full_feature_names])
    X = X_full[:, keep_mask]
    feature_names = full_feature_names[keep_mask]

    nmf = NMF(n_components=N_TOPICS, random_state=RANDOM_SEED, init="nndsvda", max_iter=600)
    W = nmf.fit_transform(X)

    topic_rows = []
    for topic_id in range(N_TOPICS):
        manual_label, dimension, category = TOPIC_INTERPRETATION[topic_id]
        top_term_idx = np.argsort(nmf.components_[topic_id])[::-1][:20]
        terms = [feature_names[i] for i in top_term_idx]
        topic_rows.append(
            {
                "topic_id": topic_id,
                "model_label": topic_label(feature_names, nmf.components_[topic_id]),
                "manual_label_10k": manual_label,
                "dimension_10k": dimension,
                "mutually_exclusive_category": category,
                "training_sentence_count_10k": int((W.argmax(axis=1) == topic_id).sum()),
                "top_terms_10k_model": ", ".join(terms),
            }
        )
    return vectorizer, keep_mask, feature_names, nmf, pd.DataFrame(topic_rows)


def extract_earnings_call_sentences(
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

        sentences = split_sentences(row[text_col])
        ai_count = 0
        boilerplate_count = 0
        weak_noise_count = 0
        for sentence_id, sentence in enumerate(sentences):
            if not AI_PATTERN.search(sentence):
                continue
            ai_count += 1
            boilerplate_noise = bool(TRANSCRIPT_BOILERPLATE_PATTERN.search(sentence))
            weak_false_positive = bool(WEAK_AI_FALSE_POSITIVE_PATTERN.search(sentence))
            boilerplate_count += int(boilerplate_noise)
            weak_noise_count += int(weak_false_positive)
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
                    "boilerplate_noise": boilerplate_noise,
                    "weak_false_positive": weak_false_positive,
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
                "ai_sentences_count": ai_count,
                "boilerplate_ai_sentence_rows": boilerplate_count,
                "weak_false_positive_ai_rows": weak_noise_count,
                "ai_ratio": safe_div(ai_count, len(sentences)),
            }
        )
    return pd.DataFrame(transcript_rows), pd.DataFrame(sentence_rows)


def project_ec_sentences(sentence_df: pd.DataFrame, vectorizer, keep_mask, nmf, topic_df: pd.DataFrame) -> pd.DataFrame:
    if sentence_df.empty:
        return sentence_df

    substantive = sentence_df.loc[
        ~(sentence_df["boilerplate_noise"].astype(bool) | sentence_df["weak_false_positive"].astype(bool))
    ].copy()
    if substantive.empty:
        noise = sentence_df.copy()
        noise["topic_id"] = np.nan
        noise["topic_weight"] = np.nan
        noise["manual_label_10k"] = "Noise"
        noise["dimension_10k"] = "exclude_noise"
        noise["mutually_exclusive_category"] = "noise"
        return noise

    X_full = vectorizer.transform(substantive["sentence"])
    X = X_full[:, keep_mask]
    W = nmf.transform(X)
    substantive["topic_id"] = W.argmax(axis=1)
    substantive["topic_weight"] = W.max(axis=1)
    substantive = substantive.merge(
        topic_df[["topic_id", "model_label", "manual_label_10k", "dimension_10k", "mutually_exclusive_category"]],
        on="topic_id",
        how="left",
    )

    noise = sentence_df.loc[
        sentence_df["boilerplate_noise"].astype(bool) | sentence_df["weak_false_positive"].astype(bool)
    ].copy()
    noise["topic_id"] = np.nan
    noise["topic_weight"] = np.nan
    noise["model_label"] = "Noise"
    noise["manual_label_10k"] = np.where(
        noise["boilerplate_noise"].astype(bool),
        "Transcript boilerplate",
        "Weak AI false positive",
    )
    noise["dimension_10k"] = "exclude_noise"
    noise["mutually_exclusive_category"] = "noise"
    return pd.concat([substantive, noise], ignore_index=True, sort=False)


def build_indexes(transcript_df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        out = transcript_df.copy()
        for col in ["ai_expertise", "ai_risk_governance_exposure", "ai_general_exposure", "noise"]:
            out[col] = 0
    else:
        counts = (
            assignments.groupby(["transcript_id", "mutually_exclusive_category"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ["ai_expertise", "ai_risk_governance_exposure", "ai_general_exposure", "noise"]:
            if col not in counts.columns:
                counts[col] = 0
        out = transcript_df.merge(counts, on="transcript_id", how="left")
        for col in ["ai_expertise", "ai_risk_governance_exposure", "ai_general_exposure", "noise"]:
            out[col] = out[col].fillna(0).astype(int)

    out["original_ai_sentence_occurrences"] = (
        out["ai_expertise"] + out["ai_risk_governance_exposure"] + out["ai_general_exposure"] + out["noise"]
    )

    topic_breadth = (
        assignments.loc[assignments["mutually_exclusive_category"] != "noise"]
        .groupby("transcript_id", dropna=False)["topic_id"]
        .nunique()
        .reset_index(name="topic_breadth")
        if not assignments.empty
        else pd.DataFrame(columns=["transcript_id", "topic_breadth"])
    )
    out = out.merge(topic_breadth, on="transcript_id", how="left")
    out["topic_breadth"] = out["topic_breadth"].fillna(0).astype(int)

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


def write_examples(assignments: pd.DataFrame, out_dir: Path) -> None:
    if assignments.empty:
        pd.DataFrame().to_csv(out_dir / "earnings_call_projected_topic_examples.csv", index=False)
        return
    examples = []
    for topic_id, topic_slice in assignments.dropna(subset=["topic_id"]).groupby("topic_id"):
        topic_slice = topic_slice.sort_values("topic_weight", ascending=False)
        for _, row in topic_slice.head(15).iterrows():
            examples.append(
                {
                    "topic_id": int(topic_id),
                    "manual_label_10k": row["manual_label_10k"],
                    "mutually_exclusive_category": row["mutually_exclusive_category"],
                    "topic_weight": row["topic_weight"],
                    "transcript_id": row["transcript_id"],
                    "company_name": row["company_name"],
                    "ticker": row["ticker"],
                    "call_date": row["call_date"],
                    "section": row["section"],
                    "sentence": row["sentence"],
                }
            )
    pd.DataFrame(examples).to_csv(out_dir / "earnings_call_projected_topic_examples.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create earnings-call AI indexes by projecting transcript AI sentences onto the fixed 10-K NMF topic space."
    )
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
    parser.add_argument(
        "--tenk-sentences",
        default=str(DEFAULT_10K_SENTENCES),
        help="10-K document_ai_sentences.csv used to rebuild the fixed 10-K topic model.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vectorizer, keep_mask, feature_names, nmf, topic_df = build_10k_topic_model(Path(args.tenk_sentences))
    topic_df.to_csv(out_dir / "fixed_10k_topic_dictionary_used_for_projection.csv", index=False)

    df = pd.read_csv(input_path, low_memory=False)
    if args.text_col not in df.columns:
        raise ValueError(f"Text column {args.text_col!r} not found. Available columns: {list(df.columns)}")

    transcript_df, sentence_df = extract_earnings_call_sentences(
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
    assignments = project_ec_sentences(sentence_df, vectorizer, keep_mask, nmf, topic_df)
    indexes = build_indexes(transcript_df, assignments)

    transcript_df.to_csv(out_dir / "earnings_call_transcript_denominators.csv", index=False)
    sentence_df.to_csv(out_dir / "earnings_call_ai_sentences_raw.csv", index=False)
    assignments.to_csv(out_dir / "earnings_call_ai_sentences_projected_to_10k_topics.csv", index=False)
    indexes.to_csv(out_dir / "earnings_call_ai_indexes_using_10k_topics.csv", index=False)
    write_examples(assignments, out_dir)

    summary = pd.DataFrame(
        [
            {"metric": "transcript_rows", "value": len(transcript_df)},
            {"metric": "total_sentences", "value": int(transcript_df["total_sentences"].sum())},
            {"metric": "raw_ai_sentence_occurrences", "value": len(sentence_df)},
            {"metric": "projected_ai_sentence_occurrences", "value": len(assignments)},
            {"metric": "topic_count_10k", "value": len(topic_df)},
            {"metric": "ai_expertise_occurrences", "value": int(indexes["ai_expertise"].sum())},
            {
                "metric": "ai_risk_governance_exposure_occurrences",
                "value": int(indexes["ai_risk_governance_exposure"].sum()),
            },
            {"metric": "ai_general_exposure_occurrences", "value": int(indexes["ai_general_exposure"].sum())},
            {"metric": "noise_occurrences", "value": int(indexes["noise"].sum())},
            {
                "metric": "projected_topic_counts",
                "value": json.dumps(dict(Counter(assignments["manual_label_10k"]))) if not assignments.empty else "{}",
            },
        ]
    )
    summary.to_csv(out_dir / "earnings_call_ai_index_summary_using_10k_topics.csv", index=False)
    print(f"Wrote earnings-call AI indexes using fixed 10-K topics to {out_dir}")


if __name__ == "__main__":
    main()
