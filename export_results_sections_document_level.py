from __future__ import annotations

import json
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

from export_results_sections_sample_style import (
    OUT_DIR,
    SOURCE_NDJSON,
    build_crsp_indexes,
    make_base_row,
    section_counts,
)
from export_section_sample_style import MAX_CRSP_MATCH_GAP_DAYS, load_ticker_map, match_info_to_crsp


DOCUMENT_COLUMNS = [
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
    "matched_date",
    "crsp_permno",
    "crsp_permco",
    "crsp_cusip",
    "ret",
    "retx",
    "crsp_match_gap_days",
    "crsp_match_method",
    "crsp_match_accepted",
    "compustat_ticker",
    "compustat_cusip",
    "compustat_gvkey",
    "compustat_date",
    "compustat_match_gap_days",
    "compustat_source",
    "sic",
    "state",
    "filename",
    "filing_html_index",
    "htm_filing_link",
    "complete_text_filing_link",
    "total_sentences",
    "ai_related_sentences_count",
    "risk_related_sentences_count",
    "ai_risk_sentences_count",
    "ai_ratio",
    "risk_ratio",
    "ai_and_risk_ratio",
    "ai_risk_index",
    "positive_count",
    "negative_count",
    "neutral_count",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "sections_present",
    "section_count",
    "crsp_error",
    "source_accession_full",
]


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def document_counts(sections: dict, record_total: int) -> dict:
    totals = Counter()
    for payload in sections.values():
        counts = section_counts(payload or {})
        totals["section_total"] += int(counts["total_sentences"])
        totals["ai_count"] += int(counts["ai_related_sentences_count"])
        totals["risk_count"] += int(counts["risk_related_sentences_count"])
        totals["ai_risk_count"] += int(counts["ai_risk_sentences_count"])
        totals["positive_count"] += int(counts["positive_count"])
        totals["negative_count"] += int(counts["negative_count"])
        totals["neutral_count"] += int(counts["neutral_count"])

    total_sentences = int(totals["section_total"] or record_total or 0)
    ai_count = int(totals["ai_count"])
    sentiment_total = int(totals["positive_count"] + totals["negative_count"] + totals["neutral_count"])

    return {
        "total_sentences": total_sentences,
        "ai_related_sentences_count": ai_count,
        "risk_related_sentences_count": int(totals["risk_count"]),
        "ai_risk_sentences_count": int(totals["ai_risk_count"]),
        "ai_ratio": safe_div(ai_count, total_sentences),
        "risk_ratio": safe_div(totals["risk_count"], total_sentences),
        "ai_and_risk_ratio": safe_div(totals["ai_risk_count"], total_sentences),
        "ai_risk_index": safe_div(totals["ai_risk_count"], ai_count),
        "positive_count": int(totals["positive_count"]),
        "negative_count": int(totals["negative_count"]),
        "neutral_count": int(totals["neutral_count"]),
        "positive_ratio": safe_div(totals["positive_count"], sentiment_total),
        "negative_ratio": safe_div(totals["negative_count"], sentiment_total),
        "neutral_ratio": safe_div(totals["neutral_count"], sentiment_total),
        "sections_present": ";".join(sorted(sections.keys())),
        "section_count": len(sections),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / "document_level.csv"
    summary_path = OUT_DIR / "document_level_summary.csv"

    ticker_map = load_ticker_map()
    crsp_by_ticker, crsp_by_cusip, compustat_by_cik = build_crsp_indexes()
    crsp_cache: dict[tuple[str, str, str], dict] = {}

    stats = Counter()
    form_counts = Counter()

    with output_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=DOCUMENT_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        with SOURCE_NDJSON.open("r", encoding="utf-8") as src:
            for line in src:
                obj = json.loads(line)
                stats["records_read"] += 1
                sections = obj.get("sections") or {}
                if not sections:
                    stats["records_without_sections"] += 1
                    continue

                row = make_base_row(obj, ticker_map)
                row = match_info_to_crsp(row, crsp_by_ticker, crsp_by_cusip, compustat_by_cik, crsp_cache)
                row.update(document_counts(sections, int(obj.get("total_sentences", 0) or 0)))
                writer.writerow(row)

                stats["rows_written"] += 1
                form_counts[str(row.get("form_type") or "unknown")] += 1
                if row.get("crsp_match_accepted"):
                    stats["accepted_crsp_matches"] += 1
                elif "exceeds" in str(row.get("crsp_error", "")):
                    stats["rejected_far_crsp_rows"] += 1

    summary = pd.DataFrame(
        [
            {"metric": "source_file", "value": SOURCE_NDJSON.name},
            {"metric": "output_file", "value": output_path.name},
            {"metric": "records_read", "value": int(stats["records_read"])},
            {"metric": "document_rows_written", "value": int(stats["rows_written"])},
            {"metric": "records_without_sections_skipped", "value": int(stats["records_without_sections"])},
            {"metric": "document_rows_by_form", "value": json.dumps(dict(sorted(form_counts.items())))},
            {"metric": "accepted_crsp_matches", "value": int(stats["accepted_crsp_matches"])},
            {"metric": "crsp_match_rate", "value": safe_div(stats["accepted_crsp_matches"], stats["rows_written"])},
            {"metric": "rejected_far_crsp_rows", "value": int(stats["rejected_far_crsp_rows"])},
            {"metric": "max_crsp_match_gap_days", "value": MAX_CRSP_MATCH_GAP_DAYS},
        ]
    )
    summary.to_csv(summary_path, index=False)
    print(f"Wrote document-level file to {output_path}")


if __name__ == "__main__":
    main()
