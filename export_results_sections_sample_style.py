from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from export_section_sample_style import (
    MAX_CRSP_MATCH_GAP_DAYS,
    add_grouped_lookup,
    finalize_lookup,
    load_compustat,
    load_crsp,
    load_ticker_map,
    match_info_to_crsp,
    normalize_cusip,
)


ROOT = Path(__file__).resolve().parent
SOURCE_NDJSON = ROOT / "results_sections.ndjson"
OUT_DIR = ROOT / "analysis" / "results_sections_sample_style"


BASE_COLUMNS = [
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
    "crsp_error",
    "section_name",
    "source_accession_full",
]


def safe_filename(section_name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", section_name.upper()).strip("_")
    return f"{name}.csv"


def normalize_date(value: object) -> str:
    raw = str(value or "").strip()
    if re.fullmatch(r"\d{8}", raw):
        return raw
    if not raw:
        return ""
    dt = pd.to_datetime(raw, errors="coerce")
    return dt.strftime("%Y%m%d") if pd.notna(dt) else ""


def build_crsp_indexes() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    crsp = load_crsp()
    compustat = load_compustat()
    compustat_by_cik = {cik: sub for cik, sub in compustat.groupby("cik")}

    ticker_parts: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for key in ["Ticker", "TradingSymbol"]:
        add_grouped_lookup(ticker_parts, crsp, key)
    crsp_by_ticker = finalize_lookup(ticker_parts)

    cusip_parts: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for key in ["CUSIP", "HdrCUSIP"]:
        add_grouped_lookup(cusip_parts, crsp, key, normalize_values=True)
    crsp_by_cusip = finalize_lookup(cusip_parts)

    return crsp_by_ticker, crsp_by_cusip, compustat_by_cik


def section_counts(payload: dict) -> dict:
    total = int(payload.get("total", 0) or 0)
    ai_count = int(payload.get("ai_count", 0) or 0)
    risk_count = int(payload.get("risk_count", 0) or 0)
    ai_risk_count = int(payload.get("ai_risk_count", 0) or 0)

    return {
        "total_sentences": total,
        "ai_related_sentences_count": ai_count,
        "risk_related_sentences_count": risk_count,
        "ai_risk_sentences_count": ai_risk_count,
        "ai_ratio": (ai_count / total) if total else 0.0,
        "risk_ratio": (risk_count / total) if total else 0.0,
        "ai_and_risk_ratio": (ai_risk_count / total) if total else 0.0,
        "ai_risk_index": (ai_risk_count / ai_count) if ai_count else 0.0,
        "positive_count": int(payload.get("positive_count", 0) or 0),
        "negative_count": int(payload.get("negative_count", 0) or 0),
        "neutral_count": int(payload.get("neutral_count", 0) or 0),
        "positive_ratio": float(payload.get("positive_ratio", 0.0) or 0.0),
        "negative_ratio": float(payload.get("negative_ratio", 0.0) or 0.0),
        "neutral_ratio": float(payload.get("neutral_ratio", 0.0) or 0.0),
    }


def make_base_row(obj: dict, ticker_map: dict[str, dict[str, str]]) -> dict:
    cik = str(obj.get("cik", "") or "").lstrip("0")
    company_meta = ticker_map.get(cik, {})
    period_end = normalize_date(obj.get("period_end"))
    filed_date = normalize_date(obj.get("filed_date") or obj.get("filing_date"))

    row = {
        "company_name": obj.get("company_name") or company_meta.get("company_name", ""),
        "cik": cik,
        "fiscal_year_end": obj.get("fiscal_year_end", ""),
        "form_type": obj.get("form_type", ""),
        "period_end": period_end,
        "filed_date": filed_date,
        "quarter": obj.get("quarter", ""),
        "form_from_path": obj.get("form_from_path") or obj.get("form_type", ""),
        "accession": obj.get("accession", ""),
        "ticker": company_meta.get("ticker", ""),
        "matched_date": "",
        "crsp_permno": None,
        "crsp_permco": None,
        "crsp_cusip": "",
        "ret": None,
        "retx": None,
        "crsp_match_gap_days": None,
        "crsp_match_method": "",
        "crsp_match_accepted": False,
        "compustat_ticker": "",
        "compustat_cusip": "",
        "compustat_gvkey": "",
        "compustat_date": "",
        "compustat_match_gap_days": None,
        "compustat_source": "",
        "sic": obj.get("sic", ""),
        "state": obj.get("state", ""),
        "filename": obj.get("filename", ""),
        "filing_html_index": obj.get("filing_html_index", ""),
        "htm_filing_link": obj.get("htm_filing_link", ""),
        "complete_text_filing_link": obj.get("complete_text_filing_link", ""),
        "crsp_error": "",
        "source_accession_full": obj.get("filename") or obj.get("accession", ""),
    }
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ticker_map = load_ticker_map()
    crsp_by_ticker, crsp_by_cusip, compustat_by_cik = build_crsp_indexes()
    crsp_cache: dict[tuple[str, str, str], dict] = {}

    writers: dict[str, csv.DictWriter] = {}
    handles = {}
    stats = Counter()
    section_stats: dict[str, Counter] = defaultdict(Counter)
    form_rows = Counter()
    sentiment_nonzero = Counter()

    try:
        with SOURCE_NDJSON.open("r", encoding="utf-8") as src:
            for line in src:
                obj = json.loads(line)
                stats["records_read"] += 1
                sections = obj.get("sections") or {}
                if not sections:
                    stats["records_without_sections"] += 1
                    continue

                base = make_base_row(obj, ticker_map)
                base = match_info_to_crsp(base, crsp_by_ticker, crsp_by_cusip, compustat_by_cik, crsp_cache)
                stats["records_with_sections"] += 1

                for section_name, payload in sections.items():
                    row = base.copy()
                    row.update(section_counts(payload or {}))
                    row["section_name"] = section_name

                    output_name = safe_filename(section_name)
                    if output_name not in writers:
                        handle = (OUT_DIR / output_name).open("w", encoding="utf-8", newline="")
                        handles[output_name] = handle
                        writers[output_name] = csv.DictWriter(handle, fieldnames=BASE_COLUMNS, extrasaction="ignore")
                        writers[output_name].writeheader()

                    writers[output_name].writerow(row)
                    stats["rows_written"] += 1
                    form_rows[str(row.get("form_type") or "unknown")] += 1
                    section_stats[section_name]["rows"] += 1
                    if row.get("crsp_match_accepted"):
                        section_stats[section_name]["matched_rows"] += 1
                    elif "exceeds" in str(row.get("crsp_error", "")):
                        section_stats[section_name]["rejected_far_crsp_rows"] += 1
                    if row["ai_related_sentences_count"] > 0:
                        section_stats[section_name]["nonzero_ai_rows"] += 1
                    if row["ai_risk_sentences_count"] > 0:
                        section_stats[section_name]["nonzero_ai_risk_rows"] += 1
                    for key in ["positive_count", "negative_count", "neutral_count"]:
                        if row[key] > 0:
                            sentiment_nonzero[key] += 1
    finally:
        for handle in handles.values():
            handle.close()

    manifest_rows = []
    quality_rows = []
    for section_name, counter in sorted(section_stats.items()):
        rows = int(counter["rows"])
        matched = int(counter["matched_rows"])
        manifest_rows.append(
            {
                "section_name": section_name,
                "rows": rows,
                "output_file": safe_filename(section_name),
                "nonzero_ai_rows": int(counter["nonzero_ai_rows"]),
                "nonzero_ai_risk_rows": int(counter["nonzero_ai_risk_rows"]),
                "crsp_match_rate": matched / rows if rows else 0.0,
                "accepted_crsp_match_rows": matched,
                "rejected_far_crsp_rows": int(counter["rejected_far_crsp_rows"]),
            }
        )
        quality_rows.append(
            {
                "section_name": section_name,
                "rows": rows,
                "matched_rows": matched,
                "unmatched_rows": rows - matched,
                "rejected_far_crsp_rows": int(counter["rejected_far_crsp_rows"]),
                "max_allowed_gap_days": MAX_CRSP_MATCH_GAP_DAYS,
            }
        )

    pd.DataFrame(manifest_rows).to_csv(OUT_DIR / "manifest.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(OUT_DIR / "crsp_match_quality.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "source_file", "value": SOURCE_NDJSON.name},
            {"metric": "records_read", "value": int(stats["records_read"])},
            {"metric": "records_with_sections", "value": int(stats["records_with_sections"])},
            {"metric": "records_without_sections", "value": int(stats["records_without_sections"])},
            {"metric": "section_files_created", "value": len(section_stats)},
            {"metric": "total_section_rows", "value": int(stats["rows_written"])},
            {"metric": "section_rows_by_form", "value": json.dumps(dict(sorted(form_rows.items())))},
            {"metric": "sentiment_nonzero_rows", "value": json.dumps(dict(sorted(sentiment_nonzero.items())))},
            {"metric": "max_crsp_match_gap_days", "value": MAX_CRSP_MATCH_GAP_DAYS},
        ]
    ).to_csv(OUT_DIR / "summary.csv", index=False)

    print(f"Wrote section-based sample-style files to {OUT_DIR}")


if __name__ == "__main__":
    main()
