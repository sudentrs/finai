from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
NDJSON_FILES = sorted(ROOT.glob("results_flat_*.ndjson"))
CRSP_PATH = ROOT / "crsp" / "j6esgwmmwlq6ktmg.csv"
COMPUSTAT_QUARTERLY_PATH = ROOT / "g7htfs0feibx39p4.csv"
COMPUSTAT_ANNUAL_PATH = ROOT / "ncw9xm2krtoqavrc.csv"
SEC_TICKERS_PATH = ROOT / "company_tickers_sec.json"
OUT_DIR = ROOT / "analysis" / "section_sample_style"
MAX_CRSP_MATCH_GAP_DAYS = 365
SECTION_METRIC_FIELDS = {
    "total_sentences",
    "ai_related_sentences_count",
    "risk_related_sentences_count",
    "ai_risk_sentences_count",
    "ai_ratio",
    "risk_ratio",
    "ai_and_risk_ratio",
    "ai_risk_index",
    "section_name",
}


def load_ticker_map() -> dict[str, dict[str, str]]:
    raw = json.loads(SEC_TICKERS_PATH.read_text(encoding="utf-8"))
    records = list(raw.values()) if isinstance(raw, dict) else raw
    mapping: dict[str, dict[str, str]] = {}
    for rec in records:
        cik = str(rec.get("cik_str", "")).strip().lstrip("0")
        if not cik:
            continue
        if cik not in mapping:
            mapping[cik] = {
                "ticker": str(rec.get("ticker", "") or ""),
                "company_name": str(rec.get("title", "") or ""),
            }
    return mapping


def load_crsp() -> pd.DataFrame:
    crsp = pd.read_csv(CRSP_PATH, dtype={"Ticker": str, "TradingSymbol": str, "CUSIP": str, "HdrCUSIP": str})
    crsp["Ticker"] = crsp["Ticker"].fillna("").astype(str).str.upper()
    crsp["TradingSymbol"] = crsp["TradingSymbol"].fillna("").astype(str).str.upper()
    crsp["CUSIP"] = crsp["CUSIP"].fillna("").astype(str).str.upper()
    crsp["HdrCUSIP"] = crsp["HdrCUSIP"].fillna("").astype(str).str.upper()
    crsp["MthCalDt"] = pd.to_datetime(crsp["MthCalDt"], errors="coerce")
    return crsp


def normalize_cusip(value: object) -> str:
    raw = "".join(ch for ch in str(value or "").upper().strip() if ch.isalnum())
    if not raw or raw == "NAN":
        return ""
    if len(raw) <= 6:
        raw = raw.zfill(9)
    if len(raw) == 7:
        raw = raw.zfill(8)
    return raw[:8] if len(raw) >= 8 else raw


def load_compustat() -> pd.DataFrame:
    frames = []
    for path, source, date_col in [
        (COMPUSTAT_ANNUAL_PATH, "compustat_annual", "datadate"),
        (COMPUSTAT_QUARTERLY_PATH, "compustat_quarterly", "datadate"),
    ]:
        df = pd.read_csv(
            path,
            usecols=["cik", "tic", "datadate", "gvkey", "conm", "cusip"],
            dtype={"cik": str, "tic": str, "gvkey": str, "conm": str, "cusip": str},
        )
        df = df.rename(columns={date_col: "datadate"})
        df["source"] = source
        frames.append(df)
    comp = pd.concat(frames, ignore_index=True)
    comp["cik"] = comp["cik"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    comp["tic"] = comp["tic"].fillna("").astype(str).str.upper().str.strip()
    comp["cusip8"] = comp["cusip"].map(normalize_cusip)
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
    comp = comp[(comp["cik"] != "") & comp["datadate"].notna()]
    return comp.drop_duplicates(subset=["cik", "tic", "cusip8", "gvkey", "datadate", "source"])


def parse_form_type(accession: str) -> str:
    parts = accession.split("_")
    if len(parts) >= 2:
        form = parts[1].replace("10Q", "10-Q").replace("10K", "10-K")
        return form
    return ""


def normalize_form_type(record_form_type: str, accession: str) -> str:
    form_type = str(record_form_type or "").strip().upper()
    if form_type:
        return form_type.replace("10Q", "10-Q").replace("10K", "10-K")
    return parse_form_type(accession)


def short_accession(accession: str) -> str:
    parts = accession.split("_")
    return parts[-1] if parts else accession


def estimate_period_end(filing_date: str, form_type: str) -> pd.Timestamp | pd.NaT:
    dt = pd.to_datetime(filing_date, errors="coerce")
    if pd.isna(dt):
        return pd.NaT

    if form_type == "10-Q":
        quarter_ends = []
        for year in [dt.year - 1, dt.year]:
            quarter_ends.extend(
                [
                    pd.Timestamp(year=year, month=3, day=31),
                    pd.Timestamp(year=year, month=6, day=30),
                    pd.Timestamp(year=year, month=9, day=30),
                    pd.Timestamp(year=year, month=12, day=31),
                ]
            )
        valid = [q for q in quarter_ends if q <= dt]
        return max(valid) if valid else pd.NaT

    if form_type == "10-K":
        year_ends = [pd.Timestamp(year=dt.year - 1, month=12, day=31), pd.Timestamp(year=dt.year, month=12, day=31)]
        valid = [y for y in year_ends if y <= dt]
        return max(valid) if valid else pd.NaT

    return pd.NaT


def quarter_label(period_end: pd.Timestamp | pd.NaT, form_type: str) -> str:
    if pd.isna(period_end):
        return ""
    if form_type == "10-Q":
        return f"Q{((period_end.month - 1) // 3) + 1} {period_end.year}"
    if form_type == "10-K":
        return f"FY {period_end.year}"
    return ""


def nearest_compustat_identifiers(
    cik: str,
    target: pd.Timestamp,
    compustat_by_cik: dict[str, pd.DataFrame],
) -> dict:
    subset = compustat_by_cik.get(cik)
    if subset is None or subset.empty or pd.isna(target):
        return {}
    subset = subset.copy()
    subset["compustat_gap_days"] = (subset["datadate"] - target).abs().dt.days
    near = subset[subset["compustat_gap_days"] <= MAX_CRSP_MATCH_GAP_DAYS]
    if near.empty:
        return {}
    best = near.sort_values(["compustat_gap_days", "source"]).iloc[0]
    return {
        "compustat_ticker": str(best["tic"] or ""),
        "compustat_cusip": str(best["cusip8"] or ""),
        "compustat_gvkey": str(best["gvkey"] or ""),
        "compustat_date": best["datadate"].strftime("%Y-%m-%d"),
        "compustat_match_gap_days": int(best["compustat_gap_days"]),
        "compustat_source": str(best["source"] or ""),
    }


def combine_candidate_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["PERMNO", "MthCalDt", "CUSIP", "Ticker", "TradingSymbol"])


def add_grouped_lookup(
    lookup_parts: dict[str, list[pd.DataFrame]],
    frame: pd.DataFrame,
    column: str,
    normalize_values: bool = False,
) -> None:
    for value, sub in frame.groupby(column):
        key = normalize_cusip(value) if normalize_values else str(value).upper().strip()
        if key:
            lookup_parts[key].append(sub)


def finalize_lookup(lookup_parts: dict[str, list[pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    return {
        key: pd.concat(parts, ignore_index=True).drop_duplicates()
        for key, parts in lookup_parts.items()
        if parts
    }


def match_info_to_crsp(
    info: dict,
    crsp_by_ticker: dict[str, pd.DataFrame],
    crsp_by_cusip: dict[str, pd.DataFrame],
    compustat_by_cik: dict[str, pd.DataFrame],
    cache: dict[tuple[str, str, str], dict],
) -> dict:
    cik = str(info.get("cik", "") or "").lstrip("0")
    period_end = str(info.get("period_end", "") or "")
    sec_ticker = str(info.get("ticker", "") or "").upper().strip()
    cache_key = (cik, sec_ticker, period_end)
    if cache_key in cache:
        info.update(cache[cache_key])
        return info

    target = pd.to_datetime(period_end, format="%Y%m%d", errors="coerce")
    if pd.isna(target):
        cached = {"crsp_error": "invalid period_end"}
        cache[cache_key] = cached
        info.update(cached)
        return info

    comp_ids = nearest_compustat_identifiers(cik, target, compustat_by_cik)
    candidate_frames = []
    candidate_labels = []
    comp_ticker = str(comp_ids.get("compustat_ticker", "") or "").upper().strip()
    comp_cusip = str(comp_ids.get("compustat_cusip", "") or "").upper().strip()

    if comp_cusip:
        candidate_frames.append(crsp_by_cusip.get(comp_cusip, pd.DataFrame()))
        candidate_labels.append(f"compustat_cusip:{comp_cusip}")
    if comp_ticker:
        candidate_frames.append(crsp_by_ticker.get(comp_ticker, pd.DataFrame()))
        candidate_labels.append(f"compustat_ticker:{comp_ticker}")
    if sec_ticker:
        candidate_frames.append(crsp_by_ticker.get(sec_ticker, pd.DataFrame()))
        candidate_labels.append(f"sec_current_ticker:{sec_ticker}")

    subset = combine_candidate_frames(candidate_frames)
    if subset.empty:
        if not sec_ticker and not comp_ticker and not comp_cusip:
            error = "missing ticker and historical identifiers"
        else:
            error = "no CRSP rows for candidate identifiers"
        cached = {
            **comp_ids,
            "crsp_match_method": ";".join(candidate_labels),
            "crsp_match_accepted": False,
            "crsp_error": error,
        }
        cache[cache_key] = cached
        info.update(cached)
        return info

    subset = subset.copy()
    subset["date_diff"] = (subset["MthCalDt"] - target).abs()
    best = subset.loc[subset["date_diff"].idxmin()]
    gap_days = int((best["MthCalDt"] - target).days) if pd.notna(best["MthCalDt"]) else None
    accepted = gap_days is not None and abs(gap_days) <= MAX_CRSP_MATCH_GAP_DAYS

    cached = {
        **comp_ids,
        "matched_date": best["MthCalDt"].strftime("%Y-%m-%d") if pd.notna(best["MthCalDt"]) else "",
        "crsp_permno": int(best["PERMNO"]) if accepted and pd.notna(best["PERMNO"]) else None,
        "crsp_permco": int(best["PERMCO"]) if accepted and pd.notna(best["PERMCO"]) else None,
        "crsp_cusip": str(best["CUSIP"]) if accepted and pd.notna(best["CUSIP"]) else "",
        "ret": best["MthRet"] if accepted and pd.notna(best["MthRet"]) else None,
        "retx": best["MthRetx"] if accepted and pd.notna(best["MthRetx"]) else None,
        "crsp_match_gap_days": gap_days,
        "crsp_match_method": ";".join(candidate_labels),
        "crsp_match_accepted": accepted,
        "crsp_error": "" if accepted else f"nearest CRSP observation exceeds {MAX_CRSP_MATCH_GAP_DAYS} days",
    }
    cache[cache_key] = cached
    info.update(cached)
    return info


def build_row(
    cik: str,
    filing_date: str,
    accession: str,
    record_form_type: str,
    section_name: str,
    section_payload: dict,
    ticker_map: dict[str, dict[str, str]],
    crsp_by_ticker: dict[str, pd.DataFrame],
    crsp_by_cusip: dict[str, pd.DataFrame],
    compustat_by_cik: dict[str, pd.DataFrame],
    cache: dict[tuple[str, str, str], dict],
) -> dict:
    form_type = normalize_form_type(record_form_type, accession)
    period_end_ts = estimate_period_end(filing_date, form_type)
    period_end = period_end_ts.strftime("%Y%m%d") if pd.notna(period_end_ts) else ""
    company_meta = ticker_map.get(cik, {})

    total = int(section_payload.get("total", 0) or 0)
    ai_count = int(section_payload.get("ai_count", 0) or 0)
    risk_count = int(section_payload.get("risk_count", 0) or 0)
    ai_risk_count = int(section_payload.get("ai_risk_count", 0) or 0)

    row = {
        "company_name": company_meta.get("company_name", ""),
        "cik": cik,
        "irs_number": "",
        "form_type": form_type,
        "period_end": period_end,
        "filed_date": pd.to_datetime(filing_date).strftime("%Y%m%d") if filing_date else "",
        "quarter": quarter_label(period_end_ts, form_type),
        "form_from_path": form_type,
        "accession": short_accession(accession),
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
        "total_sentences": total,
        "ai_related_sentences_count": ai_count,
        "risk_related_sentences_count": risk_count,
        "ai_risk_sentences_count": ai_risk_count,
        "ai_ratio": (ai_count / total) if total else 0.0,
        "risk_ratio": (risk_count / total) if total else 0.0,
        "ai_and_risk_ratio": (ai_risk_count / total) if total else 0.0,
        "ai_risk_index": (ai_risk_count / ai_count) if ai_count else 0.0,
        "crsp_error": "",
        "section_name": section_name,
        "source_filing_date": filing_date,
        "source_accession_full": accession,
    }
    return match_info_to_crsp(row, crsp_by_ticker, crsp_by_cusip, compustat_by_cik, cache)


def section_metric_values(section_name: str, section_payload: dict) -> dict:
    total = int(section_payload.get("total", 0) or 0)
    ai_count = int(section_payload.get("ai_count", 0) or 0)
    risk_count = int(section_payload.get("risk_count", 0) or 0)
    ai_risk_count = int(section_payload.get("ai_risk_count", 0) or 0)
    return {
        "total_sentences": total,
        "ai_related_sentences_count": ai_count,
        "risk_related_sentences_count": risk_count,
        "ai_risk_sentences_count": ai_risk_count,
        "ai_ratio": (ai_count / total) if total else 0.0,
        "risk_ratio": (risk_count / total) if total else 0.0,
        "ai_and_risk_ratio": (ai_risk_count / total) if total else 0.0,
        "ai_risk_index": (ai_risk_count / ai_count) if ai_count else 0.0,
        "section_name": section_name,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ticker_map = load_ticker_map()
    crsp = load_crsp()
    compustat = load_compustat()
    compustat_by_cik = {cik: sub for cik, sub in compustat.groupby("cik")}
    crsp_ticker_parts: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for key in ["Ticker", "TradingSymbol"]:
        add_grouped_lookup(crsp_ticker_parts, crsp, key)
    crsp_by_ticker = finalize_lookup(crsp_ticker_parts)

    crsp_cusip_parts: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for key in ["CUSIP", "HdrCUSIP"]:
        add_grouped_lookup(crsp_cusip_parts, crsp, key, normalize_values=True)
    crsp_by_cusip = finalize_lookup(crsp_cusip_parts)
    crsp_cache: dict[tuple[str, str, str], dict] = {}

    rows_by_section: dict[str, list[dict]] = defaultdict(list)
    stats = Counter()
    form_rows = Counter()
    form_records = Counter()

    for path in NDJSON_FILES:
        stats["input_files"] += 1
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                stats["records_read"] += 1
                sections = obj.get("sections") or {}
                if not sections:
                    stats["empty_section_records"] += 1
                    continue
                stats["records_with_sections"] += 1

                cik = str(obj.get("cik", "") or "").lstrip("0")
                filing_date = str(obj.get("filing_date", "") or "")
                accession = str(obj.get("accession", "") or "")
                form_type = normalize_form_type(str(obj.get("form_type", "") or ""), accession)
                form_records[form_type or "unknown"] += 1

                section_items = list(sections.items())
                first_section_name, first_payload = section_items[0]
                first_row = build_row(
                    cik,
                    filing_date,
                    accession,
                    form_type,
                    first_section_name,
                    first_payload,
                    ticker_map,
                    crsp_by_ticker,
                    crsp_by_cusip,
                    compustat_by_cik,
                    crsp_cache,
                )
                filing_base = {key: value for key, value in first_row.items() if key not in SECTION_METRIC_FIELDS}

                for section_name, payload in section_items:
                    if section_name == first_section_name:
                        row = first_row
                    else:
                        row = filing_base.copy()
                        row.update(section_metric_values(section_name, payload))
                    rows_by_section[section_name].append(row)
                    stats["rows_written"] += 1
                    form_rows[row["form_type"] or "unknown"] += 1

    manifest_rows = []
    quality_rows = []
    for section_name, rows in sorted(rows_by_section.items()):
        df = pd.DataFrame(rows)
        out_csv = OUT_DIR / f"{section_name}.csv"
        df.to_csv(out_csv, index=False)
        gap = pd.to_numeric(df.get("crsp_match_gap_days"), errors="coerce")
        matched = df["crsp_match_accepted"].fillna(False).astype(bool)
        rejected_far = df["crsp_error"].fillna("").astype(str).str.contains("exceeds", regex=False)
        manifest_rows.append(
            {
                "section_name": section_name,
                "rows": len(df),
                "output_file": out_csv.name,
                "nonzero_ai_rows": int((df["ai_related_sentences_count"] > 0).sum()),
                "nonzero_ai_risk_rows": int((df["ai_risk_sentences_count"] > 0).sum()),
                "forms_present": ", ".join(sorted(set(df["form_type"].dropna().astype(str)))),
                "crsp_match_rate": float(matched.mean()),
                "accepted_crsp_match_rows": int(matched.sum()),
                "rejected_far_crsp_rows": int(rejected_far.sum()),
                "median_accepted_crsp_gap_days": float(gap[matched].median()) if gap[matched].notna().any() else None,
            }
        )
        quality_rows.append(
            {
                "section_name": section_name,
                "rows": len(df),
                "matched_rows": int(matched.sum()),
                "unmatched_rows": int((~matched).sum()),
                "rejected_far_crsp_rows": int(rejected_far.sum()),
                "no_candidate_crsp_rows": int(
                    df["crsp_error"].fillna("").astype(str).str.contains("no CRSP rows", regex=False).sum()
                ),
                "missing_identifier_rows": int(
                    df["crsp_error"].fillna("").astype(str).str.contains("missing ticker", regex=False).sum()
                ),
                "median_accepted_gap_days": float(gap[matched].median()) if gap[matched].notna().any() else None,
                "p90_abs_accepted_gap_days": float(gap[matched].abs().quantile(0.9)) if gap[matched].notna().any() else None,
                "max_abs_accepted_gap_days": float(gap[matched].abs().max()) if gap[matched].notna().any() else None,
                "max_allowed_gap_days": MAX_CRSP_MATCH_GAP_DAYS,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "manifest.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(OUT_DIR / "crsp_match_quality.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "input_files_read", "value": int(stats["input_files"])},
            {"metric": "records_read", "value": int(stats["records_read"])},
            {"metric": "records_with_sections", "value": int(stats["records_with_sections"])},
            {"metric": "section_files_created", "value": len(rows_by_section)},
            {"metric": "total_section_rows", "value": int(stats["rows_written"])},
            {"metric": "records_with_empty_sections_skipped", "value": int(stats["empty_section_records"])},
            {"metric": "section_records_by_form", "value": json.dumps(dict(sorted(form_records.items())))},
            {"metric": "section_rows_by_form", "value": json.dumps(dict(sorted(form_rows.items())))},
            {"metric": "max_crsp_match_gap_days", "value": MAX_CRSP_MATCH_GAP_DAYS},
            {"metric": "compustat_identifier_rows_loaded", "value": len(compustat)},
        ]
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"Wrote section-based sample-style files to {OUT_DIR}")


if __name__ == "__main__":
    main()
