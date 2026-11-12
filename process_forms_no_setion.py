import re
import html
import json
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple
import pandas as pd
from bs4 import BeautifulSoup
import joblib




# -------------------------
# PATHS / CONFIG (FINAL — MATCHES YOUR STRUCTURE)
# -------------------------

# Script location:
# /cta/users/stiras/part2/process2.py

BASE_DIR = Path(__file__).resolve().parent  # /cta/users/stiras/part2

# Where edgar-crawler lives
EDGAR_ROOT = BASE_DIR / "edgar-crawler"

# Where models live
MODELS_ROOT = BASE_DIR.parent / "sec_risk_project" / "models"

# Data folders (from edgar-crawler)
DATA_DIR = EDGAR_ROOT / "datasets"
LOGS_DIR = EDGAR_ROOT / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Extracted filings
EXTRACTED_ROOT = DATA_DIR / "EXTRACTED_FILINGS"

# Sentence output (heavy text)
SENT_OUTPUT_ROOT = DATA_DIR / "outputs_sentences"

# SQLite metrics DB
SQLITE_PATH = DATA_DIR / "edgar_item_metrics.sqlite"

# Models (from sec_risk_project)
RISK_MODEL_PATH = MODELS_ROOT / "risk" / "svm_tfidf_calibrated.joblib"
AI_TFIDF_PATH   = MODELS_ROOT / "ai" / "ai_model_tfidf.pkl"
AI_LR_PATH      = MODELS_ROOT / "ai" / "ai_model_lr.pkl"

# -------------------------
# OUTPUT CONTROLS
# -------------------------
SAVE_INTERMEDIATE_FILES = False

# Sentiment model (downloads from HF if not cached)
SENTIMENT_MODEL_NAME = "soleimanian/financial-roberta-large-sentiment"

# -------------------------
# Data structure
# -------------------------
@dataclass
class EdgarDocument:
    doc_type: str
    sequence: Optional[str]
    filename: Optional[str]
    description: Optional[str]
    text: str

# -------------------------
# SGML extraction
# -------------------------
_DOC_BLOCK_RE = re.compile(r"<DOCUMENT>\s*(.*?)\s*</DOCUMENT>", re.IGNORECASE | re.DOTALL)
_TEXT_RE = re.compile(r"<TEXT>\s*(.*?)\s*</TEXT>", re.IGNORECASE | re.DOTALL)

def find_all_submission_files() -> List[Path]:
    filings_root = SRC_DIR / "sec-edgar-filings"
    return list(filings_root.rglob("full-submission.txt"))


def _extract_tag(block: str, tag: str) -> Optional[str]:
    rx = re.compile(
        rf"<{tag}>\s*(.*?)\s*(?=<[A-Z][A-Z0-9\-]*>|$)",
        re.IGNORECASE | re.DOTALL
    )
    m = rx.search(block)
    return m.group(1).strip() if m else None

def parse_submission_txt(raw: str) -> List[EdgarDocument]:
    docs = []
    for m in _DOC_BLOCK_RE.finditer(raw):
        block = m.group(1)
        docs.append(
            EdgarDocument(
                doc_type=(_extract_tag(block, "TYPE") or "").strip(),
                sequence=_extract_tag(block, "SEQUENCE"),
                filename=_extract_tag(block, "FILENAME"),
                description=_extract_tag(block, "DESCRIPTION"),
                text=(_TEXT_RE.search(block).group(1) if _TEXT_RE.search(block) else "")
            )
        )
    return docs

# -------------------------
# Choose main filing
# -------------------------
_PRIMARY_TYPES = {"10-K", "10-K/A", "10-Q", "10-Q/A"}

def choose_primary_document(docs: List[EdgarDocument]) -> Optional[EdgarDocument]:
    cands = [d for d in docs if d.doc_type.upper() in _PRIMARY_TYPES]
    if not cands:
        return None

    def seq(d):
        try:
            return int(d.sequence)
        except Exception:
            return 1_000_000

    def looks_html(s):
        s = (s or "").lower()
        return "<html" in s or "<body" in s or "<div" in s

    def score(d):
        return (2 if looks_html(d.text) else 0, -seq(d), len(d.text or ""))

    return sorted(cands, key=score, reverse=True)[0]

# -------------------------
# HTML cleanup
# -------------------------
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")

def html_to_text(payload: str, drop_tables: bool = True) -> str:
    soup = BeautifulSoup(payload, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for tag_name in ["ix:header", "ix:hidden"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag in soup.find_all(style=True):
        if tag is None:
            continue
        try:
            st = tag.attrs.get("style", "")
        except Exception:
            continue
        if not st:
            continue
        st = str(st).replace(" ", "").lower()
        if "display:none" in st or "visibility:hidden" in st:
            try:
                tag.decompose()
            except Exception:
                pass

    if drop_tables:
        for t in soup.find_all("table"):
            t.decompose()

    for br in soup.find_all("br"):
        br.replace_with("\n")

    text = soup.get_text(" ", strip=False)
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return text.strip()

# -------------------------
# Sentence splitter
# -------------------------
_ABBREV = {"inc", "ltd", "corp", "co", "u.s", "u.k", "e.g", "i.e", "etc"}
_SENT_RE = re.compile(r"([.!?])\s+(?=[A-Z0-9\"“‘(\[])")

def split_sentences(text: str) -> List[str]:
    t = (text or "").replace("\n", " ")
    out = []
    start = 0

    for m in _SENT_RE.finditer(t):
        end = m.end(1)
        window = t[max(0, end - 15):end].lower()
        if any(window.endswith(a + ".") for a in _ABBREV):
            continue
        sent = t[start:end].strip()
        if len(sent) > 2:
            out.append(sent)
        start = end

    tail = t[start:].strip()
    if len(tail) > 2:
        out.append(tail)
    return out

def submission_to_sentences(raw_txt: str, drop_tables: bool = True) -> List[str]:
    docs = parse_submission_txt(raw_txt)
    main = choose_primary_document(docs)
    if not main:
        return []
    clean = html_to_text(main.text, drop_tables=drop_tables)
    return split_sentences(clean)

# -------------------------
# Company info extraction
# -------------------------
def extract_quarter(period_str: str) -> Optional[str]:
    try:
        dt = datetime.datetime.strptime(period_str, "%Y%m%d").date()
        quarter_ends = {(3,31): "Q1", (6,30): "Q2", (9,30): "Q3", (12,31): "Q4"}
        for (m, d), q in quarter_ends.items():
            target = datetime.date(dt.year, m, d)
            if abs((dt - target).days) <= 3:
                return f"{q} {dt.year}"
    except Exception:
        return None
    return None

def extract_company_info(raw: str) -> dict:
    info: Dict[str, Any] = {}
    header_match = re.search(r"<SEC-HEADER>(.*?)</SEC-HEADER>", raw, re.S | re.I)
    header = header_match.group(1) if header_match else raw

    def find(pattern, key, text=header):
        m = re.search(pattern, text, re.I)
        if m:
            info[key] = m.group(1).strip()

    find(r"COMPANY CONFORMED NAME:\s*([^\n\r]+)", "company_name")
    find(r"CENTRAL INDEX KEY:\s*([0-9]+)", "cik")
    find(r"IRS NUMBER:\s*([0-9]+)", "irs_number")
    find(r"STATE OF INCORPORATION:\s*([A-Z]{2})", "state")
    find(r"FISCAL YEAR END:\s*([0-9]{4})", "fiscal_year_end")
    find(r"FORM TYPE:\s*([^\n\r]+)", "form_type")
    find(r"CONFORMED PERIOD OF REPORT:\s*([0-9]{8})", "period_end")
    find(r"FILED AS OF DATE:\s*([0-9]{8})", "filed_date")

    if "period_end" in info:
        info["quarter"] = extract_quarter(info["period_end"])

    if "company_name" not in info:
        m = re.search(r"([A-Z][A-Za-z0-9.,&\- ]+)\s*\(Exact name of Registrant", raw, re.I)
        if m:
            info["company_name"] = m.group(1).strip()

    if "form_type" not in info:
        m = re.search(r"Form\s+(10-Q(?:/A)?)", raw, re.I)
        if m:
            info["form_type"] = m.group(1).strip()

    if "period_end" not in info:
        m = re.search(r"For the quarterly period ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", raw, re.I)
        if m:
            info["period_end_text"] = m.group(1).strip()

    return info

def get_company_info(path: Path) -> Tuple[dict, str]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        raw = path.read_text(encoding="cp1252", errors="ignore")
    info = extract_company_info(raw)
    return info, raw

# -------------------------
# Ticker match (company_tickers.json)
# -------------------------
def attach_ticker_from_cik(info: dict, tickers_path: Path) -> dict:
    if "cik" not in info or not info["cik"]:
        info["ticker"] = None
        return info

    with tickers_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cik_int = int(info["cik"])

    ticker = None
    for _, v in data.items():
        cik_val = v.get("cik_str")
        try:
            cik_val_int = int(cik_val)
        except Exception:
            cik_val_int = None

        if cik_val_int == cik_int:
            ticker = v.get("ticker")
            break

    info["ticker"] = ticker
    return info

# -------------------------
# CRSP match (FIXED: DataFrame passed in)
# -------------------------
def match_info_to_crsp(info: dict, crsp: pd.DataFrame) -> Tuple[dict, pd.Series]:
    ticker = info.get("ticker")
    if not ticker:
        raise ValueError("Ticker bulunamadı (company_tickers.json eşleşmedi).")

    subset = crsp[(crsp["TICKER"] == ticker) | (crsp["TSYMBOL"] == ticker)].copy()
    if subset.empty:
        raise ValueError(f"Ticker {ticker} CRSP dosyasında bulunamadı.")

    subset["date"] = pd.to_datetime(subset["date"], errors="coerce")

    if "period_end" not in info:
        raise ValueError("info içinde period_end yok (CONFORMED PERIOD OF REPORT bulunamadı).")

    target_date = datetime.datetime.strptime(info["period_end"], "%Y%m%d")

    subset["date_diff"] = (subset["date"] - target_date).abs()
    best_row = subset.loc[subset["date_diff"].idxmin()]

    info["matched_date"] = best_row["date"].strftime("%Y-%m-%d") if pd.notna(best_row["date"]) else None
    info["crsp_permno"] = int(best_row["PERMNO"]) if pd.notna(best_row.get("PERMNO")) else None
    info["crsp_permco"] = int(best_row["PERMCO"]) if pd.notna(best_row.get("PERMCO")) else None
    info["crsp_cusip"] = str(best_row["CUSIP"]) if "CUSIP" in best_row else None
    info["ret"] = best_row.get("RET")
    info["retx"] = best_row.get("RETX")

    return info, best_row

# -------------------------
# Sentiment (Financial-RoBERTa)
# -------------------------
LABEL_KEYS = {"positive": "ppos", "negative": "pneg", "neutral": "pneu"}

def fin_roberta_classify(texts: Union[str, List[str]], device: int = -1, batch: int = 64) -> pd.DataFrame:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    if isinstance(texts, str):
        texts = [texts]

    tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    mod = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)

    clf = pipeline(
        "sentiment-analysis",
        model=mod,
        tokenizer=tok,
        device=device,
        batch_size=batch,
        truncation=True,
        max_length=512
    )

    preds = clf(texts, top_k=None)
    rows = []
    for sent, scores in zip(texts, preds):
        m = {d["label"].strip().lower(): float(d["score"]) for d in scores}
        top_label = max(m, key=m.get)
        row = {"label": top_label.upper(), "confidence": m[top_label], "sentence": sent}
        for k, col in LABEL_KEYS.items():
            row[col] = m.get(k, 0.0)
        rows.append(row)

    cols = ["label", "confidence", "pneg", "pneu", "ppos", "sentence"]
    return pd.DataFrame(rows)[cols]

# -------------------------
# Main
# -------------------------
def main():
    print("son  Full pipeline started (MULTI-FILING MODE)")

    submission_files = find_all_submission_files()
    print(f"🔎 Found {len(submission_files)} full-submission.txt files")

    if not submission_files:
        print("❌ No filings found.")
        return

    # -------------------------
    # MODELS (LOAD ONCE)
    # -------------------------
    risk_bundle = joblib.load(RISK_MODEL_PATH)
    svm = risk_bundle["model"]
    tfidf_risk = risk_bundle["vectorizer"]
    best_th = float(risk_bundle["best_threshold"])

    tfidf_ai = joblib.load(AI_TFIDF_PATH)
    lr = joblib.load(AI_LR_PATH)

    # -------------------------
    # CRSP (LOAD ONCE)  ✅ FIX
    # -------------------------
    crsp_df = pd.read_csv(CRSP_PATH)
    
    # -------------------------
    # RESUME SUPPORT (LOAD ONCE)
    # -------------------------
    results_path = DATA_DIR / "data.csv"
    
    processed_keys = set()
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        if {"cik", "accession"}.issubset(existing_df.columns):
            processed_keys = set(
                zip(
                    existing_df["cik"].astype(str),
                    existing_df["accession"].astype(str)
                )
            )
    
    print(f"↪️  Resume enabled: {len(processed_keys)} filings already processed")


    # -------------------------
    # LOOP OVER FILINGS
    # -------------------------
    for idx, filing_path in enumerate(submission_files, 1):
        print(f"\n[{idx}/{len(submission_files)}] ▶ Processing: {filing_path}")

        try:
            # ---------- PATH METADATA ----------
            rel = filing_path.relative_to(SRC_DIR / "sec-edgar-filings")
            cik, form, accession = rel.parts[:3]
            
            # ---------- RESUME GUARD ----------
            if (str(cik), str(accession)) in processed_keys:
                print("↪️  Already processed, skipping")
                continue


            OUT_DIR = DATA_DIR / "outputs" / cik / form / accession
            OUT_DIR.mkdir(parents=True, exist_ok=True)

            # ---------- 1) LOAD + SENTENCES ----------
            raw_text = filing_path.read_text(encoding="utf-8", errors="ignore")
            sentences = submission_to_sentences(raw_text, drop_tables=True)

            if not sentences:
                print("⚠️ No sentences extracted, skipping")
                continue
            if SAVE_INTERMEDIATE_FILES:
                pd.DataFrame({"sentence": sentences}).to_csv(
                    OUT_DIR / "parsed_sentences.csv", index=False
                )

            # ---------- 2) COMPANY INFO ----------
            info = extract_company_info(raw_text)
            info["cik"] = info.get("cik") or cik
            info["form_from_path"] = form
            info["accession"] = accession

            info = attach_ticker_from_cik(info, TICKERS_PATH)
            try:
                info, _ = match_info_to_crsp(info, crsp_df)   # ✅ FIX
            except Exception as e:
                info["crsp_error"] = str(e)

            # ---------- 3) INFERENCE ----------
            X_risk = tfidf_risk.transform(sentences)
            probs_risk = svm.predict_proba(X_risk)[:, 1]
            risk_labels = (probs_risk >= best_th).astype(int)

            X_ai = tfidf_ai.transform(sentences)
            probs_ai = lr.predict_proba(X_ai)[:, 1]
            ai_labels = (probs_ai >= 0.60).astype(int)

            df_labelled = pd.DataFrame({
                "sentence": sentences,
                "is_risk": risk_labels,
                "is_ai": ai_labels
            })

            if SAVE_INTERMEDIATE_FILES:
                df_labelled.to_csv(
                    OUT_DIR / "labelled_sentences_combined.csv", index=False
                )

            # ---------- 4) SENTIMENT ----------
            ai_sentences_df = df_labelled[df_labelled["is_ai"] == 1].copy()
            df_final = None

            if len(ai_sentences_df) > 0:
                df_sentiment = fin_roberta_classify(
                    ai_sentences_df["sentence"].tolist(),
                    device=-1,
                    batch=64
                )
                df_sentiment = df_sentiment.drop(columns=["sentence"], errors="ignore")

                df_final = pd.concat(
                    [ai_sentences_df.reset_index(drop=True),
                     df_sentiment.reset_index(drop=True)],
                    axis=1
                )

                if SAVE_INTERMEDIATE_FILES:
                    df_final.to_csv(OUT_DIR / "ai_sentiment_output.csv", index=False)
                    df_final[df_final["label"] == "NEGATIVE"].to_csv(OUT_DIR / "ai_negatives.csv", index=False)
                    df_final[df_final["label"] == "POSITIVE"].to_csv(OUT_DIR / "ai_positives.csv", index=False)
                    df_final[df_final["label"] == "NEUTRAL"].to_csv(OUT_DIR / "ai_neutrals.csv", index=False)

            # ---------- 5) METRICS (BİREBİR TEK-FILING) ----------
            total_sentences = len(df_labelled)
            ai_related_sentences_count = int(df_labelled["is_ai"].sum())
            risk_related_sentences_count = int(df_labelled["is_risk"].sum())

            # AI–Risk neighbor logic
            ai_risk_flags = []
            for i2 in range(len(df_labelled)):
                if df_labelled.loc[i2, "is_ai"] == 1:
                    is_risk_here = df_labelled.loc[i2, "is_risk"] == 1
                    is_risk_prev = (i2 > 0 and df_labelled.loc[i2 - 1, "is_risk"] == 1)
                    is_risk_next = (i2 < len(df_labelled) - 1 and df_labelled.loc[i2 + 1, "is_risk"] == 1)
                    ai_risk_flags.append(
                        1 if (is_risk_here or is_risk_prev or is_risk_next) else 0
                    )
                else:
                    ai_risk_flags.append(0)

            df_labelled["ai_risk_flag"] = ai_risk_flags

            ai_risk_sentences_count = int(df_labelled["ai_risk_flag"].sum())
            ai_risk_index = (
                ai_risk_sentences_count / ai_related_sentences_count
                if ai_related_sentences_count > 0 else 0.0
            )

            ai_and_risk_sentences_count = int(
                df_labelled[
                    (df_labelled["is_ai"] == 1) &
                    (df_labelled["is_risk"] == 1)
                ].shape[0]
            )

            ai_ratio = ai_related_sentences_count / total_sentences if total_sentences > 0 else 0.0
            risk_ratio = risk_related_sentences_count / total_sentences if total_sentences > 0 else 0.0
            ai_and_risk_ratio_whole = ai_risk_sentences_count / total_sentences if total_sentences > 0 else 0.0

            # Sentiment counts
            positive_count = negative_count = neutral_count = 0
            positive_ratio = negative_ratio = neutral_ratio = 0.0

            if df_final is not None and len(df_final) > 0:
                sentiment_counts = df_final["label"].value_counts()
                positive_count = int(sentiment_counts.get("POSITIVE", 0))
                negative_count = int(sentiment_counts.get("NEGATIVE", 0))
                neutral_count  = int(sentiment_counts.get("NEUTRAL", 0))

                total_ai_sent = positive_count + negative_count + neutral_count
                if total_ai_sent > 0:
                    positive_ratio = positive_count / total_ai_sent
                    negative_ratio = negative_count / total_ai_sent
                    neutral_ratio  = neutral_count  / total_ai_sent

            # ---------- 6) INFO DICT (FULL) ----------
            info["total_sentences"] = int(total_sentences)
            info["ai_related_sentences_count"] = int(ai_related_sentences_count)
            info["risk_related_sentences_count"] = int(risk_related_sentences_count)
            info["ai_risk_sentences_count"] = int(ai_risk_sentences_count)
            info["ai_risk_index"] = float(ai_risk_index)
            info["positive_count"] = int(positive_count)
            info["negative_count"] = int(negative_count)
            info["neutral_count"] = int(neutral_count)
            info["positive_ratio"] = float(positive_ratio)
            info["negative_ratio"] = float(negative_ratio)
            info["neutral_ratio"] = float(neutral_ratio)
            

            # ---------- 7) SAVE SUMMARY ----------
            if SAVE_INTERMEDIATE_FILES:
              with (OUT_DIR / "summary_info.json").open("w", encoding="utf-8") as f:
                  json.dump(info, f, ensure_ascii=False, indent=2)

            # ---------- 8) APPEND TO data.csv ----------
            results_path = DATA_DIR / "data.csv"
            new_row_df = pd.DataFrame([info])
            
            if results_path.exists():
                old_df = pd.read_csv(results_path)
            
                # 🔒 Guard against duplicate (cik, accession)
                if (
                    (old_df["cik"] == info["cik"]) &
                    (old_df["accession"] == info["accession"])
                ).any():
                    print("↪ Already in data.csv, skipping append")
                else:
                    combined_df = pd.concat([old_df, new_row_df], ignore_index=True)
                    combined_df.to_csv(results_path, index=False)
            else:
                new_row_df.to_csv(results_path, index=False)
            
            print("✅ Filing finished")
        except Exception as e:
            print(f"❌ ERROR in {filing_path}: {e}")
            continue

    print("\n🎉 ALL full-submission.txt FILES PROCESSED")




if __name__ == "__main__":
    main()