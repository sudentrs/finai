import re
import html
from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple

# Optional HTML parser; recommended.
try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4 lxml
    _HAVE_BS4 = True
except Exception:
    BeautifulSoup = None
    _HAVE_BS4 = False


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class EdgarDocument:
    doc_type: str
    sequence: Optional[str]
    filename: Optional[str]
    description: Optional[str]
    text: str  # raw <TEXT> payload


# ----------------------------
# Low-level SGML-ish extraction
# ----------------------------
_DOC_BLOCK_RE = re.compile(r"<DOCUMENT>\s*(.*?)\s*</DOCUMENT>", re.IGNORECASE | re.DOTALL)
_TAG_VALUE_RE = re.compile(rf"<{{tag}}>\s*(.*?)\s*(?=<[A-Z]|$)", re.IGNORECASE | re.DOTALL)
_TEXT_RE = re.compile(r"<TEXT>\s*(.*?)\s*</TEXT>", re.IGNORECASE | re.DOTALL)

def _extract_tag(block: str, tag: str) -> Optional[str]:
    # EDGAR "SGML" tags are not strict XML; values often run until next tag.
    # We emulate that with a regex that stops at next "<UPPERCASE" tag.
    rx = re.compile(rf"<{re.escape(tag)}>\s*(.*?)\s*(?=<[A-Z][A-Z0-9\-]*>|$)",
                    re.IGNORECASE | re.DOTALL)
    m = rx.search(block)
    if not m:
        return None
    return m.group(1).strip()

def parse_submission_txt(raw: str) -> List[EdgarDocument]:
    docs: List[EdgarDocument] = []
    for m in _DOC_BLOCK_RE.finditer(raw):
        block = m.group(1)

        doc_type = (_extract_tag(block, "TYPE") or "").strip()
        sequence = _extract_tag(block, "SEQUENCE")
        filename = _extract_tag(block, "FILENAME")
        description = _extract_tag(block, "DESCRIPTION")

        text_m = _TEXT_RE.search(block)
        text_payload = text_m.group(1) if text_m else ""

        docs.append(
            EdgarDocument(
                doc_type=doc_type,
                sequence=sequence,
                filename=filename,
                description=description,
                text=text_payload,
            )
        )
    return docs


# ----------------------------
# Choose primary 10-K/10-Q doc
# ----------------------------
_PRIMARY_TYPES = {"10-K", "10-K/A", "10-Q", "10-Q/A"}

def choose_primary_document(docs: List[EdgarDocument]) -> Optional[EdgarDocument]:
    """
    Heuristic:
      1) Prefer doc_type in {10-K,10-Q,...}
      2) Prefer lowest SEQUENCE if present
      3) Prefer HTML-like payloads over pure XML exhibits
      4) Fall back to largest TEXT among matching types
    """
    candidates = [d for d in docs if d.doc_type.upper() in _PRIMARY_TYPES]
    if not candidates:
        return None

    def seq_key(d: EdgarDocument) -> int:
        try:
            return int(d.sequence) if d.sequence else 10**9
        except Exception:
            return 10**9

    def looks_like_html(s: str) -> bool:
        s2 = s.lstrip().lower()
        return s2.startswith("<html") or "<body" in s2 or "<div" in s2 or "<p" in s2

    def looks_like_ixbrl(s: str) -> bool:
        s2 = s.lower()
        return "<ix:" in s2 or "inlineXBRL".lower() in s2 or "xmlns:ix" in s2

    # score higher = better
    def score(d: EdgarDocument) -> Tuple[int, int, int]:
        s = d.text or ""
        html_score = 2 if looks_like_html(s) else 0
        ix_score = 1 if looks_like_ixbrl(s) else 0  # still OK; it's the filing
        size_score = min(len(s), 10_000_000)  # cap for safety
        return (html_score + ix_score, -seq_key(d), size_score)

    return sorted(candidates, key=score, reverse=True)[0]


# ----------------------------
# HTML / iXBRL cleanup
# ----------------------------
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")

def html_to_text(payload: str, drop_tables: bool = True) -> str:
    """
    Cleans EDGAR HTML/iXBRL into readable text.

    - Removes script/style/noscript
    - Removes hidden iXBRL header blocks that often include tons of tags
    - Optionally drops tables (financial statements) to reduce numeric noise
    """
    if not payload:
        return ""

    # Strip common wrappers found in newer filings: <XBRL> ... </XBRL>
    # (keeps inner HTML)
    payload2 = re.sub(r"^\s*<XBRL>\s*", "", payload, flags=re.IGNORECASE)
    payload2 = re.sub(r"\s*</XBRL>\s*$", "", payload2, flags=re.IGNORECASE)

    if _HAVE_BS4:
        soup = BeautifulSoup(payload2, "lxml")

        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Remove inline-XBRL hidden/header sections if present
        # Example: <div style="display:none"><ix:header>...</ix:header></div>
        # We drop ix:header and ix:hidden entirely, but keep other visible content.
        for tag in soup.find_all(True):
            name = (tag.name or "").lower()
            if name in {"ix:header", "ix:hidden"}:
                tag.decompose()

        # Remove "display:none" blocks (common for metadata)
        for tag in soup.find_all(style=True):
            style = (tag.get("style") or "").replace(" ", "").lower()
            if "display:none" in style or "visibility:hidden" in style:
                tag.decompose()

        if drop_tables:
            for t in soup.find_all("table"):
                t.decompose()

        # Convert line-breaking tags to newlines to preserve paragraph boundaries
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for p in soup.find_all(["p", "div", "tr", "li"]):
            # ensure separation
            if p.string is None:
                p.append("\n")

        text = soup.get_text(separator=" ", strip=False)
    else:
        # Fallback: crude tag stripper (less accurate than BeautifulSoup)
        text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", payload2)
        text = re.sub(r"(?is)<br\s*/?>", "\n", text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)

    # Decode HTML entities and normalize whitespace
    text = html.unescape(text)
    text = text.replace("\xa0", " ")  # nbsp
    text = text.replace("\u200b", "")  # zero-width
    text = _WHITESPACE_RE.sub(" ", text)
    # Re-introduce newlines where they exist meaningfully
    text = text.replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n")
    text = _MULTI_NL_RE.sub("\n\n", text)

    # Remove EDGAR "page break" artifacts if present
    text = re.sub(r"\bPAGE-BREAK(-AFTER)?\b", " ", text, flags=re.IGNORECASE)

    # Final trim
    return text.strip()


# ----------------------------
# Sentence splitting
# ----------------------------
# A conservative abbreviation list to reduce false splits.
_ABBREV = {
    "inc", "ltd", "co", "corp", "dr", "mr", "mrs", "ms", "vs", "etc",
    "e.g", "i.e", "u.s", "u.k", "no", "fig", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

_SENT_BOUNDARY_RE = re.compile(r"([.!?])(\s+)(?=[A-Z0-9\"“‘(\[])")

def split_sentences(text: str, min_len: int = 2) -> List[str]:
    """
    Rule-based sentence splitter tuned for SEC narrative text.
    - Splits on . ! ? when followed by whitespace + capital/number/quote/paren
    - Avoids splitting after known abbreviations and common SEC patterns
    """
    if not text:
        return []

    # Normalize newlines to spaces but keep paragraph boundaries as mild separators
    t = re.sub(r"\n{2,}", "\n", text)
    t = t.replace("\n", " ")

    parts: List[str] = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(t):
        end_punct = m.start(1)
        end = m.end(1)

        # Look back for abbreviation token
        window = t[max(0, end_punct - 12): end_punct].lower()
        # Get last token-ish
        last = re.findall(r"([a-z]\.?){1,10}$", window)
        last_tok = last[-1].strip(".") if last else ""

        # Don’t split for abbreviations like "U.S." or "Inc."
        if last_tok in _ABBREV:
            continue

        # Don’t split inside decimals like "3.14"
        if end_punct > 0 and t[end_punct - 1].isdigit():
            if end < len(t) and t[end].isdigit():
                continue

        sent = t[start:end].strip()
        if len(sent) >= min_len:
            parts.append(sent)
        start = m.end(1) + len(m.group(2))  # skip following whitespace

    tail = t[start:].strip()
    if len(tail) >= min_len:
        parts.append(tail)

    return parts


# ----------------------------
# High-level convenience API
# ----------------------------
def submission_to_sentences(
    raw_submission_txt: str,
    drop_tables: bool = True,
) -> List[str]:
    docs = parse_submission_txt(raw_submission_txt)
    primary = choose_primary_document(docs)
    if primary is None:
        return []
    cleaned = html_to_text(primary.text, drop_tables=drop_tables)
    print("Cleaned text length:", len(cleaned))
    return split_sentences(cleaned)


# ----------------------------
# Example CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import json

    ap = argparse.ArgumentParser(description="Parse EDGAR .txt submissions into sentences.")
    ap.add_argument("input", help="Path to an EDGAR submission .txt, or a directory of .txt files")
    ap.add_argument("--drop-tables", action="store_true", help="Drop <table> blocks to reduce numeric noise")
    ap.add_argument("--keep-tables", action="store_true", help="Keep tables (overrides --drop-tables)")
    ap.add_argument("--jsonl-out", default=None, help="Write one JSON per line to this file (sentences per submission)")
    args = ap.parse_args()

    drop_tables = True
    if args.keep_tables:
        drop_tables = False
    elif args.drop_tables:
        drop_tables = True

    in_path = Path(args.input)
    files = [in_path] if in_path.is_file() else sorted(in_path.glob("*.txt"))

    out_f = open(args.jsonl_out, "w", encoding="utf-8") if args.jsonl_out else None
    try:
        for fp in files:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
            sents = submission_to_sentences(raw, drop_tables=drop_tables)
            rec = {"file": str(fp), "num_sentences": len(sents), "sentences": sents}

            if out_f:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                print(f"{fp}: {len(sents)} sentences")
                # print first 5 as a sanity check
                for s in sents[:5]:
                    print(" -", s[:200])
                print()
    finally:
        if out_f:
            out_f.close()
