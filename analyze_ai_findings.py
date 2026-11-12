from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis"
FILES = sorted(ROOT.glob("results_flat_*.ndjson"))

AI_TERMS = re.compile(
    r"\b(ai|artificial intelligence|generative ai|machine learning|large language model|llm)\b",
    re.IGNORECASE,
)

AI_RISK_THEMES = {
    "Regulation and compliance": re.compile(
        r"\b(regulat\w+|legal|laws|compliance|governance|policy|policies)\b",
        re.IGNORECASE,
    ),
    "Privacy, IP, and confidentiality": re.compile(
        r"\b(privacy|confidential|proprietary|intellectual property|copyright|trade secret|private information|sensitive data)\b",
        re.IGNORECASE,
    ),
    "Cyber, fraud, and misuse": re.compile(
        r"\b(cyber\w*|fraud|bad actors|misappropriate|security breach|attack|attacks|phishing)\b",
        re.IGNORECASE,
    ),
    "Reliability and explainability": re.compile(
        r"\b(incorrect|erroneous|inaccurate|reliab\w+|hallucinat\w+|monitoring|transparent|transparency|explain\w+|documentation)\b",
        re.IGNORECASE,
    ),
    "Bias and fairness": re.compile(
        r"\b(bias|biased|fairness|discriminat\w+)\b",
        re.IGNORECASE,
    ),
}


def period_from_filename(path: Path) -> str:
    return path.stem.replace("results_flat_", "")


def load_summary() -> dict:
    yearly = defaultdict(
        lambda: {
            "filings": 0,
            "filings_with_ai": 0,
            "filings_with_ai_risk": 0,
            "ai_sentences": 0,
            "ai_risk_sentences": 0,
            "item_1_ai": 0,
            "item_1a_ai": 0,
            "item_1c_ai": 0,
            "item_1a_ai_risk": 0,
            "item_1c_ai_risk": 0,
        }
    )
    period_rows = []
    theme_counts = Counter()
    theme_examples: dict[str, list[str]] = defaultdict(list)
    section_ai_counts = Counter()
    section_ai_risk_counts = Counter()
    examples_2024_plus: list[str] = []

    for path in FILES:
        filings = 0
        filings_with_ai = 0
        filings_with_ai_risk = 0
        ai_sentences = 0
        ai_risk_sentences = 0

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                filing_year = (record.get("filing_date") or "")[:4]
                if not filing_year:
                    continue

                filings += 1
                yearly[filing_year]["filings"] += 1

                filing_ai = 0
                filing_ai_risk = 0

                for section_name, section_data in record.get("sections", {}).items():
                    ai_count = section_data.get("ai_count", 0)
                    ai_risk_count = section_data.get("ai_risk_count", 0)

                    ai_sentences += ai_count
                    ai_risk_sentences += ai_risk_count
                    filing_ai += ai_count
                    filing_ai_risk += ai_risk_count

                    yearly[filing_year]["ai_sentences"] += ai_count
                    yearly[filing_year]["ai_risk_sentences"] += ai_risk_count

                    if ai_count:
                        section_ai_counts[section_name] += ai_count
                    if ai_risk_count:
                        section_ai_risk_counts[section_name] += ai_risk_count

                    if section_name == "ITEM_1":
                        yearly[filing_year]["item_1_ai"] += ai_count
                    elif section_name == "ITEM_1A":
                        yearly[filing_year]["item_1a_ai"] += ai_count
                        yearly[filing_year]["item_1a_ai_risk"] += ai_risk_count
                    elif section_name == "ITEM_1C":
                        yearly[filing_year]["item_1c_ai"] += ai_count
                        yearly[filing_year]["item_1c_ai_risk"] += ai_risk_count

                    for sentence_entry in section_data.get("ai_risk_sentences", []):
                        sentence = sentence_entry.get("sentence", "").strip()
                        if not sentence or not AI_TERMS.search(sentence):
                            continue
                        for theme, pattern in AI_RISK_THEMES.items():
                            if pattern.search(sentence):
                                theme_counts[theme] += 1
                                if len(theme_examples[theme]) < 3:
                                    theme_examples[theme].append(sentence)
                        if filing_year >= "2024" and len(examples_2024_plus) < 8:
                            examples_2024_plus.append(sentence)

                if filing_ai:
                    filings_with_ai += 1
                    yearly[filing_year]["filings_with_ai"] += 1
                if filing_ai_risk:
                    filings_with_ai_risk += 1
                    yearly[filing_year]["filings_with_ai_risk"] += 1

        period_rows.append(
            {
                "period": period_from_filename(path),
                "filings": filings,
                "filings_with_ai": filings_with_ai,
                "filings_with_ai_risk": filings_with_ai_risk,
                "pct_filings_with_ai": round(filings_with_ai / filings * 100, 2) if filings else 0,
                "pct_filings_with_ai_risk": round(filings_with_ai_risk / filings * 100, 2) if filings else 0,
                "ai_sentences": ai_sentences,
                "ai_risk_sentences": ai_risk_sentences,
                "ai_risk_share_of_ai_pct": round(ai_risk_sentences / ai_sentences * 100, 2) if ai_sentences else 0,
            }
        )

    yearly_rows = []
    for year in sorted(yearly):
        row = dict(yearly[year])
        filings = row["filings"]
        ai_sentences = row["ai_sentences"]
        ai_risk_sentences = row["ai_risk_sentences"]
        row["year"] = year
        row["pct_filings_with_ai"] = round(row["filings_with_ai"] / filings * 100, 2) if filings else 0
        row["pct_filings_with_ai_risk"] = round(row["filings_with_ai_risk"] / filings * 100, 2) if filings else 0
        row["ai_risk_share_of_ai_pct"] = round(ai_risk_sentences / ai_sentences * 100, 2) if ai_sentences else 0
        row["item_1a_share_of_ai_risk_pct"] = round(row["item_1a_ai_risk"] / ai_risk_sentences * 100, 2) if ai_risk_sentences else 0
        row["item_1c_share_of_ai_risk_pct"] = round(row["item_1c_ai_risk"] / ai_risk_sentences * 100, 2) if ai_risk_sentences else 0
        yearly_rows.append(row)

    return {
        "period_rows": period_rows,
        "yearly_rows": yearly_rows,
        "section_ai_counts": section_ai_counts.most_common(10),
        "section_ai_risk_counts": section_ai_risk_counts.most_common(10),
        "theme_counts": theme_counts.most_common(),
        "theme_examples": theme_examples,
        "examples_2024_plus": examples_2024_plus,
    }


def render_markdown(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# AI Findings Summary")
    lines.append("")
    lines.append("## Headline Findings")
    lines.append("")

    period_rows = summary["period_rows"]
    first = period_rows[0]
    last = period_rows[-1]
    lines.append(
        f"- AI mention prevalence nearly doubled from {first['pct_filings_with_ai']}% of filings in `{first['period']}` to {last['pct_filings_with_ai']}% in `{last['period']}`."
    )
    lines.append(
        f"- Explicit AI-risk prevalence increased more sharply, from {first['pct_filings_with_ai_risk']}% of filings in `{first['period']}` to {last['pct_filings_with_ai_risk']}% in `{last['period']}`."
    )
    lines.append(
        f"- AI-risk language shifted from being a small subset of AI discussion ({first['ai_risk_share_of_ai_pct']}%) to a major share ({last['ai_risk_share_of_ai_pct']}%), showing that AI is increasingly framed as a governance issue rather than only a product or capability."
    )

    latest_full_year = next(row for row in reversed(summary["yearly_rows"]) if row["year"] == "2025")
    lines.append(
        f"- In 2025, `ITEM_1A` (Risk Factors) accounted for {latest_full_year['item_1a_share_of_ai_risk_pct']}% of all AI-risk sentences, indicating that AI discussion is now concentrated in formal risk disclosures."
    )
    lines.append(
        f"- `ITEM_1C` (Cybersecurity) begins to appear in 2024-2025, but remains secondary: only {latest_full_year['item_1c_share_of_ai_risk_pct']}% of 2025 AI-risk sentences landed there."
    )
    lines.append("")

    lines.append("## Period Trend")
    lines.append("")
    lines.append("| Period | Filings | % with AI | % with AI risk | AI sentences | AI-risk share of AI |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in period_rows:
        lines.append(
            f"| {row['period']} | {row['filings']:,} | {row['pct_filings_with_ai']}% | {row['pct_filings_with_ai_risk']}% | {row['ai_sentences']:,} | {row['ai_risk_share_of_ai_pct']}% |"
        )
    lines.append("")

    lines.append("## Yearly Highlights")
    lines.append("")
    lines.append("| Year | % with AI | % with AI risk | AI-risk share of AI | `ITEM_1A` share of AI risk |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in summary["yearly_rows"]:
        lines.append(
            f"| {row['year']} | {row['pct_filings_with_ai']}% | {row['pct_filings_with_ai_risk']}% | {row['ai_risk_share_of_ai_pct']}% | {row['item_1a_share_of_ai_risk_pct']}% |"
        )
    lines.append("")

    lines.append("## Dominant AI-Risk Themes")
    lines.append("")
    for theme, count in summary["theme_counts"]:
        lines.append(f"- {theme}: {count:,} matched AI-risk sentences")
        for example in summary["theme_examples"][theme][:2]:
            lines.append(f"  - Example: {example}")
    lines.append("")

    lines.append("## Representative 2024-2026 AI-Risk Language")
    lines.append("")
    for sentence in summary["examples_2024_plus"][:6]:
        lines.append(f"- {sentence}")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Early AI mentions are often descriptive: firms present AI as a capability, product feature, or innovation signal.")
    lines.append("- Starting around 2023 and accelerating in 2024-2025, filings increasingly treat AI as a standalone enterprise risk topic.")
    lines.append("- The strongest risk framing clusters around regulation, privacy and IP, cyber misuse, and model reliability rather than only direct cybersecurity.")
    lines.append("- This suggests a useful thesis for your report: AI risk disclosure is maturing from optional innovation language into standardized governance and risk-factor language.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = load_summary()
    json_path = OUTPUT_DIR / "ai_findings_summary.json"
    md_path = OUTPUT_DIR / "ai_findings_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
