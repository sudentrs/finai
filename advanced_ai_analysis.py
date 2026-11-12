from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis" / "advanced"
PLOT_DIR = OUTPUT_DIR / "plots"
FILES = sorted(ROOT.glob("results_flat_*.ndjson"))
RANDOM_SEED = 42

SECTION_ORDER = [
    "ITEM_1",
    "ITEM_1A",
    "ITEM_1B",
    "ITEM_1C",
    "ITEM_2",
    "ITEM_3",
    "ITEM_4",
    "ITEM_5",
    "ITEM_6",
    "ITEM_7",
    "ITEM_7A",
    "ITEM_8",
    "ITEM_9",
    "ITEM_9A",
    "ITEM_9B",
    "ITEM_9C",
    "ITEM_10",
    "ITEM_11",
    "ITEM_12",
    "ITEM_13",
    "ITEM_14",
    "ITEM_15",
    "ITEM_16",
]

TEXT_STOPWORDS = {
    "ai",
    "artificial",
    "intelligence",
    "machine",
    "learning",
    "generative",
    "model",
    "models",
    "technology",
    "technologies",
    "system",
    "systems",
    "use",
    "used",
    "using",
    "include",
    "includes",
    "including",
    "company",
    "business",
    "may",
    "risk",
    "risks",
    "data",
    "xbrl",
    "document",
    "taxonomy",
    "linkbase",
    "extension",
    "inline",
    "platform",
    "platforms",
    "solution",
    "solutions",
    "product",
    "products",
    "service",
    "services",
    "customer",
    "customers",
    "offering",
    "offerings",
    "develop",
    "developed",
    "developing",
    "operations",
    "capabilities",
    "tools",
    "software",
    "applications",
    "provided",
    "provide",
    "provides",
    "new",
}

AI_TERM_PATTERN = re.compile(
    r"\b(ai|artificial intelligence|machine learning|generative ai|large language model|llm)\b",
    re.IGNORECASE,
)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def clean_text(text: str) -> str:
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def period_from_name(path: Path) -> str:
    return path.stem.replace("results_flat_", "")


def topic_label(feature_names: np.ndarray, weights: np.ndarray, top_n: int = 3) -> str:
    top_idx = np.argsort(weights)[::-1][:top_n]
    words = [feature_names[i] for i in top_idx]
    return " / ".join(words)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def stream_metrics() -> dict:
    document_rows: list[dict] = []
    doc_year_agg = defaultdict(lambda: {"filings": 0, "total_sentences": 0, "ai_count": 0, "risk_count": 0, "ai_risk_count": 0})
    section_year_agg = defaultdict(lambda: {"section_total": 0, "ai_count": 0, "risk_count": 0, "ai_risk_count": 0, "documents": 0})
    section_overall = defaultdict(lambda: {"section_total": 0, "ai_count": 0, "risk_count": 0, "ai_risk_count": 0, "documents": 0})
    company_year = defaultdict(lambda: {"filings": 0, "total_sentences": 0, "ai_count": 0, "risk_count": 0, "ai_risk_count": 0})
    company_overall = defaultdict(lambda: {"filings": 0, "total_sentences": 0, "ai_count": 0, "risk_count": 0, "ai_risk_count": 0, "years": set()})
    ai_sentences: list[dict] = []
    ai_risk_sentences: list[dict] = []

    section_path = OUTPUT_DIR / "section_metrics.csv"
    with section_path.open("w", newline="", encoding="utf-8") as section_file:
        section_writer = csv.DictWriter(
            section_file,
            fieldnames=[
                "period",
                "filing_date",
                "year",
                "quarter",
                "cik",
                "accession",
                "source",
                "form_type",
                "section",
                "section_total",
                "section_ai_count",
                "section_risk_count",
                "section_ai_risk_count",
                "section_ai_ratio",
                "section_risk_ratio",
                "section_ai_risk_ratio",
            ],
        )
        section_writer.writeheader()

        for path in FILES:
            period = period_from_name(path)
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    filing_date = record.get("filing_date", "")
                    year = filing_date[:4]
                    if not year:
                        continue

                    source = record.get("source", "")
                    form_type = record.get("form_type", "")
                    cik = str(record.get("cik", ""))
                    accession = record.get("accession", "")
                    quarter = record.get("quarter", "")
                    total_sentences = int(record.get("total_sentences", 0) or 0)

                    doc_ai_count = 0
                    doc_risk_count = 0
                    doc_ai_risk_count = 0

                    for section, payload in record.get("sections", {}).items():
                        section_total = int(payload.get("total", 0) or 0)
                        ai_count = int(payload.get("ai_count", 0) or 0)
                        risk_count = int(payload.get("risk_count", 0) or 0)
                        ai_risk_count = int(payload.get("ai_risk_count", 0) or 0)

                        doc_ai_count += ai_count
                        doc_risk_count += risk_count
                        doc_ai_risk_count += ai_risk_count

                        section_writer.writerow(
                            {
                                "period": period,
                                "filing_date": filing_date,
                                "year": year,
                                "quarter": quarter,
                                "cik": cik,
                                "accession": accession,
                                "source": source,
                                "form_type": form_type,
                                "section": section,
                                "section_total": section_total,
                                "section_ai_count": ai_count,
                                "section_risk_count": risk_count,
                                "section_ai_risk_count": ai_risk_count,
                                "section_ai_ratio": safe_div(ai_count, section_total),
                                "section_risk_ratio": safe_div(risk_count, section_total),
                                "section_ai_risk_ratio": safe_div(ai_risk_count, section_total),
                            }
                        )

                        sec_key = (year, section)
                        section_year_agg[sec_key]["section_total"] += section_total
                        section_year_agg[sec_key]["ai_count"] += ai_count
                        section_year_agg[sec_key]["risk_count"] += risk_count
                        section_year_agg[sec_key]["ai_risk_count"] += ai_risk_count
                        section_year_agg[sec_key]["documents"] += 1

                        section_overall[section]["section_total"] += section_total
                        section_overall[section]["ai_count"] += ai_count
                        section_overall[section]["risk_count"] += risk_count
                        section_overall[section]["ai_risk_count"] += ai_risk_count
                        section_overall[section]["documents"] += 1

                        for sent in payload.get("ai_sentences", []):
                            ai_sentences.append(
                                {
                                    "year": year,
                                    "cik": cik,
                                    "section": section,
                                    "sentence": clean_text(sent.get("sentence", "")),
                                }
                            )
                        for sent in payload.get("ai_risk_sentences", []):
                            sentence = clean_text(sent.get("sentence", ""))
                            ai_risk_sentences.append(
                                {
                                    "year": year,
                                    "cik": cik,
                                    "section": section,
                                    "sentence": sentence,
                                }
                            )

                    document_rows.append(
                        {
                            "period": period,
                            "filing_date": filing_date,
                            "year": year,
                            "quarter": quarter,
                            "cik": cik,
                            "accession": accession,
                            "source": source,
                            "form_type": form_type,
                            "total_sentences": total_sentences,
                            "ai_count": doc_ai_count,
                            "risk_count": doc_risk_count,
                            "ai_risk_count": doc_ai_risk_count,
                            "ai_ratio": safe_div(doc_ai_count, total_sentences),
                            "risk_ratio": safe_div(doc_risk_count, total_sentences),
                            "ai_risk_ratio": safe_div(doc_ai_risk_count, total_sentences),
                        }
                    )

                    doc_year_agg[year]["filings"] += 1
                    doc_year_agg[year]["total_sentences"] += total_sentences
                    doc_year_agg[year]["ai_count"] += doc_ai_count
                    doc_year_agg[year]["risk_count"] += doc_risk_count
                    doc_year_agg[year]["ai_risk_count"] += doc_ai_risk_count

                    comp_year_key = (cik, year)
                    company_year[comp_year_key]["filings"] += 1
                    company_year[comp_year_key]["total_sentences"] += total_sentences
                    company_year[comp_year_key]["ai_count"] += doc_ai_count
                    company_year[comp_year_key]["risk_count"] += doc_risk_count
                    company_year[comp_year_key]["ai_risk_count"] += doc_ai_risk_count

                    company_overall[cik]["filings"] += 1
                    company_overall[cik]["total_sentences"] += total_sentences
                    company_overall[cik]["ai_count"] += doc_ai_count
                    company_overall[cik]["risk_count"] += doc_risk_count
                    company_overall[cik]["ai_risk_count"] += doc_ai_risk_count
                    company_overall[cik]["years"].add(year)

    document_df = pd.DataFrame(document_rows)
    document_df.to_csv(OUTPUT_DIR / "document_metrics.csv", index=False)

    yearly_df = (
        pd.DataFrame(
            [
                {
                    "year": year,
                    "filings": payload["filings"],
                    "total_sentences": payload["total_sentences"],
                    "ai_count": payload["ai_count"],
                    "risk_count": payload["risk_count"],
                    "ai_risk_count": payload["ai_risk_count"],
                    "ai_ratio_weighted": safe_div(payload["ai_count"], payload["total_sentences"]),
                    "risk_ratio_weighted": safe_div(payload["risk_count"], payload["total_sentences"]),
                    "ai_risk_ratio_weighted": safe_div(payload["ai_risk_count"], payload["total_sentences"]),
                }
                for year, payload in sorted(doc_year_agg.items())
            ]
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    yearly_df.to_csv(OUTPUT_DIR / "yearly_document_ratios.csv", index=False)

    section_year_df = pd.DataFrame(
        [
            {
                "year": year,
                "section": section,
                "documents": payload["documents"],
                "section_total": payload["section_total"],
                "ai_count": payload["ai_count"],
                "risk_count": payload["risk_count"],
                "ai_risk_count": payload["ai_risk_count"],
                "ai_ratio_weighted": safe_div(payload["ai_count"], payload["section_total"]),
                "risk_ratio_weighted": safe_div(payload["risk_count"], payload["section_total"]),
                "ai_risk_ratio_weighted": safe_div(payload["ai_risk_count"], payload["section_total"]),
            }
            for (year, section), payload in section_year_agg.items()
        ]
    )
    section_year_df["section"] = pd.Categorical(section_year_df["section"], categories=SECTION_ORDER, ordered=True)
    section_year_df = section_year_df.sort_values(["year", "section"]).reset_index(drop=True)
    section_year_df.to_csv(OUTPUT_DIR / "yearly_section_ratios.csv", index=False)

    section_overall_df = pd.DataFrame(
        [
            {
                "section": section,
                "documents": payload["documents"],
                "section_total": payload["section_total"],
                "ai_count": payload["ai_count"],
                "risk_count": payload["risk_count"],
                "ai_risk_count": payload["ai_risk_count"],
                "ai_ratio_weighted": safe_div(payload["ai_count"], payload["section_total"]),
                "risk_ratio_weighted": safe_div(payload["risk_count"], payload["section_total"]),
                "ai_risk_ratio_weighted": safe_div(payload["ai_risk_count"], payload["section_total"]),
            }
            for section, payload in section_overall.items()
        ]
    )
    section_overall_df["section"] = pd.Categorical(section_overall_df["section"], categories=SECTION_ORDER, ordered=True)
    section_overall_df = section_overall_df.sort_values("section").reset_index(drop=True)
    section_overall_df.to_csv(OUTPUT_DIR / "overall_section_ratios.csv", index=False)

    company_year_df = pd.DataFrame(
        [
            {
                "cik": cik,
                "year": year,
                "filings": payload["filings"],
                "total_sentences": payload["total_sentences"],
                "ai_count": payload["ai_count"],
                "risk_count": payload["risk_count"],
                "ai_risk_count": payload["ai_risk_count"],
                "ai_ratio": safe_div(payload["ai_count"], payload["total_sentences"]),
                "risk_ratio": safe_div(payload["risk_count"], payload["total_sentences"]),
                "ai_risk_ratio": safe_div(payload["ai_risk_count"], payload["total_sentences"]),
                "risk_framing_ratio": safe_div(payload["ai_risk_count"], payload["ai_count"]),
            }
            for (cik, year), payload in company_year.items()
        ]
    ).sort_values(["cik", "year"])
    company_year_df.to_csv(OUTPUT_DIR / "company_year_metrics.csv", index=False)

    company_overall_rows = []
    for cik, payload in company_overall.items():
        years = sorted(payload["years"])
        company_slice = company_year_df[company_year_df["cik"] == cik].sort_values("year")
        first_ai_year = None
        first_ai_risk_year = None
        if not company_slice.empty:
            ai_years = company_slice.loc[company_slice["ai_count"] > 0, "year"].tolist()
            ai_risk_years = company_slice.loc[company_slice["ai_risk_count"] > 0, "year"].tolist()
            first_ai_year = ai_years[0] if ai_years else ""
            first_ai_risk_year = ai_risk_years[0] if ai_risk_years else ""
            latest_ai_ratio = company_slice.iloc[-1]["ai_ratio"]
            latest_ai_risk_ratio = company_slice.iloc[-1]["ai_risk_ratio"]
        else:
            latest_ai_ratio = 0.0
            latest_ai_risk_ratio = 0.0

        pre_2024 = company_slice[company_slice["year"].isin(["2021", "2022", "2023"])]
        post_2024 = company_slice[company_slice["year"].isin(["2024", "2025"])]
        ai_surge = post_2024["ai_ratio"].mean() - pre_2024["ai_ratio"].mean() if not post_2024.empty and not pre_2024.empty else 0.0

        company_overall_rows.append(
            {
                "cik": cik,
                "filings": payload["filings"],
                "years_covered": len(years),
                "first_year": years[0] if years else "",
                "last_year": years[-1] if years else "",
                "first_ai_year": first_ai_year,
                "first_ai_risk_year": first_ai_risk_year,
                "total_sentences": payload["total_sentences"],
                "ai_count": payload["ai_count"],
                "risk_count": payload["risk_count"],
                "ai_risk_count": payload["ai_risk_count"],
                "ai_ratio": safe_div(payload["ai_count"], payload["total_sentences"]),
                "risk_ratio": safe_div(payload["risk_count"], payload["total_sentences"]),
                "ai_risk_ratio": safe_div(payload["ai_risk_count"], payload["total_sentences"]),
                "risk_framing_ratio": safe_div(payload["ai_risk_count"], payload["ai_count"]),
                "latest_ai_ratio": latest_ai_ratio,
                "latest_ai_risk_ratio": latest_ai_risk_ratio,
                "ai_surge_2024_2025_vs_2021_2023": ai_surge,
            }
        )

    company_overall_df = pd.DataFrame(company_overall_rows).sort_values(["years_covered", "filings", "ai_count"], ascending=[False, False, False])
    company_overall_df.to_csv(OUTPUT_DIR / "company_behavior_summary.csv", index=False)

    ai_df = pd.DataFrame(ai_sentences).drop_duplicates(subset=["year", "cik", "section", "sentence"]).reset_index(drop=True)
    ai_risk_df = pd.DataFrame(ai_risk_sentences).drop_duplicates(subset=["year", "cik", "section", "sentence"]).reset_index(drop=True)
    ai_df.to_csv(OUTPUT_DIR / "ai_sentences.csv", index=False)
    ai_risk_df.to_csv(OUTPUT_DIR / "ai_risk_sentences.csv", index=False)

    return {
        "document_df": document_df,
        "yearly_df": yearly_df,
        "section_year_df": section_year_df,
        "section_overall_df": section_overall_df,
        "company_year_df": company_year_df,
        "company_overall_df": company_overall_df,
        "ai_df": ai_df,
        "ai_risk_df": ai_risk_df,
    }


def plot_yearly_trends(yearly_df: pd.DataFrame) -> None:
    df = yearly_df.copy()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    palette = {
        "AI ratio": "#1f77b4",
        "Risk ratio": "#6b7280",
        "AI-risk ratio": "#d9485f",
    }

    trend = df.melt(
        id_vars="year",
        value_vars=["ai_ratio_weighted", "risk_ratio_weighted", "ai_risk_ratio_weighted"],
        var_name="metric",
        value_name="value",
    )
    trend["metric"] = trend["metric"].map(
        {
            "ai_ratio_weighted": "AI ratio",
            "risk_ratio_weighted": "Risk ratio",
            "ai_risk_ratio_weighted": "AI-risk ratio",
        }
    )
    sns.lineplot(data=trend, x="year", y="value", hue="metric", marker="o", linewidth=2.8, palette=palette, ax=axes[0])
    axes[0].set_title("Document-Level Ratio Trends")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Weighted ratio")
    axes[0].tick_params(axis="x", rotation=45)

    base = df[["year", "ai_ratio_weighted", "ai_risk_ratio_weighted"]].copy()
    base["risk_framing_share"] = np.where(base["ai_ratio_weighted"] > 0, base["ai_risk_ratio_weighted"] / base["ai_ratio_weighted"], 0)
    sns.barplot(data=base, x="year", y="risk_framing_share", color="#ef8a62", ax=axes[1])
    axes[1].set_title("Share of AI Mentions Framed as AI Risk")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("AI-risk / AI")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("How AI Discussion Changes Over Time", fontsize=16, fontweight="bold")
    fig.savefig(PLOT_DIR / "yearly_ratio_trends.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_section_heatmaps(section_year_df: pd.DataFrame) -> None:
    active_sections = (
        section_year_df.groupby("section")["section_total"].sum().sort_values(ascending=False).head(12).index.tolist()
    )
    df = section_year_df[section_year_df["section"].isin(active_sections)].copy()
    df["section"] = pd.Categorical(df["section"], categories=[s for s in SECTION_ORDER if s in active_sections], ordered=True)
    df = df.sort_values(["section", "year"])

    metrics = [
        ("ai_ratio_weighted", "AI Ratio", ["#f7fbff", "#6baed6", "#08306b"]),
        ("risk_ratio_weighted", "Risk Ratio", ["#f7f7f7", "#969696", "#252525"]),
        ("ai_risk_ratio_weighted", "AI-Risk Ratio", ["#fff5f0", "#fb6a4a", "#67000d"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True)
    for ax, (metric, title, colors) in zip(axes, metrics):
        pivot = df.pivot(index="section", columns="year", values=metric).fillna(0.0)
        cmap = LinearSegmentedColormap.from_list(f"{metric}_cmap", colors)
        sns.heatmap(pivot, cmap=cmap, ax=ax, linewidths=0.4, linecolor="white")
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("")
    fig.suptitle("Where Filings Talk About AI, Risk, and AI-Risk", fontsize=16, fontweight="bold")
    fig.savefig(PLOT_DIR / "section_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def cluster_company_behavior(company_overall_df: pd.DataFrame) -> pd.DataFrame:
    eligible = company_overall_df[(company_overall_df["years_covered"] >= 3) & (company_overall_df["filings"] >= 3)].copy()
    if eligible.empty:
        return eligible

    feature_cols = ["ai_ratio", "ai_risk_ratio", "risk_framing_ratio", "ai_surge_2024_2025_vs_2021_2023", "latest_ai_ratio"]
    X = eligible[feature_cols].fillna(0.0).to_numpy()
    X_scaled = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    n_clusters = 4 if len(eligible) >= 12 else max(2, min(3, len(eligible)))
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=20)
    eligible["cluster"] = model.fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X_scaled)
    eligible["pc1"] = coords[:, 0]
    eligible["pc2"] = coords[:, 1]

    label_df = eligible.groupby("cluster")[feature_cols].mean().reset_index()
    label_map = {}
    used_labels = Counter()
    for _, row in label_df.iterrows():
        if row["risk_framing_ratio"] > 0.5:
            label = "Risk-heavy"
        elif row["ai_surge_2024_2025_vs_2021_2023"] > np.median(label_df["ai_surge_2024_2025_vs_2021_2023"]):
            label = "Late surge"
        elif row["ai_ratio"] > np.median(label_df["ai_ratio"]):
            label = "AI-intensive"
        else:
            label = "Low-signal"
        used_labels[label] += 1
        if used_labels[label] > 1:
            label = f"{label} {used_labels[label]}"
        label_map[int(row["cluster"])] = label
    eligible["cluster_label"] = eligible["cluster"].map(label_map)
    eligible.to_csv(OUTPUT_DIR / "company_behavior_clusters.csv", index=False)
    return eligible


def plot_company_behavior(clustered_df: pd.DataFrame) -> None:
    if clustered_df.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    palette = sns.color_palette("Set2", clustered_df["cluster_label"].nunique())
    sns.scatterplot(
        data=clustered_df,
        x="pc1",
        y="pc2",
        hue="cluster_label",
        size="filings",
        sizes=(20, 300),
        alpha=0.75,
        palette=palette,
        ax=ax,
    )
    top_points = clustered_df.nlargest(18, "ai_risk_count")
    for _, row in top_points.iterrows():
        ax.text(row["pc1"], row["pc2"], row["cik"], fontsize=8, alpha=0.8)
    ax.set_title("Company-Level AI Disclosure Behavior")
    ax.set_xlabel("Behavior axis 1")
    ax.set_ylabel("Behavior axis 2")
    ax.legend(loc="best", frameon=True)
    fig.savefig(PLOT_DIR / "company_behavior_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_company_heatmap(company_year_df: pd.DataFrame) -> None:
    recent = company_year_df[company_year_df["year"].isin(["2024", "2025"])].copy()
    if recent.empty:
        return
    top_ciks = (
        recent.groupby("cik")["ai_risk_count"].sum().sort_values(ascending=False).head(25).index.tolist()
    )
    plot_df = company_year_df[company_year_df["cik"].isin(top_ciks)].copy()
    pivot = plot_df.pivot_table(index="cik", columns="year", values="ai_risk_ratio", aggfunc="mean").fillna(0.0)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("airisk_company", ["#fff5f0", "#fb6a4a", "#67000d"])
    sns.heatmap(pivot, cmap=cmap, linewidths=0.3, linecolor="white", ax=ax)
    ax.set_title("Top Company AI-Risk Trajectories by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("CIK")
    fig.savefig(PLOT_DIR / "company_ai_risk_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def model_topics(
    sentence_df: pd.DataFrame,
    prefix: str,
    n_topics: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    text_df = sentence_df.copy()
    text_df["sentence"] = text_df["sentence"].fillna("").map(clean_text)
    text_df = text_df[text_df["sentence"].str.len() > 30].drop_duplicates(subset=["sentence"]).reset_index(drop=True)
    if text_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=8,
        max_df=0.6,
        ngram_range=(1, 2),
        max_features=3000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vectorizer.fit_transform(text_df["sentence"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    keep_mask = np.array([term not in TEXT_STOPWORDS and not term.endswith(" ai") for term in feature_names])
    X = X[:, keep_mask]
    feature_names = feature_names[keep_mask]

    nmf = NMF(n_components=n_topics, random_state=RANDOM_SEED, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    H = nmf.components_
    topic_ids = W.argmax(axis=1)
    text_df["topic_id"] = topic_ids
    text_df["topic_weight"] = W.max(axis=1)

    topic_rows = []
    rep_rows = []
    for topic_idx in range(n_topics):
        label = topic_label(feature_names, H[topic_idx], top_n=3)
        top_term_idx = np.argsort(H[topic_idx])[::-1][:12]
        terms = [feature_names[i] for i in top_term_idx]
        topic_slice = text_df[text_df["topic_id"] == topic_idx].sort_values("topic_weight", ascending=False)
        topic_rows.append(
            {
                "topic_id": topic_idx,
                "label": label,
                "top_terms": ", ".join(terms),
                "sentence_count": int(len(topic_slice)),
            }
        )
        for _, row in topic_slice.head(5).iterrows():
            rep_rows.append(
                {
                    "topic_id": topic_idx,
                    "label": label,
                    "year": row["year"],
                    "section": row["section"],
                    "cik": row["cik"],
                    "topic_weight": row["topic_weight"],
                    "sentence": row["sentence"],
                }
            )

    topic_df = pd.DataFrame(topic_rows).sort_values("sentence_count", ascending=False).reset_index(drop=True)
    rep_df = pd.DataFrame(rep_rows).sort_values(["topic_id", "topic_weight"], ascending=[True, False]).reset_index(drop=True)

    assigned_path = OUTPUT_DIR / f"{prefix}_topic_assignments.csv"
    text_df.to_csv(assigned_path, index=False)
    topic_df.to_csv(OUTPUT_DIR / f"{prefix}_topics.csv", index=False)
    rep_df.to_csv(OUTPUT_DIR / f"{prefix}_topic_examples.csv", index=False)
    return topic_df, text_df


def plot_topics(ai_topics: pd.DataFrame, airisk_topics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    panels = [
        (ai_topics, "AI Context Topics", "#3182bd", axes[0]),
        (airisk_topics, "AI-Risk Context Topics", "#cb181d", axes[1]),
    ]
    for df, title, color, ax in panels:
        if df.empty:
            ax.axis("off")
            continue
        plot_df = df.copy().sort_values("sentence_count", ascending=True)
        sns.barplot(data=plot_df, x="sentence_count", y="label", color=color, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Sentences assigned to topic")
        ax.set_ylabel("")
    fig.suptitle("Unsupervised Topic Modeling of AI Mentions", fontsize=16, fontweight="bold")
    fig.savefig(PLOT_DIR / "topic_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_section_topic_network(assignments: pd.DataFrame, topics: pd.DataFrame, prefix: str, color: str) -> None:
    if assignments.empty or topics.empty:
        return
    label_map = dict(zip(topics["topic_id"], topics["label"]))
    links = (
        assignments.groupby(["section", "topic_id"]).size().reset_index(name="weight")
        .sort_values("weight", ascending=False)
        .head(35)
    )

    G = nx.Graph()
    for _, row in links.iterrows():
        topic_node = f"topic_{int(row['topic_id'])}"
        section_node = f"section_{row['section']}"
        G.add_node(section_node, kind="section", label=row["section"])
        raw_label = label_map.get(int(row["topic_id"]), str(row["topic_id"]))
        short_label = raw_label if len(raw_label) <= 28 else raw_label[:28].rstrip() + "..."
        G.add_node(topic_node, kind="topic", label=short_label)
        G.add_edge(section_node, topic_node, weight=float(row["weight"]))

    section_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "section"]
    topic_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "topic"]
    pos = {}
    for i, node in enumerate(section_nodes):
        pos[node] = (-1.0, 1 - i * (2 / max(1, len(section_nodes) - 1)))
    for i, node in enumerate(topic_nodes):
        pos[node] = (1.0, 1 - i * (2 / max(1, len(topic_nodes) - 1)))

    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=[0.4 + w / max(weights) * 5 for w in weights], alpha=0.35, edge_color="#7f7f7f", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=section_nodes, node_color="#4d4d4d", node_size=1400, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, node_color=color, node_size=1900, ax=ax)
    labels = {node: data["label"] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color="white", ax=ax)
    ax.set_title(f"{prefix} Section-Topic Network")
    ax.axis("off")
    fig.savefig(PLOT_DIR / f"{prefix.lower().replace(' ', '_')}_network.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_report(metrics: dict, clustered_df: pd.DataFrame, ai_topics: pd.DataFrame, airisk_topics: pd.DataFrame) -> None:
    yearly_df = metrics["yearly_df"]
    section_overall_df = metrics["section_overall_df"].copy()
    section_overall_df["section"] = section_overall_df["section"].astype(str)
    top_ai_sections = section_overall_df.sort_values("ai_ratio_weighted", ascending=False).head(5)
    top_airisk_sections = section_overall_df.sort_values("ai_risk_ratio_weighted", ascending=False).head(5)
    top_companies = clustered_df.sort_values("ai_risk_count", ascending=False).head(12) if not clustered_df.empty else pd.DataFrame()

    lines = []
    lines.append("# Advanced AI Disclosure Analysis")
    lines.append("")
    lines.append("## What Was Produced")
    lines.append("")
    lines.append("- `document_metrics.csv`: document-level `ai_ratio`, `risk_ratio`, `ai_risk_ratio` for every filing.")
    lines.append("- `section_metrics.csv`: section-level `ai_ratio`, `risk_ratio`, `ai_risk_ratio` for every filing-section.")
    lines.append("- Trend, heatmap, company-behavior, topic-model, and network graphics in `analysis/advanced/plots/`.")
    lines.append("")
    lines.append("## Quick Findings")
    lines.append("")
    latest = yearly_df.iloc[-2] if "2026" in yearly_df["year"].values and len(yearly_df) > 1 else yearly_df.iloc[-1]
    earliest = yearly_df.iloc[0]
    lines.append(
        f"- Weighted document-level AI ratio increased from {earliest['ai_ratio_weighted']:.4f} in {earliest['year']} to {latest['ai_ratio_weighted']:.4f} in {latest['year']}."
    )
    lines.append(
        f"- Weighted document-level AI-risk ratio increased from {earliest['ai_risk_ratio_weighted']:.4f} in {earliest['year']} to {latest['ai_risk_ratio_weighted']:.4f} in {latest['year']}."
    )
    lines.append("- AI disclosure is concentrated in a small set of sections, while AI-risk is even more concentrated in formal risk sections.")
    lines.append("")
    lines.append("### Sections With Highest AI Ratios")
    lines.append("")
    for _, row in top_ai_sections.iterrows():
        lines.append(f"- {row['section']}: AI ratio {row['ai_ratio_weighted']:.4f}, AI-risk ratio {row['ai_risk_ratio_weighted']:.4f}")
    lines.append("")
    lines.append("### Sections With Highest AI-Risk Ratios")
    lines.append("")
    for _, row in top_airisk_sections.iterrows():
        lines.append(f"- {row['section']}: AI-risk ratio {row['ai_risk_ratio_weighted']:.4f}, risk ratio {row['risk_ratio_weighted']:.4f}")
    lines.append("")
    if not top_companies.empty:
        lines.append("### Company Behavior Clusters")
        lines.append("")
        for label, sub in top_companies.groupby("cluster_label"):
            lines.append(f"- {label}: {len(sub)} of the highest AI-risk-volume companies in the top slice.")
        lines.append("")
    if not ai_topics.empty:
        lines.append("### AI Topics")
        lines.append("")
        for _, row in ai_topics.head(6).iterrows():
            lines.append(f"- {row['label']}: {int(row['sentence_count']):,} sentences")
        lines.append("")
    if not airisk_topics.empty:
        lines.append("### AI-Risk Topics")
        lines.append("")
        for _, row in airisk_topics.head(6).iterrows():
            lines.append(f"- {row['label']}: {int(row['sentence_count']):,} sentences")
        lines.append("")
    (OUTPUT_DIR / "report_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ensure_dirs()

    metrics = stream_metrics()
    plot_yearly_trends(metrics["yearly_df"])
    plot_section_heatmaps(metrics["section_year_df"])

    clustered_df = cluster_company_behavior(metrics["company_overall_df"])
    plot_company_behavior(clustered_df)
    plot_company_heatmap(metrics["company_year_df"])

    ai_topics, ai_assignments = model_topics(metrics["ai_df"], prefix="ai", n_topics=6)
    airisk_topics, airisk_assignments = model_topics(metrics["ai_risk_df"], prefix="ai_risk", n_topics=6)
    plot_topics(ai_topics, airisk_topics)
    build_section_topic_network(ai_assignments, ai_topics, prefix="AI", color="#3182bd")
    build_section_topic_network(airisk_assignments, airisk_topics, prefix="AI Risk", color="#cb181d")

    make_report(metrics, clustered_df, ai_topics, airisk_topics)
    print(f"Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
