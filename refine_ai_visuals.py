from __future__ import annotations

from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "analysis" / "advanced"
PLOTS = BASE / "plots"
ACADEMIC = PLOTS / "academic"
INFO = PLOTS / "infographic"
MAPPING_PATH = ROOT / "company_tickers.json"

SECTION_LABELS = {
    "ITEM_1": "Item 1\nBusiness",
    "ITEM_1A": "Item 1A\nRisk Factors",
    "ITEM_1B": "Item 1B\nComments",
    "ITEM_1C": "Item 1C\nCybersecurity",
    "ITEM_2": "Item 2\nProperties",
    "ITEM_3": "Item 3\nLegal Proceedings",
    "ITEM_4": "Item 4\nMine Safety",
    "ITEM_5": "Item 5\nMarket",
    "ITEM_6": "Item 6\nSelected Financial Data",
    "ITEM_7": "Item 7\nMD&A",
    "ITEM_7A": "Item 7A\nMarket Risk",
    "ITEM_8": "Item 8\nFinancial Statements",
    "ITEM_9": "Item 9\nChanges/Disagreements",
    "ITEM_9A": "Item 9A\nControls",
    "ITEM_9B": "Item 9B\nOther Information",
    "ITEM_9C": "Item 9C\nDisclosure",
    "ITEM_10": "Item 10\nDirectors",
    "ITEM_11": "Item 11\nCompensation",
    "ITEM_12": "Item 12\nOwnership",
    "ITEM_13": "Item 13\nRelationships",
    "ITEM_14": "Item 14\nFees",
    "ITEM_15": "Item 15\nExhibits",
    "ITEM_16": "Item 16\n10-K Summary",
}


def ensure_dirs() -> None:
    ACADEMIC.mkdir(parents=True, exist_ok=True)
    INFO.mkdir(parents=True, exist_ok=True)


def load_data() -> dict:
    yearly = pd.read_csv(BASE / "yearly_document_ratios.csv", dtype={"year": str})
    section_year = pd.read_csv(BASE / "yearly_section_ratios.csv", dtype={"year": str})
    company_year = pd.read_csv(BASE / "company_year_metrics.csv", dtype={"year": str, "cik": str})
    company_summary = pd.read_csv(BASE / "company_behavior_summary.csv", dtype={"cik": str})
    company_clusters = pd.read_csv(BASE / "company_behavior_clusters.csv", dtype={"cik": str}) if (BASE / "company_behavior_clusters.csv").exists() else pd.DataFrame()
    ai_topics = pd.read_csv(BASE / "ai_topics.csv")
    ai_risk_topics = pd.read_csv(BASE / "ai_risk_topics.csv")
    return {
        "yearly": yearly,
        "section_year": section_year,
        "company_year": company_year,
        "company_summary": company_summary,
        "company_clusters": company_clusters,
        "ai_topics": ai_topics,
        "ai_risk_topics": ai_risk_topics,
    }


def load_company_mapping() -> pd.DataFrame:
    raw = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    records = list(raw.values()) if isinstance(raw, dict) else raw
    df = pd.DataFrame(records)
    df["cik"] = df["cik_str"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["ticker"] = df["ticker"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)

    # Prefer the shortest clean ticker per CIK to avoid odd derivative symbols dominating.
    df["ticker_len"] = df["ticker"].str.len()
    df = (
        df.sort_values(["cik", "ticker_len", "ticker", "title"])
        .drop_duplicates(subset=["cik"], keep="first")
        .drop(columns=["ticker_len"])
        .reset_index(drop=True)
    )
    return df


def enrich_company_frames(data: dict, mapping: pd.DataFrame) -> dict:
    out = dict(data)
    for key in ["company_year", "company_summary", "company_clusters"]:
        if key in out and not out[key].empty:
            out[key] = out[key].merge(mapping[["cik", "ticker", "title"]], on="cik", how="left")
            out[key]["ticker"] = out[key]["ticker"].fillna("")
            out[key]["title"] = out[key]["title"].fillna("")
            out[key]["display_name"] = np.where(
                out[key]["ticker"].ne(""),
                out[key]["ticker"] + " - " + out[key]["title"].replace("", np.nan).fillna(out[key]["cik"]),
                np.where(out[key]["title"].ne(""), out[key]["title"], out[key]["cik"]),
            )
    if not out["company_summary"].empty:
        out["company_summary"].to_csv(BASE / "company_behavior_summary_enriched.csv", index=False)
    if not out["company_year"].empty:
        out["company_year"].to_csv(BASE / "company_year_metrics_enriched.csv", index=False)
    if "company_clusters" in out and not out["company_clusters"].empty:
        out["company_clusters"].to_csv(BASE / "company_behavior_clusters_enriched.csv", index=False)
    return out


def short_company_label(row: pd.Series) -> str:
    ticker = str(row.get("ticker", "") or "").strip()
    title = str(row.get("title", "") or "").strip()
    cik = str(row.get("cik", "") or "").strip()
    if ticker:
        return ticker
    if title:
        return title[:18]
    return cik


def add_section_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["section_label"] = out["section"].map(SECTION_LABELS).fillna(out["section"])
    return out


def clean_topic_label(label: str) -> str:
    parts = [p.strip() for p in str(label).split("/") if p.strip()]
    seen = []
    for part in parts:
        if part not in seen:
            seen.append(part)
    return ", ".join(seen[:3]).title()


def plot_academic_trends(yearly: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    df = yearly[yearly["year"] != "2026"].copy()

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), constrained_layout=True, sharex=True)

    ax = axes[0]
    ax.plot(df["year"], df["ai_ratio_weighted"], marker="o", linewidth=2.8, color="#1f4e79", label="AI ratio")
    ax.plot(df["year"], df["ai_risk_ratio_weighted"], marker="o", linewidth=2.8, color="#b22222", label="AI-risk ratio")
    ax.axvspan("2023", "2025", color="#f3f0e6", alpha=0.6)
    ax.annotate(
        "Acceleration after 2023",
        xy=("2024", float(df.loc[df["year"] == "2024", "ai_risk_ratio_weighted"].iloc[0])),
        xytext=("2019", float(df["ai_risk_ratio_weighted"].max()) * 0.8),
        arrowprops={"arrowstyle": "->", "color": "#444"},
        fontsize=11,
    )
    ax.set_title("Document-Level AI Disclosure Intensifies After 2023")
    ax.set_ylabel("Weighted ratio")
    ax.legend(frameon=False, loc="upper left")

    ax2 = axes[1]
    share = np.where(df["ai_ratio_weighted"] > 0, df["ai_risk_ratio_weighted"] / df["ai_ratio_weighted"], 0)
    ax2.plot(df["year"], share, marker="o", linewidth=3, color="#8b0000")
    ax2.fill_between(df["year"], share, alpha=0.15, color="#8b0000")
    ax2.axvspan("2023", "2025", color="#f3f0e6", alpha=0.6)
    ax2.set_title("Risk Framing Share of AI Mentions")
    ax2.set_ylabel("AI-risk / AI")
    ax2.set_xlabel("Year")
    fig.savefig(ACADEMIC / "01_trends_academic.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_academic_section_heatmap(section_year: pd.DataFrame) -> None:
    sns.set_theme(style="white", context="talk")
    df = add_section_labels(section_year)
    df = df[df["year"] != "2026"].copy()
    focus = ["ITEM_1", "ITEM_1A", "ITEM_1C", "ITEM_7", "ITEM_7A", "ITEM_8", "ITEM_9A", "ITEM_15"]
    df = df[df["section"].isin(focus)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
    for ax, metric, title, colors in [
        (axes[0], "ai_ratio_weighted", "AI Ratio by Filing Section", ["#f7fbff", "#6baed6", "#08306b"]),
        (axes[1], "ai_risk_ratio_weighted", "AI-Risk Ratio by Filing Section", ["#fff5f0", "#fb6a4a", "#67000d"]),
    ]:
        pivot = df.pivot(index="section_label", columns="year", values=metric).fillna(0)
        cmap = LinearSegmentedColormap.from_list(metric, colors)
        sns.heatmap(pivot, cmap=cmap, linewidths=0.5, linecolor="white", cbar=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("")
    fig.savefig(ACADEMIC / "02_section_heatmap_academic.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_academic_company(company_summary: pd.DataFrame, company_year: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    top = company_summary.sort_values("ai_risk_count", ascending=False).head(20).copy()
    top_ids = top["cik"].tolist()
    df = company_year[(company_year["cik"].isin(top_ids)) & (company_year["year"] != "2026")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

    scatter = top.copy()
    scatter["risk_share"] = np.where(scatter["ai_ratio"] > 0, scatter["ai_risk_ratio"] / scatter["ai_ratio"], 0)
    sns.scatterplot(
        data=scatter,
        x="latest_ai_ratio",
        y="risk_share",
        size="filings",
        sizes=(60, 350),
        color="#8b1e3f",
        alpha=0.8,
        ax=axes[0],
    )
    for _, row in scatter.head(10).iterrows():
        axes[0].text(row["latest_ai_ratio"], row["risk_share"], short_company_label(row), fontsize=9)
    axes[0].set_title("Top Companies by AI-Risk Volume")
    axes[0].set_xlabel("Latest AI ratio")
    axes[0].set_ylabel("Risk framing share")

    heat = df.pivot_table(index="cik", columns="year", values="ai_risk_ratio", aggfunc="mean").fillna(0)
    heat = heat.loc[heat.mean(axis=1).sort_values(ascending=False).index]
    label_map = (
        company_summary[["cik", "ticker", "title"]]
        .drop_duplicates(subset=["cik"])
        .set_index("cik")
        .apply(lambda x: x["ticker"] if str(x["ticker"]).strip() else (x["title"][:22] if str(x["title"]).strip() else x.name), axis=1)
        .to_dict()
    )
    heat.index = [label_map.get(idx, idx) for idx in heat.index]
    sns.heatmap(heat, cmap=LinearSegmentedColormap.from_list("heat", ["#fff5f0", "#fb6a4a", "#67000d"]), ax=axes[1])
    axes[1].set_title("AI-Risk Trajectories for Top Companies")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("")
    fig.savefig(ACADEMIC / "03_company_academic.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_academic_topics(ai_topics: pd.DataFrame, ai_risk_topics: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    ai = ai_topics.copy()
    air = ai_risk_topics.copy()
    ai["clean"] = ai["label"].map(clean_topic_label)
    air["clean"] = air["label"].map(clean_topic_label)
    ai = ai[~ai["clean"].str.contains("Xbrl", na=False)].head(5)
    air = air.head(5)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    sns.barplot(data=ai.sort_values("sentence_count"), x="sentence_count", y="clean", color="#3b82b6", ax=axes[0])
    sns.barplot(data=air.sort_values("sentence_count"), x="sentence_count", y="clean", color="#c23b22", ax=axes[1])
    axes[0].set_title("AI Mention Topics")
    axes[1].set_title("AI-Risk Mention Topics")
    for ax in axes:
        ax.set_xlabel("Sentences")
        ax.set_ylabel("")
    fig.savefig(ACADEMIC / "04_topics_academic.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_infographic_story(yearly: pd.DataFrame) -> None:
    sns.set_theme(style="white", context="poster")
    df = yearly[yearly["year"] != "2026"].copy()
    risk_share = np.where(df["ai_ratio_weighted"] > 0, df["ai_risk_ratio_weighted"] / df["ai_ratio_weighted"], 0)

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2])

    cards = [
        ("2015 AI ratio", f"{df.iloc[0]['ai_ratio_weighted']:.4f}", "#1f4e79"),
        ("2025 AI ratio", f"{df.iloc[-1]['ai_ratio_weighted']:.4f}", "#1f4e79"),
        ("2025 risk-framed AI", f"{risk_share[-1]:.1%}", "#8b0000"),
    ]
    for i, (title, value, color) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(color)
        ax.text(0.05, 0.72, title, color="white", fontsize=20, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.20, value, color="white", fontsize=34, fontweight="bold", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(df["year"], df["ai_ratio_weighted"], linewidth=4, color="#4ea8de", label="AI ratio")
    ax.plot(df["year"], df["ai_risk_ratio_weighted"], linewidth=4, color="#ef476f", label="AI-risk ratio")
    ax.fill_between(df["year"], df["ai_risk_ratio_weighted"], alpha=0.15, color="#ef476f")
    ax.axvline("2023", linestyle="--", linewidth=2.5, color="#444")
    ax.text(8.1, float(df["ai_risk_ratio_weighted"].max()) * 0.92, "2023 break", fontsize=16, color="#444")
    ax.set_title("AI Disclosures Shifted From Capability Talk to Risk Talk", fontsize=24, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Weighted ratio")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(INFO / "01_story_trends.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_infographic_sections(section_year: pd.DataFrame) -> None:
    sns.set_theme(style="white", context="talk")
    df = add_section_labels(section_year)
    df = df[(df["year"].isin(["2019", "2021", "2023", "2024", "2025"])) & (df["section"].isin(["ITEM_1", "ITEM_1A", "ITEM_1C", "ITEM_7", "ITEM_8", "ITEM_9A"]))].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, metric, title, cmap in [
        (axes[0], "ai_ratio_weighted", "Where AI Lives in the Filing", ["#eff6ff", "#60a5fa", "#1d4ed8"]),
        (axes[1], "ai_risk_ratio_weighted", "Where AI-Risk Lives in the Filing", ["#fff1f2", "#fb7185", "#be123c"]),
    ]:
        pivot = df.pivot(index="section_label", columns="year", values=metric).fillna(0)
        sns.heatmap(pivot, cmap=LinearSegmentedColormap.from_list(metric, cmap), annot=True, fmt=".3f", linewidths=0.5, linecolor="white", ax=ax, cbar=False)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.savefig(INFO / "02_story_sections.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_infographic_company(company_summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    df = company_summary.sort_values("ai_risk_count", ascending=False).head(15).copy()
    df["risk_share"] = np.where(df["ai_ratio"] > 0, df["ai_risk_ratio"] / df["ai_ratio"], 0)
    df["adopter_type"] = np.select(
        [
            df["first_ai_risk_year"].fillna("").astype(str).le("2022") & (df["first_ai_risk_year"].fillna("") != ""),
            df["ai_surge_2024_2025_vs_2021_2023"] > df["ai_surge_2024_2025_vs_2021_2023"].median(),
        ],
        ["Earlier risk adopters", "Late surge firms"],
        default="Steady movers",
    )

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    sns.scatterplot(
        data=df,
        x="latest_ai_ratio",
        y="risk_share",
        hue="adopter_type",
        size="ai_risk_count",
        sizes=(80, 500),
        palette={"Earlier risk adopters": "#577590", "Late surge firms": "#f94144", "Steady movers": "#90be6d"},
        alpha=0.9,
        ax=ax,
    )
    for _, row in df.iterrows():
        ax.text(row["latest_ai_ratio"], row["risk_share"], short_company_label(row), fontsize=9)
    ax.set_title("Companies Follow Different AI-Risk Paths", fontweight="bold")
    ax.set_xlabel("Latest AI intensity")
    ax.set_ylabel("Share of AI talk framed as risk")
    ax.legend(frameon=True, loc="best")
    fig.savefig(INFO / "03_story_companies.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_infographic_topics(ai_topics: pd.DataFrame, ai_risk_topics: pd.DataFrame) -> None:
    sns.set_theme(style="white", context="talk")
    ai = ai_topics.copy()
    air = ai_risk_topics.copy()
    ai["clean"] = ai["label"].map(clean_topic_label)
    air["clean"] = air["label"].map(clean_topic_label)
    ai = ai[~ai["clean"].str.contains("Xbrl", na=False)].head(4)
    air = air.head(4)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    for ax, df, title, color in [
        (axes[0], ai, "AI Mentions Cluster Around Capability, Analytics, and Regulation", "#3a86ff"),
        (axes[1], air, "AI-Risk Mentions Cluster Around Regulation, IP, Harm, and Attacks", "#ef476f"),
    ]:
        plot_df = df.sort_values("sentence_count", ascending=True)
        ax.barh(plot_df["clean"], plot_df["sentence_count"], color=color)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Sentences assigned to topic")
        ax.set_ylabel("")
        for i, v in enumerate(plot_df["sentence_count"]):
            ax.text(v, i, f" {int(v):,}", va="center", fontsize=11)
    fig.savefig(INFO / "04_story_topics.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_notes() -> None:
    text = """# Refined Visual Sets

## Academic Set

- `01_trends_academic.png`: cleaner time-series for a paper or thesis.
- `02_section_heatmap_academic.png`: section-level concentration with reader-friendly labels.
- `03_company_academic.png`: interpretable company comparison plus trajectory heatmap.
- `04_topics_academic.png`: cleaned unsupervised topic comparison.

## Infographic Set

- `01_story_trends.png`: key numbers plus the main shift over time.
- `02_story_sections.png`: annotated section heatmaps for presentation use.
- `03_story_companies.png`: company typology in more interpretable axes.
- `04_story_topics.png`: topic narrative in a presentation-friendly layout.
"""
    (PLOTS / "refined_sets.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    data = load_data()
    mapping = load_company_mapping()
    data = enrich_company_frames(data, mapping)
    plot_academic_trends(data["yearly"])
    plot_academic_section_heatmap(data["section_year"])
    plot_academic_company(data["company_summary"], data["company_year"])
    plot_academic_topics(data["ai_topics"], data["ai_risk_topics"])

    plot_infographic_story(data["yearly"])
    plot_infographic_sections(data["section_year"])
    plot_infographic_company(data["company_summary"])
    plot_infographic_topics(data["ai_topics"], data["ai_risk_topics"])
    write_notes()
    print(f"Refined visual sets written to {PLOTS}")


if __name__ == "__main__":
    main()
