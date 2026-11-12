from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "analysis" / "advanced"
RESEARCH = BASE / "research_extensions"
INDMARKET = RESEARCH / "industry_market"
OUT = BASE / "final_publication_pack"
PLOTS = OUT / "plots"

SECTION_MAP = {
    "ITEM_1": "Business",
    "ITEM_1A": "Risk Factors",
    "ITEM_1C": "Cybersecurity",
    "ITEM_7": "MD&A",
    "ITEM_7A": "Market Risk",
    "ITEM_8": "Financial Statements",
    "ITEM_9A": "Controls",
    "ITEM_15": "Exhibits",
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)


def load() -> dict:
    return {
        "yearly": pd.read_csv(BASE / "yearly_document_ratios.csv", dtype={"year": str}),
        "section_year": pd.read_csv(BASE / "yearly_section_ratios.csv", dtype={"year": str}),
        "company_summary": pd.read_csv(BASE / "company_behavior_summary_enriched.csv", dtype={"cik": str}),
        "maturity": pd.read_csv(RESEARCH / "maturity_ladder_company_year.csv", dtype={"cik": str, "year": str}),
        "migration": pd.read_csv(RESEARCH / "section_migration_yearly.csv", dtype={"year": str}),
        "topic_ai": pd.read_csv(RESEARCH / "topic_evolution_ai.csv", dtype={"year": str}),
        "topic_airisk": pd.read_csv(RESEARCH / "topic_evolution_ai_risk.csv", dtype={"year": str}),
        "industry_ff12": pd.read_csv(INDMARKET / "industry_ff12_yearly.csv", dtype={"year": str}),
        "size_buckets": pd.read_csv(INDMARKET / "size_bucket_yearly.csv", dtype={"year": str}),
        "leader_follow": pd.read_csv(RESEARCH / "leader_follower_similarity.csv", dtype={"cik": str, "year": str}),
        "contradictions": pd.read_csv(RESEARCH / "contradiction_profiles.csv", dtype={"cik": str, "year": str}),
        "risk_network": pd.read_csv(RESEARCH / "risk_comention_network.csv"),
        "framing": pd.read_csv(RESEARCH / "framing_yearly.csv", dtype={"year": str}),
    }


def setup() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def plot_01_regime_shift(yearly: pd.DataFrame) -> None:
    df = yearly[yearly["year"] != "2026"].copy()
    risk_share = np.where(df["ai_ratio_weighted"] > 0, df["ai_risk_ratio_weighted"] / df["ai_ratio_weighted"], 0)

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), constrained_layout=True, sharex=True)
    axes[0].plot(df["year"], df["ai_ratio_weighted"], marker="o", linewidth=3, color="#1d4ed8")
    axes[0].plot(df["year"], df["ai_risk_ratio_weighted"], marker="o", linewidth=3, color="#be123c")
    axes[0].axvspan("2023", "2025", color="#f3f0e6", alpha=0.7)
    axes[0].text(7.9, float(df["ai_ratio_weighted"].max()) * 0.92, "Post-2023 regime", fontsize=12, color="#444")
    axes[0].set_title("Figure 1. AI Disclosure Becomes Risk Disclosure After 2023")
    axes[0].set_ylabel("Weighted ratio")
    axes[0].text(df["year"].iloc[-1], df["ai_ratio_weighted"].iloc[-1], "  AI ratio", va="center", color="#1d4ed8")
    axes[0].text(df["year"].iloc[-1], df["ai_risk_ratio_weighted"].iloc[-1], "  AI-risk ratio", va="center", color="#be123c")

    axes[1].plot(df["year"], risk_share, marker="o", linewidth=3, color="#7f1d1d")
    axes[1].fill_between(df["year"], risk_share, alpha=0.15, color="#7f1d1d")
    axes[1].axvspan("2023", "2025", color="#f3f0e6", alpha=0.7)
    axes[1].set_title("Share of AI Mentions Framed as Risk")
    axes[1].set_ylabel("AI-risk / AI")
    axes[1].set_xlabel("Year")
    fig.savefig(PLOTS / "01_regime_shift.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_02_section_story(section_year: pd.DataFrame) -> None:
    df = section_year[section_year["year"] != "2026"].copy()
    focus = ["ITEM_1", "ITEM_1A", "ITEM_1C", "ITEM_7", "ITEM_8", "ITEM_9A"]
    df = df[df["section"].isin(focus)].copy()
    df["section_label"] = df["section"].map(SECTION_MAP)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for ax, metric, title, colors in [
        (axes[0], "ai_ratio_weighted", "AI mention intensity", ["#eff6ff", "#60a5fa", "#1d4ed8"]),
        (axes[1], "ai_risk_ratio_weighted", "AI-risk mention intensity", ["#fff1f2", "#fb7185", "#be123c"]),
    ]:
        pivot = df.pivot(index="section_label", columns="year", values=metric).fillna(0)
        sns.heatmap(pivot, cmap=LinearSegmentedColormap.from_list(metric, colors), linewidths=0.5, linecolor="white", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("")
    fig.suptitle("Figure 2. AI-Risk Migrates Toward Formal Risk and Cybersecurity Sections", fontsize=18, fontweight="bold")
    fig.savefig(PLOTS / "02_section_story.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_03_maturity(maturity: pd.DataFrame) -> None:
    order = ["0_No_AI", "1_Capability", "2_Operational_Use", "3_Governance_Cyber", "4_Formal_Risk"]
    share = maturity.groupby(["year", "maturity_stage"]).size().reset_index(name="count")
    share["share"] = share["count"] / share.groupby("year")["count"].transform("sum")
    pivot = share.pivot(index="year", columns="maturity_stage", values="share").fillna(0)
    pivot = pivot[[c for c in order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    pivot.plot(kind="area", stacked=True, ax=ax, colormap="RdYlBu_r")
    ax.set_title("Figure 3. Firms Move Up an AI Disclosure Maturity Ladder")
    ax.set_ylabel("Share of company-years")
    ax.set_xlabel("Year")
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(PLOTS / "03_maturity_ladder.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_04_industry_heatmap(industry_ff12: pd.DataFrame) -> None:
    top = industry_ff12.groupby("ff12_name")["avg_ai_risk_ratio"].mean().sort_values(ascending=False).head(10).index.tolist()
    df = industry_ff12[industry_ff12["ff12_name"].isin(top)].copy()
    pivot = df.pivot(index="ff12_name", columns="year", values="avg_ai_risk_ratio").fillna(0)
    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    sns.heatmap(pivot, cmap=LinearSegmentedColormap.from_list("ind", ["#fff5f0", "#fb6a4a", "#67000d"]), linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Figure 4. AI-Risk Intensity Diffuses Unevenly Across Industries")
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    fig.savefig(PLOTS / "04_industry_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_05_size_effects(size_buckets: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sns.lineplot(data=size_buckets, x="year", y="avg_ai_ratio", hue="size_bucket", marker="o", linewidth=2.8, ax=axes[0])
    axes[0].set_title("AI ratio by market-cap bucket")
    axes[0].set_ylabel("Average AI ratio")
    axes[0].set_xlabel("Year")
    axes[0].legend(title="", frameon=False)

    sns.lineplot(data=size_buckets, x="year", y="avg_ai_risk_ratio", hue="size_bucket", marker="o", linewidth=2.8, ax=axes[1])
    axes[1].set_title("AI-risk ratio by market-cap bucket")
    axes[1].set_ylabel("Average AI-risk ratio")
    axes[1].set_xlabel("Year")
    axes[1].legend(title="", frameon=False)
    fig.suptitle("Figure 5. Larger Firms Lead the Disclosure Shift", fontsize=18, fontweight="bold")
    fig.savefig(PLOTS / "05_size_effects.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_06_company_typology(company_summary: pd.DataFrame) -> None:
    df = company_summary.sort_values("ai_risk_count", ascending=False).head(18).copy()
    df = df.reset_index(drop=True)
    df["risk_share"] = np.where(df["ai_ratio"] > 0, df["ai_risk_ratio"] / df["ai_ratio"], 0)
    df["adopter_type"] = np.select(
        [
            pd.to_numeric(df["first_ai_risk_year"], errors="coerce").le(2022),
            df["ai_surge_2024_2025_vs_2021_2023"] > df["ai_surge_2024_2025_vs_2021_2023"].median(),
        ],
        ["Earlier risk adopters", "Late surge firms"],
        default="Steady movers",
    )
    label = np.where(df["ticker"].notna() & df["ticker"].ne(""), df["ticker"], df["cik"])
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    sns.scatterplot(
        data=df,
        x="latest_ai_ratio",
        y="risk_share",
        hue="adopter_type",
        size="ai_risk_count",
        sizes=(70, 450),
        palette={"Earlier risk adopters": "#577590", "Late surge firms": "#f94144", "Steady movers": "#90be6d"},
        alpha=0.9,
        ax=ax,
    )
    for i, row in df.iterrows():
        ax.text(row["latest_ai_ratio"], row["risk_share"], str(label[i]), fontsize=9)
    ax.set_title("Figure 6. Companies Follow Different AI-Risk Paths")
    ax.set_xlabel("Latest AI intensity")
    ax.set_ylabel("Share of AI talk framed as risk")
    ax.legend(frameon=True, loc="best")
    fig.savefig(PLOTS / "06_company_typology.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_07_topic_evolution(topic_ai: pd.DataFrame, topic_airisk: pd.DataFrame) -> None:
    ai_top = topic_ai.groupby("topic_label")["count"].sum().sort_values(ascending=False).head(5).index.tolist()
    air_top = topic_airisk.groupby("topic_label")["count"].sum().sort_values(ascending=False).head(5).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, df, labels, title, colors in [
        (axes[0], topic_ai[topic_ai["topic_label"].isin(ai_top)], ai_top, "AI topic evolution", ["#eff6ff", "#60a5fa", "#1d4ed8"]),
        (axes[1], topic_airisk[topic_airisk["topic_label"].isin(air_top)], air_top, "AI-risk topic evolution", ["#fff1f2", "#fb7185", "#be123c"]),
    ]:
        pivot = df.pivot(index="topic_label", columns="year", values="count").fillna(0)
        sns.heatmap(pivot, cmap=LinearSegmentedColormap.from_list(title, colors), ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("")
    fig.suptitle("Figure 7. Topics Shift from Capability and Analytics to Governance, Harm, and Regulation", fontsize=18, fontweight="bold")
    fig.savefig(PLOTS / "07_topic_evolution.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_08_risk_ecosystem(risk_network: pd.DataFrame) -> None:
    top = risk_network.sort_values("weight", ascending=False).head(10).copy()
    top["pair"] = top["source"] + " x " + top["target"]
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    sns.barplot(data=top.sort_values("weight"), x="weight", y="pair", color="#ef476f", ax=ax)
    ax.set_title("Figure 8. The AI-Risk Ecosystem Centers on Regulation, Privacy/IP, and Operations")
    ax.set_xlabel("Co-mention weight")
    ax.set_ylabel("")
    fig.savefig(PLOTS / "08_risk_ecosystem.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_09_leader_follower(leader_follow: pd.DataFrame) -> None:
    if leader_follow.empty:
        return
    yearly = leader_follow.groupby("year")["max_similarity_to_leader"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(yearly["year"], yearly["max_similarity_to_leader"], marker="o", linewidth=3, color="#6a4c93")
    ax.fill_between(yearly["year"], yearly["max_similarity_to_leader"], alpha=0.15, color="#6a4c93")
    ax.set_title("Figure 9. AI-Risk Language Increasingly Resembles Large-Platform Leaders")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average max similarity to leader")
    fig.savefig(PLOTS / "09_leader_follower.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_10_framing(framing: pd.DataFrame) -> None:
    df = framing[framing["dataset"] == "ai_risk"].copy()
    top = df.groupby("frame")["count"].sum().sort_values(ascending=False).head(5).index.tolist()
    df = df[df["frame"].isin(top)]
    pivot = df.pivot(index="frame", columns="year", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.heatmap(pivot, cmap=LinearSegmentedColormap.from_list("frame", ["#fff5f0", "#fb6a4a", "#67000d"]), ax=ax)
    ax.set_title("Figure 10. AI-Risk Framing Moves Toward Governance and Harm")
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    fig.savefig(PLOTS / "10_framing_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_guide() -> None:
    text = """# Final Publication Pack

## Recommended Main-Text Figures

1. `01_regime_shift.png`
Use for the core time-series argument: AI disclosure becomes AI-risk disclosure after 2023.

2. `02_section_story.png`
Use to show where inside filings this shift happens, especially Risk Factors and Cybersecurity.

3. `03_maturity_ladder.png`
Use to frame the disclosure process as a staged transition rather than a one-time jump.

4. `04_industry_heatmap.png`
Use to show uneven diffusion across industries.

5. `05_size_effects.png`
Use to show that larger firms lead the disclosure shift.

6. `06_company_typology.png`
Use for firm heterogeneity and named examples.

## Strong Appendix / Secondary Figures

7. `07_topic_evolution.png`
Use to support the narrative that context changes over time.

8. `08_risk_ecosystem.png`
Use to summarize which risks cluster around AI-risk language.

9. `09_leader_follower.png`
Use if you want a diffusion / imitation argument.

10. `10_framing_heatmap.png`
Use if you want a communication/framing angle.
"""
    (OUT / "guide.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    setup()
    data = load()
    plot_01_regime_shift(data["yearly"])
    plot_02_section_story(data["section_year"])
    plot_03_maturity(data["maturity"])
    plot_04_industry_heatmap(data["industry_ff12"])
    plot_05_size_effects(data["size_buckets"])
    plot_06_company_typology(data["company_summary"])
    plot_07_topic_evolution(data["topic_ai"], data["topic_airisk"])
    plot_08_risk_ecosystem(data["risk_network"])
    plot_09_leader_follower(data["leader_follow"])
    plot_10_framing(data["framing"])
    write_guide()
    print(f"Final publication pack written to {OUT}")


if __name__ == "__main__":
    main()
