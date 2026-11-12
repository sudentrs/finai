from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "analysis" / "advanced"
OUT = BASE / "research_extensions"
PLOTS = OUT / "plots"

SECTION_LABELS = {
    "ITEM_1": "Business",
    "ITEM_1A": "Risk Factors",
    "ITEM_1C": "Cybersecurity",
    "ITEM_7": "MD&A",
    "ITEM_7A": "Market Risk",
    "ITEM_8": "Financial Statements",
    "ITEM_9A": "Controls",
    "ITEM_15": "Exhibits",
}

RISK_THEME_PATTERNS = {
    "cyber_fraud": re.compile(r"\b(cyber|attack|attacks|fraud|phishing|malicious|ransomware|credential|bad actors)\b", re.I),
    "privacy_ip": re.compile(r"\b(privacy|confidential|proprietary|intellectual property|copyright|trade secret|rights|infringement)\b", re.I),
    "regulation_compliance": re.compile(r"\b(regulation|regulations|regulatory|compliance|law|laws|act|governance|policy|policies)\b", re.I),
    "model_reliability": re.compile(r"\b(incorrect|inaccurate|erroneous|hallucinat|reliable|reliability|monitoring|explain|transparency|bias|biased|fairness)\b", re.I),
    "operations_reputation": re.compile(r"\b(reputation|reputational|operations|operational|adversely affect|financial condition|results of operations|harm|liability)\b", re.I),
}

FRAME_PATTERNS = {
    "opportunity": re.compile(r"\b(opportunity|growth|competitive advantage|efficiency|improve|innovation|enhance|accelerate|enable)\b", re.I),
    "product_capability": re.compile(r"\b(platform|solution|product|service|tool|feature|analytics|automation|software|application)\b", re.I),
    "governance_regulatory": re.compile(r"\b(regulation|regulatory|compliance|law|laws|act|governance|policy|policies|consumer protection)\b", re.I),
    "harm_uncertainty": re.compile(r"\b(risk|uncertain|uncertainty|harm|adversely affect|challenge|liability|misuse|incorrect|bias|breach)\b", re.I),
    "cybersecurity": re.compile(r"\b(cyber|attack|fraud|phishing|ransomware|security breach|bad actors)\b", re.I),
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)


def load() -> dict:
    company_year = pd.read_csv(BASE / "company_year_metrics_enriched.csv", dtype={"cik": str, "year": str})
    company_summary = pd.read_csv(BASE / "company_behavior_summary_enriched.csv", dtype={"cik": str})
    section_metrics = pd.read_csv(BASE / "section_metrics.csv", dtype={"cik": str, "year": str})
    ai_sentences = pd.read_csv(BASE / "ai_sentences.csv", dtype={"cik": str, "year": str})
    ai_risk_sentences = pd.read_csv(BASE / "ai_risk_sentences.csv", dtype={"cik": str, "year": str})
    ai_topics = pd.read_csv(BASE / "ai_topics.csv")
    ai_risk_topics = pd.read_csv(BASE / "ai_risk_topics.csv")
    ai_topic_assignments = pd.read_csv(BASE / "ai_topic_assignments.csv", dtype={"cik": str, "year": str})
    ai_risk_topic_assignments = pd.read_csv(BASE / "ai_risk_topic_assignments.csv", dtype={"cik": str, "year": str})
    return {
        "company_year": company_year[company_year["year"] != "2026"].copy(),
        "company_summary": company_summary.copy(),
        "section_metrics": section_metrics[section_metrics["year"] != "2026"].copy(),
        "ai_sentences": ai_sentences[ai_sentences["year"] != "2026"].copy(),
        "ai_risk_sentences": ai_risk_sentences[ai_risk_sentences["year"] != "2026"].copy(),
        "ai_topics": ai_topics.copy(),
        "ai_risk_topics": ai_risk_topics.copy(),
        "ai_topic_assignments": ai_topic_assignments[ai_topic_assignments["year"] != "2026"].copy(),
        "ai_risk_topic_assignments": ai_risk_topic_assignments[ai_risk_topic_assignments["year"] != "2026"].copy(),
    }


def build_maturity(company_year: pd.DataFrame, section_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    section_slice = section_metrics[section_metrics["section"].isin(["ITEM_1", "ITEM_1A", "ITEM_1C", "ITEM_7"])].copy()
    sec_agg = (
        section_slice.groupby(["cik", "year", "section"], as_index=False)[["section_ai_count", "section_ai_risk_count"]].sum()
    )
    pivot_ai = sec_agg.pivot_table(index=["cik", "year"], columns="section", values="section_ai_count", aggfunc="sum", fill_value=0)
    pivot_airisk = sec_agg.pivot_table(index=["cik", "year"], columns="section", values="section_ai_risk_count", aggfunc="sum", fill_value=0)
    for col in ["ITEM_1", "ITEM_1A", "ITEM_1C", "ITEM_7"]:
        if col not in pivot_ai.columns:
            pivot_ai[col] = 0
        if col not in pivot_airisk.columns:
            pivot_airisk[col] = 0

    df = company_year.merge(pivot_ai.reset_index(), on=["cik", "year"], how="left", suffixes=("", "_ai"))
    df = df.merge(pivot_airisk.reset_index(), on=["cik", "year"], how="left", suffixes=("", "_airisk"))
    df = df.fillna(0)

    def classify(row: pd.Series) -> str:
        if row["ai_count"] <= 0:
            return "0_No_AI"
        if row["ai_risk_count"] <= 0 and row["ITEM_1"] > 0:
            return "1_Capability"
        if row["ai_risk_count"] <= 0 and row["ITEM_7"] > 0:
            return "2_Operational_Use"
        if row["ITEM_1A_airisk"] > 0 or row["ITEM_1C_airisk"] > 0:
            if row["ITEM_1A_airisk"] > 0:
                return "4_Formal_Risk"
            return "3_Governance_Cyber"
        return "2_Operational_Use"

    # Rename merged columns for readability.
    rename_map = {
        "ITEM_1_airisk": "ITEM_1_airisk",
        "ITEM_1A_airisk": "ITEM_1A_airisk",
        "ITEM_1C_airisk": "ITEM_1C_airisk",
        "ITEM_7_airisk": "ITEM_7_airisk",
    }
    for col in ["ITEM_1_airisk", "ITEM_1A_airisk", "ITEM_1C_airisk", "ITEM_7_airisk"]:
        if col not in df.columns:
            df[col] = 0
    df["maturity_stage"] = df.apply(classify, axis=1)
    maturity = df[
        [
            "cik",
            "year",
            "ticker",
            "title",
            "display_name",
            "ai_count",
            "ai_risk_count",
            "ai_ratio",
            "ai_risk_ratio",
            "risk_framing_ratio",
            "maturity_stage",
        ]
    ].copy()
    maturity.to_csv(OUT / "maturity_ladder_company_year.csv", index=False)

    transitions = maturity.sort_values(["cik", "year"]).copy()
    transitions["next_stage"] = transitions.groupby("cik")["maturity_stage"].shift(-1)
    transitions["next_year"] = transitions.groupby("cik")["year"].shift(-1)
    transitions = transitions.dropna(subset=["next_stage"])
    transition_counts = (
        transitions.groupby(["maturity_stage", "next_stage"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )
    transition_counts.to_csv(OUT / "maturity_transitions.csv", index=False)
    return maturity, transition_counts


def adoption_timing(company_summary: pd.DataFrame) -> pd.DataFrame:
    df = company_summary.copy()
    df["first_ai_year"] = pd.to_numeric(df["first_ai_year"], errors="coerce")
    df["first_ai_risk_year"] = pd.to_numeric(df["first_ai_risk_year"], errors="coerce")
    df["lag_ai_to_ai_risk"] = df["first_ai_risk_year"] - df["first_ai_year"]
    df["adoption_pattern"] = np.select(
        [
            df["first_ai_year"].isna(),
            df["lag_ai_to_ai_risk"].fillna(999) == 0,
            df["lag_ai_to_ai_risk"].between(1, 2, inclusive="both"),
            df["lag_ai_to_ai_risk"] >= 3,
        ],
        [
            "No AI adoption",
            "Immediate risk framing",
            "Short lag to risk framing",
            "Long lag to risk framing",
        ],
        default="AI without explicit risk framing",
    )
    df.to_csv(OUT / "adoption_timing_patterns.csv", index=False)
    return df


def section_migration(section_metrics: pd.DataFrame) -> pd.DataFrame:
    focus = section_metrics[section_metrics["section"].isin(["ITEM_1", "ITEM_1A", "ITEM_1C", "ITEM_7"])].copy()
    agg = focus.groupby(["cik", "year", "section"], as_index=False)[["section_ai_count", "section_ai_risk_count"]].sum()
    ai_home = agg.loc[agg.groupby(["cik", "year"])["section_ai_count"].idxmax()].copy()
    ai_home = ai_home[ai_home["section_ai_count"] > 0][["cik", "year", "section"]].rename(columns={"section": "dominant_ai_section"})
    airisk_home = agg.loc[agg.groupby(["cik", "year"])["section_ai_risk_count"].idxmax()].copy()
    airisk_home = airisk_home[airisk_home["section_ai_risk_count"] > 0][["cik", "year", "section"]].rename(columns={"section": "dominant_ai_risk_section"})

    mig = pd.merge(ai_home, airisk_home, on=["cik", "year"], how="outer")
    mig["dominant_ai_section"] = mig["dominant_ai_section"].fillna("No AI")
    mig["dominant_ai_risk_section"] = mig["dominant_ai_risk_section"].fillna("No AI-Risk")
    yearly = mig.groupby(["year", "dominant_ai_section", "dominant_ai_risk_section"]).size().reset_index(name="count")
    yearly.to_csv(OUT / "section_migration_yearly.csv", index=False)
    return yearly


def breadth_analysis(section_metrics: pd.DataFrame, company_year: pd.DataFrame) -> pd.DataFrame:
    sec = section_metrics.copy()
    sec["ai_flag"] = sec["section_ai_count"] > 0
    sec["ai_risk_flag"] = sec["section_ai_risk_count"] > 0
    breadth = (
        sec.groupby(["cik", "year"], as_index=False)
        .agg(
            ai_section_breadth=("ai_flag", "sum"),
            ai_risk_section_breadth=("ai_risk_flag", "sum"),
            sections_observed=("section", "nunique"),
        )
    )
    out = company_year.merge(breadth, on=["cik", "year"], how="left").fillna(0)
    out["ai_breadth_ratio"] = np.where(out["sections_observed"] > 0, out["ai_section_breadth"] / out["sections_observed"], 0)
    out["ai_risk_breadth_ratio"] = np.where(out["sections_observed"] > 0, out["ai_risk_section_breadth"] / out["sections_observed"], 0)
    out.to_csv(OUT / "disclosure_breadth_company_year.csv", index=False)
    return out


def aggregate_text(sentence_df: pd.DataFrame, min_sentences: int = 3) -> pd.DataFrame:
    grouped = sentence_df.groupby(["cik", "year"], as_index=False).agg(
        sentence_count=("sentence", "size"),
        text=("sentence", lambda s: " ".join(map(str, s))),
    )
    return grouped[grouped["sentence_count"] >= min_sentences].reset_index(drop=True)


def convergence_and_boilerplate(ai_risk_sentences: pd.DataFrame, company_year: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = aggregate_text(ai_risk_sentences, min_sentences=3)
    if grouped.empty:
        empty = pd.DataFrame()
        empty.to_csv(OUT / "convergence_yearly.csv", index=False)
        empty.to_csv(OUT / "boilerplate_company_year.csv", index=False)
        return empty, empty

    vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.8, ngram_range=(1, 2), max_features=4000)
    X = vectorizer.fit_transform(grouped["text"])
    sims = cosine_similarity(X)

    boiler = []
    for i, row in grouped.iterrows():
        sim_row = sims[i].copy()
        sim_row[i] = 0.0
        same_year_idx = grouped.index[grouped["year"] == row["year"]].tolist()
        same_year_scores = [sim_row[j] for j in same_year_idx if j != i]
        boiler.append(
            {
                "cik": row["cik"],
                "year": row["year"],
                "sentence_count": row["sentence_count"],
                "avg_similarity_same_year": float(np.mean(same_year_scores)) if same_year_scores else 0.0,
                "max_similarity_same_year": float(np.max(same_year_scores)) if same_year_scores else 0.0,
                "novelty_score": 1.0 - (float(np.max(same_year_scores)) if same_year_scores else 0.0),
            }
        )
    boiler_df = pd.DataFrame(boiler).merge(company_year[["cik", "year", "ticker", "title", "display_name"]], on=["cik", "year"], how="left")
    boiler_df.to_csv(OUT / "boilerplate_company_year.csv", index=False)

    yearly = []
    for year, sub in grouped.groupby("year"):
        idx = sub.index.to_list()
        if len(idx) < 2:
            continue
        mat = sims[np.ix_(idx, idx)]
        mask = ~np.eye(len(idx), dtype=bool)
        yearly.append(
            {
                "year": year,
                "company_years": len(idx),
                "avg_pairwise_similarity": float(mat[mask].mean()) if mask.sum() else 0.0,
                "median_pairwise_similarity": float(np.median(mat[mask])) if mask.sum() else 0.0,
            }
        )
    yearly_df = pd.DataFrame(yearly)
    yearly_df.to_csv(OUT / "convergence_yearly.csv", index=False)
    return yearly_df, boiler_df


def leader_follower(ai_risk_sentences: pd.DataFrame, company_year: pd.DataFrame) -> pd.DataFrame:
    grouped = aggregate_text(ai_risk_sentences, min_sentences=3)
    if grouped.empty:
        out = pd.DataFrame()
        out.to_csv(OUT / "leader_follower_similarity.csv", index=False)
        return out

    merged = grouped.merge(company_year[["cik", "year", "ticker"]], on=["cik", "year"], how="left")
    leaders = merged[merged["ticker"].isin(["MSFT", "NVDA", "META", "ADBE", "CRM", "ACN", "GOOGL", "AMZN"])].copy()
    if leaders.empty:
        out = pd.DataFrame()
        out.to_csv(OUT / "leader_follower_similarity.csv", index=False)
        return out

    year_groups = dict(tuple(merged.groupby("year")))
    rows = []
    for year, sub in year_groups.items():
        leader_sub = leaders[leaders["year"] == year]
        if leader_sub.empty or len(sub) < 2:
            continue
        texts = pd.concat([leader_sub[["cik", "text"]], sub[["cik", "text"]]], ignore_index=True)
        vect = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.8, ngram_range=(1, 2), max_features=3000)
        X = vect.fit_transform(texts["text"])
        leader_X = X[: len(leader_sub)]
        firm_X = X[len(leader_sub) :]
        sim = cosine_similarity(firm_X, leader_X)
        firm_ids = sub["cik"].tolist()
        leader_ids = leader_sub["ticker"].fillna(leader_sub["cik"]).tolist()
        for i, firm in enumerate(firm_ids):
            rows.append(
                {
                    "year": year,
                    "cik": firm,
                    "max_similarity_to_leader": float(sim[i].max()),
                    "closest_leader": leader_ids[int(sim[i].argmax())],
                }
            )
    out = pd.DataFrame(rows).merge(company_year[["cik", "year", "ticker", "title", "display_name"]], on=["cik", "year"], how="left")
    out.to_csv(OUT / "leader_follower_similarity.csv", index=False)
    return out


def risk_comentions(ai_risk_sentences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in ai_risk_sentences.iterrows():
        sentence = str(row["sentence"])
        hit_labels = [label for label, pattern in RISK_THEME_PATTERNS.items() if pattern.search(sentence)]
        if not hit_labels:
            continue
        for label in hit_labels:
            rows.append({"year": row["year"], "theme": label})
    counts = pd.DataFrame(rows).groupby(["year", "theme"]).size().reset_index(name="count")
    counts.to_csv(OUT / "risk_comention_year_theme.csv", index=False)

    pair_rows = []
    for _, row in ai_risk_sentences.iterrows():
        sentence = str(row["sentence"])
        hit_labels = sorted([label for label, pattern in RISK_THEME_PATTERNS.items() if pattern.search(sentence)])
        for i in range(len(hit_labels)):
            for j in range(i + 1, len(hit_labels)):
                pair_rows.append({"source": hit_labels[i], "target": hit_labels[j]})
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pair_counts = pairs.groupby(["source", "target"]).size().reset_index(name="weight").sort_values("weight", ascending=False)
    else:
        pair_counts = pd.DataFrame(columns=["source", "target", "weight"])
    pair_counts.to_csv(OUT / "risk_comention_network.csv", index=False)
    return counts


def framing_analysis(ai_sentences: pd.DataFrame, ai_risk_sentences: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def classify_sentence(sentence: str) -> list[str]:
        return [label for label, pattern in FRAME_PATTERNS.items() if pattern.search(sentence)]

    rows = []
    for dataset_name, df in [("ai", ai_sentences), ("ai_risk", ai_risk_sentences)]:
        for _, row in df.iterrows():
            sentence = str(row["sentence"])
            frames = classify_sentence(sentence)
            if not frames:
                continue
            for frame in frames:
                rows.append({"dataset": dataset_name, "year": row["year"], "cik": row["cik"], "frame": frame})
    frame_df = pd.DataFrame(rows)
    year_frame = frame_df.groupby(["dataset", "year", "frame"]).size().reset_index(name="count")
    year_frame.to_csv(OUT / "framing_yearly.csv", index=False)

    company_frame = frame_df.groupby(["dataset", "cik", "year", "frame"]).size().reset_index(name="count")
    company_frame.to_csv(OUT / "framing_company_year.csv", index=False)
    return year_frame, company_frame


def topic_evolution(ai_assign: pd.DataFrame, ai_risk_assign: pd.DataFrame, ai_topics: pd.DataFrame, ai_risk_topics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ai_label_map = dict(zip(ai_topics["topic_id"], ai_topics["label"]))
    air_label_map = dict(zip(ai_risk_topics["topic_id"], ai_risk_topics["label"]))
    ai = ai_assign.copy()
    ai["topic_label"] = ai["topic_id"].map(ai_label_map)
    air = ai_risk_assign.copy()
    air["topic_label"] = air["topic_id"].map(air_label_map)
    ai_year = ai.groupby(["year", "topic_label"]).size().reset_index(name="count")
    air_year = air.groupby(["year", "topic_label"]).size().reset_index(name="count")
    ai_year.to_csv(OUT / "topic_evolution_ai.csv", index=False)
    air_year.to_csv(OUT / "topic_evolution_ai_risk.csv", index=False)

    section_topic = air.groupby(["section", "topic_label"]).size().reset_index(name="count").sort_values("count", ascending=False)
    section_topic.to_csv(OUT / "section_topic_matrix_ai_risk.csv", index=False)
    return ai_year, air_year


def disclosure_aggressiveness(company_year: pd.DataFrame, breadth: pd.DataFrame, boiler: pd.DataFrame) -> pd.DataFrame:
    df = company_year.merge(
        breadth[["cik", "year", "ai_section_breadth", "ai_risk_section_breadth", "ai_breadth_ratio", "ai_risk_breadth_ratio"]],
        on=["cik", "year"],
        how="left",
    )
    df = df.merge(boiler[["cik", "year", "novelty_score"]], on=["cik", "year"], how="left")
    for col in ["ai_ratio", "ai_risk_ratio", "risk_framing_ratio", "ai_breadth_ratio", "ai_risk_breadth_ratio", "novelty_score"]:
        df[col] = df[col].fillna(0.0)
        std = df[col].std()
        df[f"z_{col}"] = 0.0 if std == 0 or math.isnan(std) else (df[col] - df[col].mean()) / std
    df["aggressiveness_index"] = (
        0.25 * df["z_ai_ratio"]
        + 0.20 * df["z_ai_risk_ratio"]
        + 0.15 * df["z_risk_framing_ratio"]
        + 0.15 * df["z_ai_breadth_ratio"]
        + 0.15 * df["z_ai_risk_breadth_ratio"]
        + 0.10 * df["z_novelty_score"]
    )
    df.to_csv(OUT / "disclosure_aggressiveness_company_year.csv", index=False)
    return df


def contradiction_analysis(company_year: pd.DataFrame) -> pd.DataFrame:
    df = company_year.copy()
    ai_hi = df["ai_ratio"].quantile(0.9)
    airisk_lo = df["risk_framing_ratio"].quantile(0.25)
    airisk_hi = df["risk_framing_ratio"].quantile(0.9)
    ai_lo = df["ai_ratio"].quantile(0.25)
    df["contradiction_type"] = np.select(
        [
            (df["ai_ratio"] >= ai_hi) & (df["risk_framing_ratio"] <= airisk_lo),
            (df["ai_ratio"] <= ai_lo) & (df["risk_framing_ratio"] >= airisk_hi) & (df["ai_count"] > 0),
        ],
        [
            "Promotional AI, low risk framing",
            "Low AI volume, high risk framing",
        ],
        default="None",
    )
    out = df[df["contradiction_type"] != "None"].copy()
    out.to_csv(OUT / "contradiction_profiles.csv", index=False)
    return out


def event_window(company_year: pd.DataFrame) -> pd.DataFrame:
    df = company_year.copy()
    df["window"] = np.where(df["year"].isin(["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]), "Pre-2023", "2023-2025")
    out = (
        df.groupby("window")
        .agg(
            firm_years=("cik", "size"),
            avg_ai_ratio=("ai_ratio", "mean"),
            avg_ai_risk_ratio=("ai_risk_ratio", "mean"),
            avg_risk_framing=("risk_framing_ratio", "mean"),
            avg_ai_count=("ai_count", "mean"),
        )
        .reset_index()
    )
    out.to_csv(OUT / "event_window_summary.csv", index=False)
    return out


def plot_maturity(maturity: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    order = ["0_No_AI", "1_Capability", "2_Operational_Use", "3_Governance_Cyber", "4_Formal_Risk"]
    share = maturity.groupby(["year", "maturity_stage"]).size().reset_index(name="count")
    total = share.groupby("year")["count"].transform("sum")
    share["share"] = share["count"] / total
    pivot = share.pivot(index="year", columns="maturity_stage", values="share").fillna(0)
    pivot = pivot[[c for c in order if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    pivot.plot(kind="area", stacked=True, ax=ax, colormap="RdYlBu_r")
    ax.set_title("AI Disclosure Maturity Ladder Over Time")
    ax.set_ylabel("Share of company-years")
    ax.set_xlabel("Year")
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(PLOTS / "maturity_ladder_area.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_migration(section_mig: pd.DataFrame) -> None:
    focus = section_mig[section_mig["year"].isin(["2019", "2023", "2025"])].copy()
    if focus.empty:
        return
    focus["pair"] = focus["dominant_ai_section"] + " -> " + focus["dominant_ai_risk_section"]
    top_pairs = focus.groupby("pair")["count"].sum().sort_values(ascending=False).head(10).index.tolist()
    focus = focus[focus["pair"].isin(top_pairs)]
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    sns.barplot(data=focus, x="pair", y="count", hue="year", ax=ax)
    ax.set_title("How Dominant AI Sections Map into Dominant AI-Risk Sections")
    ax.set_xlabel("")
    ax.set_ylabel("Company-years")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(PLOTS / "section_migration_pairs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_breadth(breadth: pd.DataFrame) -> None:
    yearly = breadth.groupby("year")[["ai_section_breadth", "ai_risk_section_breadth"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(yearly["year"], yearly["ai_section_breadth"], marker="o", linewidth=2.8, label="AI breadth", color="#1f77b4")
    ax.plot(yearly["year"], yearly["ai_risk_section_breadth"], marker="o", linewidth=2.8, label="AI-risk breadth", color="#d62728")
    ax.set_title("Disclosure Breadth Across Filing Sections")
    ax.set_ylabel("Average number of sections")
    ax.set_xlabel("Year")
    ax.legend(frameon=False)
    fig.savefig(PLOTS / "breadth_trends.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(yearly_conv: pd.DataFrame, boiler: pd.DataFrame) -> None:
    if yearly_conv.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    axes[0].plot(yearly_conv["year"], yearly_conv["avg_pairwise_similarity"], marker="o", linewidth=2.8, color="#8c564b")
    axes[0].set_title("Convergence of AI-Risk Disclosure Language")
    axes[0].set_ylabel("Average pairwise similarity")
    axes[0].set_xlabel("Year")

    top = boiler.sort_values("novelty_score", ascending=False).head(15).copy()
    top["label"] = top["ticker"].fillna(top["cik"])
    sns.barplot(data=top.sort_values("novelty_score"), x="novelty_score", y="label", color="#2ca02c", ax=axes[1])
    axes[1].set_title("Most Novel AI-Risk Disclosures")
    axes[1].set_xlabel("Novelty score")
    axes[1].set_ylabel("")
    fig.savefig(PLOTS / "convergence_novelty.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_topics(ai_year: pd.DataFrame, air_year: pd.DataFrame) -> None:
    def top_labels(df: pd.DataFrame) -> list[str]:
        return df.groupby("topic_label")["count"].sum().sort_values(ascending=False).head(5).index.tolist()

    ai_top = top_labels(ai_year)
    air_top = top_labels(air_year)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, df, labels, title, colors in [
        (axes[0], ai_year[ai_year["topic_label"].isin(ai_top)], ai_top, "AI Topic Evolution", ["#eff6ff", "#60a5fa", "#1d4ed8"]),
        (axes[1], air_year[air_year["topic_label"].isin(air_top)], air_top, "AI-Risk Topic Evolution", ["#fff1f2", "#fb7185", "#be123c"]),
    ]:
        pivot = df.pivot(index="topic_label", columns="year", values="count").fillna(0)
        sns.heatmap(pivot, cmap=LinearSegmentedColormap.from_list(title, colors), ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("")
    fig.savefig(PLOTS / "topic_evolution_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_risk_network() -> None:
    pair_path = OUT / "risk_comention_network.csv"
    if not pair_path.exists():
        return
    pairs = pd.read_csv(pair_path)
    if pairs.empty:
        return
    G = nx.Graph()
    for _, row in pairs.head(10).iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])
    pos = nx.spring_layout(G, seed=42, weight="weight")
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=[1 + w / max(weights) * 6 for w in weights], alpha=0.35, edge_color="#999999", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color="#ef476f", node_size=2200, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=10, ax=ax)
    ax.set_title("AI-Risk Co-Mention Ecosystem")
    ax.axis("off")
    fig.savefig(PLOTS / "risk_comention_network.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_contradictions(contradictions: pd.DataFrame) -> None:
    if contradictions.empty:
        return
    top = contradictions.copy().sort_values(["contradiction_type", "ai_ratio"], ascending=[True, False]).head(20)
    top["label"] = top["ticker"].fillna(top["cik"])
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    sns.scatterplot(
        data=top,
        x="ai_ratio",
        y="risk_framing_ratio",
        hue="contradiction_type",
        size="ai_risk_count",
        sizes=(60, 300),
        ax=ax,
    )
    for _, row in top.iterrows():
        ax.text(row["ai_ratio"], row["risk_framing_ratio"], row["label"], fontsize=8)
    ax.set_title("Contradiction Profiles in AI Disclosure")
    ax.set_xlabel("AI ratio")
    ax.set_ylabel("AI-risk / AI")
    fig.savefig(PLOTS / "contradiction_profiles.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_leader_follower(leader_follow: pd.DataFrame) -> None:
    if leader_follow.empty:
        return
    yearly = leader_follow.groupby("year")["max_similarity_to_leader"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(yearly["year"], yearly["max_similarity_to_leader"], marker="o", linewidth=2.8, color="#6a4c93")
    ax.set_title("How Much AI-Risk Language Resembles Large-Platform Leaders")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average max similarity to leader")
    fig.savefig(PLOTS / "leader_follower_trend.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(event_df: pd.DataFrame, yearly_conv: pd.DataFrame, breadth: pd.DataFrame) -> None:
    lines = ["# Research Extensions Summary", ""]
    if not event_df.empty:
        pre = event_df[event_df["window"] == "Pre-2023"].iloc[0]
        post = event_df[event_df["window"] == "2023-2025"].iloc[0]
        lines.append("## Event-Window Shift")
        lines.append("")
        lines.append(f"- Average AI ratio rises from {pre['avg_ai_ratio']:.4f} pre-2023 to {post['avg_ai_ratio']:.4f} in 2023-2025.")
        lines.append(f"- Average AI-risk ratio rises from {pre['avg_ai_risk_ratio']:.4f} to {post['avg_ai_risk_ratio']:.4f}.")
        lines.append(f"- Risk framing share rises from {pre['avg_risk_framing']:.3f} to {post['avg_risk_framing']:.3f}.")
        lines.append("")
    if not yearly_conv.empty:
        lines.append("## Convergence")
        lines.append("")
        latest = yearly_conv.sort_values("year").iloc[-1]
        earliest = yearly_conv.sort_values("year").iloc[0]
        lines.append(f"- Average pairwise similarity in AI-risk disclosure rises from {earliest['avg_pairwise_similarity']:.3f} in {earliest['year']} to {latest['avg_pairwise_similarity']:.3f} in {latest['year']}.")
        lines.append("")
    if not breadth.empty:
        b = breadth.groupby("year")[["ai_section_breadth", "ai_risk_section_breadth"]].mean().reset_index()
        first = b.iloc[0]
        last = b.iloc[-1]
        lines.append("## Breadth")
        lines.append("")
        lines.append(f"- Average AI section breadth increases from {first['ai_section_breadth']:.2f} sections in {first['year']} to {last['ai_section_breadth']:.2f} in {last['year']}.")
        lines.append(f"- Average AI-risk breadth increases from {first['ai_risk_section_breadth']:.2f} to {last['ai_risk_section_breadth']:.2f}.")
        lines.append("")
    lines.append("## Optional External Data That Would Unlock More")
    lines.append("")
    lines.append("- SEC/Compustat/WRDS firm metadata with SIC/NAICS/GICS or Fama-French industry mapping for sector-level comparison.")
    lines.append("- Firm size and market variables such as market cap, assets, sales, or returns for testing whether larger or more visible firms lead disclosure shifts.")
    lines.append("- Filing acceptance timestamps or exact EDGAR metadata if you want tighter event studies around disclosure dates.")
    lines.append("- Patent, litigation, or cyber incident data if you want to link disclosure themes to real external shocks.")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    data = load()

    maturity, transitions = build_maturity(data["company_year"], data["section_metrics"])
    adoption = adoption_timing(data["company_summary"])
    migration = section_migration(data["section_metrics"])
    breadth = breadth_analysis(data["section_metrics"], data["company_year"])
    yearly_conv, boiler = convergence_and_boilerplate(data["ai_risk_sentences"], data["company_year"])
    leader_follow = leader_follower(data["ai_risk_sentences"], data["company_year"])
    risk_year = risk_comentions(data["ai_risk_sentences"])
    framing_year, framing_company = framing_analysis(data["ai_sentences"], data["ai_risk_sentences"])
    ai_year, air_year = topic_evolution(
        data["ai_topic_assignments"],
        data["ai_risk_topic_assignments"],
        data["ai_topics"],
        data["ai_risk_topics"],
    )
    aggressiveness = disclosure_aggressiveness(data["company_year"], breadth, boiler)
    contradictions = contradiction_analysis(data["company_year"])
    event_df = event_window(data["company_year"])

    plot_maturity(maturity)
    plot_migration(migration)
    plot_breadth(breadth)
    plot_convergence(yearly_conv, boiler)
    plot_topics(ai_year, air_year)
    plot_risk_network()
    plot_contradictions(contradictions)
    plot_leader_follower(leader_follow)
    write_summary(event_df, yearly_conv, breadth)
    print(f"Research extensions written to {OUT}")


if __name__ == "__main__":
    main()
