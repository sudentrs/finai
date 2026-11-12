"""Rebuild non-patent publication tables with explicit sample and definition audits."""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

ROOT = Path(os.environ.get("FINAI_DATA_ROOT", Path(__file__).resolve().parent)).resolve()
OUT = Path(os.environ.get("FINAI_TABLES_OUT", ROOT / "analysis/publication_tables_20261112")).resolve()
SOURCE = ROOT / "analysis/professor_filing_level_package_20260804"
KEY = ["cik", "filed_date", "period_end"]
LABELS = {"ln_at": "Log(total assets)", "rd_at": "R&D / assets", "rd_missing": "R&D missing",
          "capx_at": "Capex / assets", "tobinq": "Tobin's q", "roa": "Operating ROA",
          "tangibility": "Tangibility", "cash_at": "Cash / assets", "leverage": "Book leverage",
          "sales_growth": "Sales growth", "ln_item1": "Log(Item 1 sentences)",
          "ln_document": "Log(10-K sentences)"}
BASE = [x for x in LABELS if x != "rd_missing"]


def identifiers(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")


def date_parse(series):
    return pd.to_datetime(pd.to_numeric(series, errors="coerce").astype("Int64").astype(str),
                          format="%Y%m%d", errors="coerce")


def clean_keys(frame):
    frame = frame.copy()
    frame["cik"] = identifiers(frame.cik)
    for key in ("filed_date", "period_end"):
        frame[key] = pd.to_numeric(frame[key], errors="coerce").astype("Int64")
    return frame


def industry(sic):
    lookup, code = {}, None
    for line in (ROOT / "Siccodes48.txt").read_text().splitlines():
        header = re.match(r"^\s*(\d+)\s+[A-Za-z]", line)
        interval = re.match(r"^\s*(\d{4})-(\d{4})", line)
        if header:
            code = int(header.group(1))
        if interval:
            lo, hi = map(int, interval.groups())
            lookup.update(dict.fromkeys(range(lo, hi + 1), code))
    return pd.to_numeric(sic, errors="coerce").map(lookup)


def build_samples():
    item = clean_keys(pd.read_csv(SOURCE / "ITEM_1_filing_standardized.csv", low_memory=False))
    document = clean_keys(pd.read_csv(SOURCE / "DOCUMENT_filing_standardized.csv",
                                      usecols=KEY + ["total_sentences"]))
    if item.duplicated(KEY).any() or document.duplicated(KEY).any():
        raise ValueError("Ambiguous filing-level source keys")
    item = item.merge(document.rename(columns={"total_sentences": "document_sentences"}),
                      on=KEY, how="left", validate="one_to_one")
    item["gvkey"] = identifiers(item.compustat_gvkey)
    item["permco"] = identifiers(item.crsp_permco)
    item["ff48"] = industry(item.sic)
    item["expertise_ratio"] = item.ai_expertise_ratio
    item["ai_ratio_used"] = item.ai_ratio
    item["reported_year"] = date_parse(item.period_end).dt.year
    item = item[item.reported_year.between(2015, 2025) &
                item.filed_date.between(20150101, 20251231)].copy()
    columns = KEY + ["mdate", "prc", "expertiseratio", "airatio", "permco",
                     "ff48", "crsp_permno", "compustat_gvkey"]
    panel = clean_keys(pd.read_stata(ROOT / "publication pack/ITEM1_reg.dta",
                                     columns=columns, convert_categoricals=False))
    panel["permco"] = identifiers(panel.permco)
    panel = panel.loc[panel.prc.ge(5) & panel.expertiseratio.notna() & panel.airatio.notna()].copy()
    counts = panel.groupby(KEY).agg(eligible_stock_months=("mdate", "size"),
                                    expertise_versions=("expertiseratio", "nunique"),
                                    ai_versions=("airatio", "nunique"),
                                    permco_versions=("permco", "nunique")).reset_index()
    if counts.expertise_versions.gt(1).any() or counts.ai_versions.gt(1).any():
        raise ValueError("Within-filing ratios change in the professor's panel")
    roster = panel.sort_values(["mdate", "crsp_permno"]).drop_duplicates(KEY)
    roster = roster[KEY + ["permco", "ff48", "compustat_gvkey", "expertiseratio", "airatio"]]
    filtered = item.drop(columns=["permco", "ff48"]).merge(
        roster.rename(columns={"compustat_gvkey": "professor_gvkey"}),
        on=KEY, how="inner", validate="one_to_one")
    filtered["expertise_ratio"] = filtered.expertiseratio
    filtered["ai_ratio_used"] = filtered.airatio
    filtered["gvkey"] = identifiers(filtered.professor_gvkey).combine_first(filtered.gvkey)
    filtered = filtered.merge(counts, on=KEY, validate="one_to_one")
    filtered["ratio_difference_vs_source"] = (
        filtered.expertise_ratio.sub(filtered.ai_expertise_ratio).abs().gt(1e-6) |
        filtered.ai_ratio_used.sub(filtered.ai_ratio).abs().gt(1e-6))
    counts.to_csv(OUT / "eligible_filing_month_audit.csv", index=False)
    return {"broad": item, "professor_price_ge5": filtered}


def accounting():
    raw = pd.read_csv(ROOT / "x9m8tdipe4kxamf0.csv/x9m8tdipe4kxamf0.csv", low_memory=False)
    raw["gvkey"] = identifiers(raw.gvkey)
    raw = raw[raw.curcd.eq("USD") & raw.datafmt.eq("STD") &
              raw.consol.eq("C") & raw.indfmt.eq("INDL")].copy()
    raw.loc[~raw["at"].gt(0)].to_csv(OUT / "accounting_without_positive_assets.csv", index=False)
    raw = raw.loc[raw["at"].gt(0)].copy()
    if raw.duplicated(["gvkey", "fyear"]).any():
        raise ValueError("Ambiguous standardized industrial accounting records")
    raw = raw.sort_values(["gvkey", "fyear"])
    previous_sales = raw.groupby("gvkey").sale.shift()
    consecutive = raw.fyear.sub(raw.groupby("gvkey").fyear.shift()).eq(1)
    raw["sales_growth"] = (raw.sale / previous_sales - 1).where(previous_sales.gt(0) & consecutive)
    positive = raw["at"].where(raw["at"].gt(0))
    raw["ln_at"] = np.log(positive)
    raw["rd_missing"] = raw.xrd.isna().astype(int)
    for name, value in {"rd_at": raw.xrd, "capx_at": raw.capx, "roa": raw.oibdp,
                        "tangibility": raw.ppent, "cash_at": raw.che,
                        "leverage": raw.dltt + raw.dlc,
                        "tobinq": raw["at"] - raw.ceq + raw.prcc_f * raw.csho}.items():
        raw[name] = value / positive
    return raw


def add_accounting(frame, raw):
    frame = frame.copy()
    frame["period_date"] = date_parse(frame.period_end)
    dates = pd.read_csv(ROOT / "analysis/advanced/research_extensions/industry_market/compustat_annual_primary.csv",
                        usecols=["gvkey", "datadate", "fyear"])
    dates["gvkey"] = identifiers(dates.gvkey)
    dates["period_date"] = pd.to_datetime(dates.datadate, errors="coerce")
    dates = dates[["gvkey", "period_date", "fyear"]].drop_duplicates()
    if dates.duplicated(["gvkey", "period_date"]).any():
        raise ValueError("Conflicting explicit accounting fiscal-year map")
    frame = frame.merge(dates, on=["gvkey", "period_date"], how="left", validate="many_to_one")
    frame["fyear_match"] = np.where(frame.fyear.notna(), "exact_gvkey_datadate", "unmatched")
    # Month-matched fiscal-calendar fallback is kept separately in the audit.
    fallback = frame.period_date.dt.year - frame.period_date.dt.month.le(5).astype(int)
    probe = frame.loc[frame.fyear.isna(), ["gvkey", "period_date"]].copy()
    probe["fallback"] = fallback[frame.fyear.isna()]
    probe["row"] = probe.index
    probe = probe.merge(raw[["gvkey", "fyear", "fyr"]], left_on=["gvkey", "fallback"],
                        right_on=["gvkey", "fyear"], how="left", validate="many_to_one")
    valid = probe.fyr.eq(probe.period_date.dt.month)
    rows = probe.loc[valid, "row"].to_numpy()
    frame.loc[rows, "fyear"] = probe.loc[valid, "fyear"].to_numpy()
    frame.loc[rows, "fyear_match"] = "month_matched_fiscal_calendar"
    keep = ["gvkey", "fyear", "costat", "fyr", "xrd", *[x for x in BASE if x not in ("ln_item1", "ln_document")],
            "rd_missing"]
    frame = frame.merge(raw[keep], on=["gvkey", "fyear"], how="left", validate="many_to_one")
    frame["ln_item1"] = np.log(frame.total_sentences.where(frame.total_sentences.gt(0)))
    frame["ln_document"] = np.log(frame.document_sentences.where(frame.document_sentences.gt(0)))
    frame["expertise_pct"] = frame.expertise_ratio * 100
    frame["ai_pct"] = frame.ai_ratio_used * 100
    frame["permco"] = identifiers(frame.permco)
    return frame


def demean(values, groups):
    out = np.asarray(values, dtype=float).copy()
    for _ in range(1000):
        previous = out.copy()
        for codes in groups:
            sizes = np.bincount(codes)
            means = np.vstack([np.bincount(codes, weights=out[:, j], minlength=len(sizes)) / sizes
                               for j in range(out.shape[1])]).T
            out -= means[codes]
        if np.max(np.abs(out - previous)) < 1e-9:
            return out
    raise ValueError("Fixed-effect absorption did not converge")


def fe_rank(groups):
    a, b = groups
    na, nb = a.max() + 1, b.max() + 1
    graph = coo_matrix((np.ones(len(a)), (a, b + na)), shape=(na + nb, na + nb))
    count = connected_components(graph + graph.T, directed=False, return_labels=False)
    return int(na + nb - count)


def regression(frame, dependent, controls, firm_fe):
    columns = [dependent, *controls, "permco", "fyear", "ff48"]
    data = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(dict.fromkeys(columns))).copy()
    groups = [pd.factorize(data.permco if firm_fe else data.ff48)[0], pd.factorize(data.fyear)[0]]
    matrix = demean(data[[dependent, *controls]].to_numpy(), groups)
    y, x = matrix[:, 0], matrix[:, 1:]
    if np.linalg.matrix_rank(x) < len(controls):
        return None, {"status": "rank_deficient", "n": len(data)}
    fit = sm.OLS(y, x).fit()
    n, g = len(data), data.permco.nunique()
    k = len(controls) + fe_rank(groups)
    covariance = cov_cluster(fit, data.permco, use_correction=False)
    covariance *= g / (g - 1) * (n - 1) / (n - k)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    t = fit.params / se
    p = 2 * stats.t.sf(np.abs(t), g - 1)
    ss_total = np.square(data[dependent] - data[dependent].mean()).sum()
    adjusted = 1 - (np.square(fit.resid).sum() / (n - k)) / (ss_total / (n - 1))
    rows = [dict(variable=name, coefficient=float(coef), se=float(s), t=float(tv), p=float(pv))
            for name, coef, s, tv, pv in zip(controls, fit.params, se, t, p)]
    return rows, {"status": "estimated", "n": n, "firms": g, "adjusted_r2": float(adjusted),
                  "fe_rank": k - len(controls), "cluster_df": g - 1,
                  "firm_fe": firm_fe, "industry_fe": not firm_fe}


def estimates(frame, folder):
    raw = frame.copy()
    d = raw.loc[raw.fyear.between(2015, 2025)].copy()
    winsor_cols = [*BASE, "expertise_pct", "ai_pct"]
    bounds = []
    for year, indices in d.groupby("fyear").groups.items():
        for col in winsor_cols:
            lo, hi = d.loc[indices, col].quantile([.01, .99])
            d.loc[indices, col] = d.loc[indices, col].clip(lo, hi)
            bounds.append(dict(fyear=year, variable=col, p01=lo, p99=hi))
    pd.DataFrame(bounds).to_csv(folder / "winsorization_bounds.csv", index=False)
    outputs, summaries = [], []
    for y in ("expertise_pct", "ai_pct"):
        for spec in (1, 2, 3, 4, 6):
            controls = BASE.copy()
            sample = d.copy()
            if spec == 4:
                controls.remove("rd_at")
            elif spec == 3:
                sample = sample.loc[sample.rd_missing.eq(0)]
            else:
                sample.loc[sample.rd_missing.eq(1) & sample.ln_at.notna(), "rd_at"] = 0
            if spec in (2, 6):
                controls.append("rd_missing")
            rows, info = regression(sample, y, controls, spec == 6)
            info.update(dependent=y, specification=spec)
            summaries.append(info)
            if rows:
                outputs.extend(dict(row, dependent=y, specification=spec) for row in rows)
    coef = pd.DataFrame(outputs)
    summary = pd.DataFrame(summaries)
    coef.to_csv(folder / "table_331_coefficients.csv", index=False)
    summary.to_csv(folder / "table_331_model_statistics.csv", index=False)
    return coef, summary


def excerpts(frame, sentences, folder, length_filter):
    d = frame.loc[frame.reported_year.between(2015, 2025)].copy()
    if length_filter:
        d = d.loc[d.total_sentences.ge(25)]
    if d.duplicated(["cik", "period_end"]).any():
        # Latest filing for a reporting period, not an extra firm-year.
        d = d.sort_values(["filed_date", "accession"]).drop_duplicates(["cik", "period_end"], keep="last")
    firms = d.groupby("cik").agg(filings=("accession", "size"), expertise=("expertise_ratio", "mean"),
                                ai=("ai_ratio_used", "mean"), max_expertise=("expertise_ratio", "max"))
    eligible = firms.loc[firms.filings.ge(2)]
    selected = [("A", eligible.sort_values(["expertise", "ai"], ascending=False).head(10)),
                ("B", eligible.loc[eligible.max_expertise.eq(0) & eligible.ai.gt(0)]
                 .sort_values("ai", ascending=False).head(5))]
    results = []
    for panel, chosen in selected:
        for rank, (cik, avg) in enumerate(chosen.iterrows(), 1):
            measure = "expertise_ratio" if panel == "A" else "ai_ratio_used"
            filing = d.loc[d.cik.eq(cik)].sort_values(
                [measure, "reported_year", "filed_date", "accession"],
                ascending=[False, False, False, True]).iloc[0]
            candidates = sentences.loc[sentences.cik.eq(cik) & sentences.accession.eq(filing.accession)]
            raw_candidates = len(candidates)
            substantive_candidates = int(candidates.mutually_exclusive_category.ne("noise").sum())
            candidates = candidates.loc[candidates.sentence.fillna("").str.split().str.len().ge(10)]
            if panel == "A":
                candidates = candidates.loc[candidates.mutually_exclusive_category.eq("ai_expertise")]
            else:
                candidates = candidates.loc[~candidates.mutually_exclusive_category.eq("noise")]
            candidate = candidates.sort_values(["topic_weight", "occurrence_index"], ascending=[False, True])
            result = {"panel": panel, "rank": rank, "company": filing.company_name, "cik": cik,
                      "permco": filing.permco, "fiscal_year_label": int(filing.reported_year),
                      "filings": int(avg.filings), "average_expertise_pct": avg.expertise * 100,
                      "average_ai_pct": avg.ai * 100, "accession": filing.accession,
                      "filed_date": filing.filed_date, "period_end": filing.period_end,
                      "item1_sentences": filing.total_sentences,
                      "raw_ai_occurrence_candidates": raw_candidates,
                      "nonnoise_occurrence_candidates": substantive_candidates,
                      "excerpt": "", "excerpt_status": "no_eligible_sentence"}
            if len(candidate):
                sentence = candidate.iloc[0]
                result.update(excerpt=sentence.sentence, sentence_id=sentence.occurrence_index,
                              assigned_topic=sentence.topic_id, topic_weight=sentence.topic_weight,
                              excerpt_status="selected")
            results.append(result)
    result = pd.DataFrame(results)
    result.to_csv(folder / ("table_311_length25.csv" if length_filter else "table_311.csv"), index=False)
    return result


def latex_escape(text):
    return str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def export_tex(coef, summary, folder):
    for y in ("expertise_pct", "ai_pct"):
        specs = (1, 2, 3, 4, 6)
        lines = [r"\begin{table}[htbp]\centering\small",
                 r"\caption{Draft firm-characteristic regressions: " +
                 ("ExpertiseRatio" if y == "expertise_pct" else "AIRatio") + r"}",
                 r"\begin{tabular}{lrrrrr}\toprule",
                 " & (1) & (2) & (3) & (4) & (6) " + r"\\\midrule"]
        for variable, label in LABELS.items():
            cells, stats_cells = [], []
            for spec in specs:
                row = coef.loc[coef.dependent.eq(y) & coef.specification.eq(spec) & coef.variable.eq(variable)]
                if row.empty:
                    cells.append("")
                    stats_cells.append("")
                else:
                    row = row.iloc[0]
                    star = "***" if row.p < .01 else "**" if row.p < .05 else "*" if row.p < .1 else ""
                    cells.append(f"{row.coefficient:.3f}{star}")
                    stats_cells.append(f"({row.t:.2f})")
            lines += [latex_escape(label) + " & " + " & ".join(cells) + r"\\",
                      " & " + " & ".join(stats_cells) + r"\\"]
        lines.append(r"\midrule")
        for field, label in (("n", "Observations"), ("firms", "Firms"), ("adjusted_r2", "Adjusted R-squared")):
            values = []
            for spec in specs:
                row = summary.loc[summary.dependent.eq(y) & summary.specification.eq(spec)].iloc[0]
                values.append(f"{row[field]:,.0f}" if field != "adjusted_r2" else f"{row[field]:.3f}")
            lines.append(label + " & " + " & ".join(values) + r"\\")
        lines += [r"\bottomrule\end{tabular}",
                  r"\par\vspace{4pt}\begin{minipage}{0.98\linewidth}\footnotesize",
                  r"\textit{Notes:} Draft, not a final reproduction of the manuscript. Ratios are in percentage points. "
                  r"All columns include fiscal-year effects; columns 1--4 include FF48 effects and column 6 firm effects. "
                  r"Firm-clustered t-statistics are in parentheses; *** p<0.01, ** p<0.05, * p<0.10 (two-sided). "
                  r"Continuous variables are winsorized at 1/99 percentiles by accounting fiscal year before R\&D "
                  r"imputation. Column 1 sets missing R\&D to zero; 2 and 6 add a missing indicator; 3 uses reporters "
                  r"only; 4 omits R\&D. Firm age is unavailable and omitted. "
                  r"10-K length uses sentence count, not submission-file bytes. Column 5 is not estimated: "
                  r"the reduced specification is not yet agreed. See the accompanying audit for sample and matching details.",
                  r"\end{minipage}\end{table}"]
        (folder / f"table_331_{y}.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples, comp = build_samples(), accounting()
    sentences = clean_keys(pd.read_csv(ROOT /
        "analysis/mutually_exclusive_ai_constructs_occurrence/original_ai_sentence_occurrence_assignments.csv",
        low_memory=False))
    sentences = sentences.loc[sentences.section.eq("ITEM_1")]
    audit, workbook = {}, {}
    for name, frame in samples.items():
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        frame = add_accounting(frame, comp)
        frame.to_csv(folder / "filing_analysis_data.csv", index=False)
        tables = excerpts(frame, sentences, folder, False)
        length = excerpts(frame, sentences, folder, True)
        coef, summary = estimates(frame, folder)
        export_tex(coef, summary, folder)
        audit[name] = {"filings": len(frame), "ciks": frame.cik.nunique(),
                       "accounting_link": frame.fyear_match.value_counts().to_dict(),
                       "accounting_assets_present": int(frame.ln_at.notna().sum()),
                       "expertise_above_ai": int(frame.expertise_ratio.gt(frame.ai_ratio_used + 1e-7).sum()),
                       "missing_cluster": int(frame.permco.isna().sum()),
                       "table311_excerpt_missing": int(tables.excerpt_status.ne("selected").sum()),
                       "model_statistics": summary.to_dict("records")}
        workbook[name] = {"examples": tables.fillna("").to_dict("records"),
                          "examples_length25": length.fillna("").to_dict("records"),
                          "coefficients": coef.fillna("").to_dict("records"),
                          "models": summary.fillna("").to_dict("records")}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    (OUT / "workbook_data.json").write_text(json.dumps(workbook, default=str))
    print(json.dumps(audit, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
