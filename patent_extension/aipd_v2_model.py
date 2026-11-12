"""Compare recency and component recipes; freeze a single deployment model."""
import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from aipd_extension import FRAME, TARGET, fingerprint, metrics, operating_threshold, score_recent, write_json
from aipd_v2_prepare import FLAGS, SEED, V1, V2

RECIPES = ("binary", "binary_recent_weight", "components", "components_recent_weight")


def split(frame, year, fresh=False):
    old = frame[~frame.fresh_benchmark & frame.hash.ne("")]
    if fresh:
        test = frame[frame.fresh_benchmark & frame.hash.ne("")]
        end = year - 1
    else:
        test = old[old.year.eq(year)]
        end = year - 1
    fit_pool = old[old.year.le(end) & ~old.hash.isin(test.hash)]
    latest = fit_pool[fit_pool.year.eq(end)]
    cal = latest.sample(frac=.25, random_state=SEED)
    fit = fit_pool[~fit_pool.hash.isin(cal.hash)]
    return fit, cal, test


def vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=180_000,
                           sublinear_tf=True, dtype=np.float32)


def fit_recipe(x, fit, recipe):
    labels = FLAGS if recipe.startswith("components") else [TARGET]
    weights = np.exp2((fit.year - fit.year.max()) / 2).to_numpy() if recipe.endswith("weight") else None
    if weights is not None:
        weights /= weights.mean()
    return {label: LogisticRegression(solver="liblinear", C=2, max_iter=300, random_state=SEED)
            .fit(x, fit[label], sample_weight=weights) for label in labels}


def combined(heads, x):
    return np.maximum.reduce([head.predict_proba(x)[:, 1] for head in heads.values()])


class Scorer:
    def __init__(self, bundle):
        self.bundle = bundle

    def predict_proba(self, texts):
        p = combined(self.bundle["heads"], self.bundle["vectorizer"].transform(texts))
        return np.column_stack((1 - p, p))


def report_heads(heads, cal, xc, test, xt):
    results = {}
    for label, head in heads.items():
        threshold = operating_threshold(cal[label], head.predict_proba(xc)[:, 1])
        results[label] = dict(threshold=threshold, **metrics(test[label], head.predict_proba(xt)[:, 1], threshold))
    return results


def train(out):
    frame = pd.read_csv(out / "historical_abstracts_labels.csv", keep_default_na=False,
                        dtype={"pgpub_id": str, "application_id": str})
    if frame.application_id.duplicated().any():
        raise ValueError("Nonunique application keys")
    results, technologies = [], []
    for year in (2021, 2022):
        fit, cal, test = split(frame, year)
        vec = vectorizer()
        x = vec.fit_transform(fit.application_abstract)
        xc, xt = (vec.transform(part.application_abstract) for part in (cal, test))
        for recipe in RECIPES:
            heads = fit_recipe(x, fit, recipe)
            threshold = operating_threshold(cal[TARGET], combined(heads, xc))
            result = metrics(test[TARGET], combined(heads, xt), threshold)
            results.append(dict(test_year=year, recipe=recipe, fit_n=len(fit), calibration_n=len(cal),
                                threshold=threshold, **result))
            for label, row in report_heads(heads, cal, xc, test, xt).items():
                technologies.append(dict(test_year=year, recipe=recipe, target=label, **row))
            print(year, recipe, round(result["f1"], 4), flush=True)
    rolling = pd.DataFrame(results)
    rolling.to_csv(out / "rolling_benchmarks.csv", index=False)
    pd.DataFrame(technologies).to_csv(out / "rolling_component_benchmarks.csv", index=False)
    winner = rolling.groupby("recipe").f1.mean().sort_values(ascending=False, kind="stable").index[0]
    write_json(out / "recipe_selection.json", {"selected": winner,
               "rule": "highest equal-weight mean 2021 and 2022 forward-test F1; alphabetical tie-break",
               "mean_f1": rolling.groupby("recipe").f1.mean().to_dict(),
               "recent_weight": "two-year half-life; weights normalized to mean one"})
    fit, cal, test = split(frame, 2023, fresh=True)
    vec = vectorizer()
    x = vec.fit_transform(fit.application_abstract)
    xc = vec.transform(cal.application_abstract)
    heads = fit_recipe(x, fit, winner)
    cutoff = operating_threshold(cal[TARGET], combined(heads, xc))
    benchmark_path = out / "frozen_benchmark.joblib"
    joblib.dump(dict(vectorizer=vec, heads=heads, threshold=cutoff, recipe=winner), benchmark_path, compress=3)
    write_json(out / "benchmark_freeze.json", {"sha256": fingerprint(benchmark_path),
               "recipe": winner, "threshold": cutoff, "fit_through": 2022, "calibration_year": 2022})
    xt = vec.transform(test.application_abstract)
    p = combined(heads, xt)
    predictions = test.drop(columns="application_abstract").assign(score=p, predicted=(p >= cutoff).astype(int))
    predictions.to_csv(out / "fresh_2023_predictions.csv", index=False)
    component_recipe = winner if winner.startswith("components") else "components"
    component_heads = heads if winner.startswith("components") else fit_recipe(x, fit, component_recipe)
    component_metrics = report_heads(component_heads, cal, xc, test, xt)
    strata = {}
    for label in FLAGS:
        is_component = test[label].eq(1)
        strata[label] = {"official_positive_n": int(is_component.sum()),
                         "combined_label_recall_among_official_component_positives":
                         float((p[is_component] >= cutoff).mean()) if is_component.any() else None}
    prior = pd.read_csv(V1 / "historical_split_manifest.csv", usecols=["split", "hash", "included"])
    v1_exposed_hash = prior.loc[prior.included & prior.split.isin(["train", "development"]), "hash"]
    same = ~test.hash.isin(v1_exposed_hash)
    v1 = joblib.load(V1 / "frozen_surrogate.joblib")
    p1 = v1["model"].predict_proba(test.loc[same, "application_abstract"])[:, 1]
    write_json(out / "fresh_2023_benchmark.json", {
        "recipe": winner, "fit_n": len(fit), "calibration_n": len(cal),
        "overall": metrics(test[TARGET], p, cutoff),
        "paired_comparison": {"v1": metrics(test.loc[same, TARGET], p1, v1["threshold"]),
                              "v2": metrics(test.loc[same, TARGET], p[same], cutoff)},
        "component_recipe": component_recipe, "component_heads": component_metrics,
        "combined_recall_by_official_component": strata,
        "family_overlap": "unresolved; no exact normalized abstract or application overlap in v2",
    })
    print(f"Fresh 2023: {len(test):,} test; recipe {winner}; F1 {metrics(test[TARGET], p, cutoff)['f1']:.4f}",
          flush=True)

    # Fresh test and its exact duplicates never enter the deployment fit or calibration.
    pool = frame[~frame.fresh_benchmark & frame.hash.ne("") & ~frame.hash.isin(test.hash)]
    cal = pool[pool.year.eq(2023)].sample(frac=.25, random_state=SEED)
    fit = pool[~pool.hash.isin(cal.hash)]
    vec = vectorizer()
    heads = fit_recipe(vec.fit_transform(fit.application_abstract), fit, winner)
    cal_score = combined(heads, vec.transform(cal.application_abstract))
    cutoff = operating_threshold(cal[TARGET], cal_score)
    path = out / "frozen_deployment.joblib"
    joblib.dump(dict(vectorizer=vec, heads=heads, threshold=cutoff, recipe=winner), path, compress=3)
    write_json(out / "deployment_freeze.json", {"sha256": fingerprint(path), "recipe": winner,
               "threshold": cutoff, "fit_n": len(fit), "fit_years": fit.groupby("year").size().to_dict(),
               "calibration_n": len(cal), "calibration_year": 2023,
               "fresh_benchmark_excluded": True, "score": "binary probability or max component probability",
               "scope": "same frozen model for 2024 and 2025 first utility pre-grant publications",
               "calibration_agreement": metrics(cal[TARGET], cal_score, cutoff)})


def deploy(out):
    model_path = out / "frozen_deployment.joblib"
    freeze = json.loads((out / "deployment_freeze.json").read_text())
    if fingerprint(model_path) != freeze["sha256"]:
        raise ValueError("Deployment model changed")
    bundle = joblib.load(model_path)
    with sqlite3.connect(FRAME.as_uri() + "?mode=ro", uri=True) as db:
        recent = pd.read_sql_query(
            """SELECT application_id, pgpub_id, published_date, filing_date FROM applications
               WHERE published_date BETWEEN '2024-01-01' AND '2025-12-31'""", db)
    coverage = score_recent(Scorer(bundle), bundle["threshold"], V1 / "staged_recent_abstracts.csv", recent, out)
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=V2)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if not args.score_only:
        train(args.out)
    deploy(args.out)
