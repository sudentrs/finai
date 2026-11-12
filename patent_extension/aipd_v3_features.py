"""Test stronger patent-text features before replacing the AIPD extension."""
import gc
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from aipd_extension import FRAME, ROOT, TARGET, fingerprint, metrics, operating_threshold, write_json
from aipd_v2_model import Scorer, split, vectorizer
from aipd_v2_prepare import SEED, V1, V2

OUT = ROOT / "aipd_surrogate_v3"
RECIPES = ("title_logistic", "title_word_char_svm", "title_word_char_nb_svm")


def titles(year_start, year_end):
    with sqlite3.connect(FRAME.as_uri() + "?mode=ro", uri=True) as db:
        return pd.read_sql_query(
            """SELECT pgpub_id, title FROM applications
               WHERE published_date BETWEEN ? AND ?""", db,
            params=(f"{year_start}-01-01", f"{year_end}-12-31"),
        ).set_index("pgpub_id", verify_integrity=True).title


def char_input(frame):
    # Bound memory and retain title plus the leading abstract text.
    return frame.title + " " + frame.application_abstract.str[:650]


def matrices(fit, others, use_char):
    word = vectorizer()
    title = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000,
                            sublinear_tf=True, dtype=np.float32)
    char = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4), n_features=2**18,
                             alternate_sign=False, binary=True, dtype=np.float32) if use_char else None
    wx, tx = word.fit_transform(fit.application_abstract), title.fit_transform(fit.title)
    enc = (word, title, char)
    x = combine_features(wx, tx, char, fit)
    return enc, x, [transform(enc, part) for part in others]


def combine_features(wx, tx, char, frame):
    blocks = [wx, tx * np.float32(.65)]
    if char is not None:
        blocks.append(char.transform(char_input(frame)) * np.float32(.8))
    return sparse.hstack(blocks, format="csr", dtype=np.float32)


def transform(enc, frame):
    word, title, char = enc
    return combine_features(word.transform(frame.application_abstract), title.transform(frame.title), char, frame)


def nb_weights(x, y):
    # Presence counts estimate a smoothed class-conditional feature ratio.
    counts = []
    for label in (0, 1):
        rows = y == label
        count = np.bincount(x[rows].indices, minlength=x.shape[1])
        counts.append((count + 1) / (rows.sum() + 1))
    return np.log(counts[1] / counts[0]).astype(np.float32)


def fit_head(x, y, recipe):
    if recipe == "title_logistic":
        head = LogisticRegression(solver="liblinear", C=2, max_iter=300, random_state=SEED)
        ratio = None
    else:
        head = LinearSVC(C=.5, dual=False, max_iter=2000, random_state=SEED)
        ratio = nb_weights(x, y) if recipe.endswith("nb_svm") else None
    head.fit(weighted(x, ratio), y)
    return head, ratio


def weighted(x, ratio):
    return x.multiply(ratio).tocsr() if ratio is not None else x


def score(head, ratio, x):
    x = weighted(x, ratio)
    if hasattr(head, "predict_proba"):
        return head.predict_proba(x)[:, 1]
    # A monotone transform preserves the ranking; this is not a calibrated probability.
    return expit(head.decision_function(x))


def evaluate(y, p, cutoff):
    result = metrics(y, p, cutoff)
    del result["brier"]
    return result


def fit_candidate(fit, cal, test, recipe):
    enc, x, (xc, xt) = matrices(fit, [cal, test], recipe != "title_logistic")
    head, ratio = fit_head(x, fit[TARGET].to_numpy(), recipe)
    cutoff = operating_threshold(cal[TARGET], score(head, ratio, xc))
    p = score(head, ratio, xt)
    return dict(encoder=enc, head=head, ratio=ratio, threshold=cutoff, recipe=recipe), p


def run():
    OUT.mkdir(exist_ok=False)
    frame = pd.read_csv(V2 / "historical_abstracts_labels.csv", keep_default_na=False,
                        dtype={"pgpub_id": str, "application_id": str})
    frame["title"] = frame.pgpub_id.map(titles(2015, 2023)).fillna("")
    fit, cal, test = split(frame, 2022)
    rows = []
    for use_char, recipes in ((False, RECIPES[:1]), (True, RECIPES[1:])):
        _, x, (xc, xt) = matrices(fit, [cal, test], use_char)
        for recipe in recipes:
            head, ratio = fit_head(x, fit[TARGET].to_numpy(), recipe)
            cutoff = operating_threshold(cal[TARGET], score(head, ratio, xc))
            result = evaluate(test[TARGET], score(head, ratio, xt), cutoff)
            rows.append(dict(recipe=recipe, threshold=cutoff, **result))
            print("2022", recipe, round(result["f1"], 4), flush=True)
        del x, xc, xt
        gc.collect()
    comparison = pd.DataFrame(rows).sort_values(["f1", "recipe"], ascending=[False, True])
    comparison.to_csv(OUT / "selection_2022.csv", index=False)
    winner = comparison.iloc[0].recipe
    write_json(OUT / "selection_rule.json", {
        "recipe": winner, "selected_by": "highest 2022 F1 among challengers, alphabetic tie-break",
        "v2_replacement_screen": "at least 1 percentage point more F1 and no more than 1 point lower precision",
        "2023": "already examined in v2; comparison is diagnostic, not untouched confirmation",
    })
    fit, cal, test = split(frame, 2023, fresh=True)
    bundle, p = fit_candidate(fit, cal, test, winner)
    path = OUT / "benchmark_model.joblib"
    joblib.dump(bundle, path, compress=3)
    baseline = joblib.load(V2 / "frozen_benchmark.joblib")
    p0 = Scorer(baseline).predict_proba(test.application_abstract)[:, 1]
    v2 = evaluate(test[TARGET], p0, baseline["threshold"])
    v3 = evaluate(test[TARGET], p, bundle["threshold"])
    keep = v3["f1"] - v2["f1"] >= .01 and v3["precision"] - v2["precision"] >= -.01
    report = {"recipe": winner, "v2": v2, "v3": v3, "promote": bool(keep),
              "benchmark_model_sha256": fingerprint(path),
              "fit_n": len(fit), "calibration_n": len(cal), "benchmark_reused": True,
              "historical_missing_titles": int(frame.title.str.strip().eq("").sum()),
              "family_clearance": False}
    write_json(OUT / "paired_2023_benchmark.json", report)
    test[["pgpub_id", "published_date", TARGET]].assign(
        v2_score=p0, v3_score=p, v2_label=(p0 >= baseline["threshold"]).astype(int),
        v3_label=(p >= bundle["threshold"]).astype(int)).to_csv(OUT / "paired_2023_predictions.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)
    if keep:
        fit_deployment(frame, test, winner)


def fit_deployment(frame, test, winner):
    pool = frame[~frame.fresh_benchmark & frame.hash.ne("") & ~frame.hash.isin(test.hash)]
    cal = pool[pool.year.eq(2023)].sample(frac=.25, random_state=SEED)
    fit = pool[~pool.hash.isin(cal.hash)]
    enc, x, (xc,) = matrices(fit, [cal], winner != "title_logistic")
    head, ratio = fit_head(x, fit[TARGET].to_numpy(), winner)
    cutoff = operating_threshold(cal[TARGET], score(head, ratio, xc))
    path = OUT / "frozen_deployment.joblib"
    joblib.dump(dict(encoder=enc, head=head, ratio=ratio, threshold=cutoff, recipe=winner), path, compress=3)
    del x, xc
    write_json(OUT / "deployment_freeze.json", {"sha256": fingerprint(path), "threshold": cutoff,
               "recipe": winner, "fit_n": len(fit), "calibration_n": len(cal), "fit_through": 2023})
    deploy()


def deploy():
    freeze = json.loads((OUT / "deployment_freeze.json").read_text())
    model_path = OUT / "frozen_deployment.joblib"
    if fingerprint(model_path) != freeze["sha256"]:
        raise ValueError("Deployment fingerprint mismatch")
    bundle = joblib.load(model_path)
    types = {"pgpub_id": str, "application_id": str}
    frame = pd.read_csv(V2 / "publications_2024_2025.csv", dtype=types).set_index("pgpub_id", verify_integrity=True)
    title = titles(2024, 2025)
    output = OUT / "publications_2024_2025.csv"
    seen, first, missing_titles = set(), True, 0
    for part in pd.read_csv(V1 / "staged_recent_abstracts.csv", dtype=str, keep_default_na=False, chunksize=5000):
        if part.pgpub_id.duplicated().any() or not seen.isdisjoint(part.pgpub_id):
            raise ValueError("Repeated deployment text ID")
        seen.update(part.pgpub_id)
        part["title"] = part.pgpub_id.map(title).fillna("")
        missing_titles += int(part.title.str.strip().eq("").sum())
        rows = frame.loc[part.pgpub_id, ["application_id", "published_date", "filing_date", "score_status"]].reset_index()
        if not rows.score_status.eq("scored").all():
            raise ValueError("Unexpected unscored record in deployment text")
        values = score(bundle["head"], bundle["ratio"], transform(bundle["encoder"], part))
        rows["aipd_surrogate_score"] = values
        rows["aipd_surrogate_label"] = (values >= bundle["threshold"]).astype(int)
        rows.to_csv(output, mode="w" if first else "a", index=False, header=first)
        first = False
    if seen != set(frame.index):
        raise ValueError("Deployment publication frame did not reconcile")
    final = pd.read_csv(output, dtype=types, float_precision="round_trip")
    if not final.aipd_surrogate_label.eq(final.aipd_surrogate_score.ge(bundle["threshold"])).all():
        raise ValueError("Saved classification did not match the frozen threshold")
    coverage = final.assign(year=final.published_date.str[:4]).groupby("year").agg(
        scored=("pgpub_id", "size"), predicted_ai=("aipd_surrogate_label", "sum"))
    coverage.to_csv(OUT / "recent_coverage.csv")
    write_json(OUT / "output_verification.json", {"rows": len(final),
               "prediction_sha256": fingerprint(output), "missing_titles": missing_titles,
               "family_overlap_cleared": False, "recent_human_validation_complete": False})
    print(coverage.to_string(), flush=True)


if __name__ == "__main__":
    run()
