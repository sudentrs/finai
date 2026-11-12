"""Scale claims training and compare against v2 on a new publication sample."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from aipd_claims_benchmark import SOURCE, claims_for, vectorizer
from aipd_claims_fresh_sample import OUT
from aipd_extension import TARGET, fingerprint, metrics, normalized_hash, operating_threshold, write_json
from aipd_v2_model import Scorer
from aipd_v2_prepare import SEED, V2

TRAIN_YEARS = (2018, 2019, 2020, 2021, 2022)
WEIGHTS = (0, .15, .30, .45, .60, .75, 1)
COLS = {"pgpub_id": str, "application_id": str}


def frame_for(year):
    if year == 2023:
        return pd.read_csv(OUT / "new_2023_frame.csv", dtype=COLS, keep_default_na=False)
    hist = pd.read_csv(V2 / "historical_abstracts_labels.csv", dtype=COLS, keep_default_na=False)
    return hist.loc[hist.year.eq(year) & hist.hash.ne("")].copy()


def extract(year):
    frame = frame_for(year)
    source = SOURCE / f"pg_claims_{year}.tsv.zip"
    claims = claims_for(source, set(frame.pgpub_id))
    joined = frame.merge(claims, on="pgpub_id", how="left", validate="one_to_one")
    joined["claims_hash"] = joined.claim_text.fillna("").map(normalized_hash)
    write_json(OUT / f"claims_{year}_coverage.json", {
        "year": year, "requested": len(frame), "matched": int(joined.claims_hash.ne("").sum()),
        "archive_sha256": fingerprint(source),
    })
    joined[["pgpub_id", "claim_text", "claims_hash", "claim_count"]].to_csv(
        OUT / f"claims_{year}_joined.csv", index=False)
    print(f"Claims {year}: {len(claims):,}/{len(frame):,}", flush=True)


def training_parts():
    hist = pd.read_csv(V2 / "historical_abstracts_labels.csv", dtype=COLS, keep_default_na=False)
    prior_abstracts = set(hist.hash) - {""}
    hist = hist[hist.year.le(2022) & hist.hash.ne("")].copy()
    test = frame_for(2023)
    claim_cols = [pd.read_csv(OUT / f"claims_{y}_joined.csv", dtype=COLS, keep_default_na=False)
                  for y in (*TRAIN_YEARS, 2023)]
    claims = pd.concat(claim_cols, ignore_index=True)
    test = test.merge(claims, on="pgpub_id", how="left", validate="one_to_one")
    test = test[test.hash.ne("") & test.claims_hash.fillna("").ne("")].copy()
    exact_abstract_exclusions = int(test.hash.isin(prior_abstracts).sum())
    test = test[~test.hash.isin(prior_abstracts)].copy()
    if not set(hist.application_id).isdisjoint(test.application_id):
        raise ValueError("New test overlaps historical applications")
    hist = hist.merge(claims, on="pgpub_id", how="left", validate="one_to_one").fillna({"claims_hash": ""})
    hist = hist[~hist.hash.isin(test.hash) & ~hist.claims_hash.isin(test.claims_hash)]
    cal = hist[hist.year.eq(2022)].sample(frac=.25, random_state=SEED)
    cal = cal[cal.claims_hash.ne("")].copy()
    fit = hist[~hist.hash.isin(cal.hash) & ~hist.claims_hash.isin(cal.claims_hash)]
    claims_fit = fit[fit.claims_hash.ne("") & fit.year.isin(TRAIN_YEARS)].copy()
    return fit, claims_fit, cal, test, exact_abstract_exclusions


def train_text(fit, cal, text):
    vec = vectorizer()
    x = vec.fit_transform(fit[text])
    head = LogisticRegression(solver="liblinear", C=2, max_iter=300, random_state=SEED)
    head.fit(x, fit[TARGET])
    p_cal = head.predict_proba(vec.transform(cal[text]))[:, 1]
    return vec, head, p_cal


def predict(vec, head, frame, text):
    return head.predict_proba(vec.transform(frame[text]))[:, 1]


def blend(a, c, weight):
    # Blend log odds to keep the claims contribution explicit and auditable.
    return expit((1 - weight) * logit(np.clip(a, 1e-7, 1 - 1e-7))
                 + weight * logit(np.clip(c, 1e-7, 1 - 1e-7)))


def paired_week_bootstrap(test, baseline, challenger):
    week = pd.to_datetime(test.published_date).dt.to_period("W")
    actual = test[TARGET].eq(1).to_numpy()
    rows = []
    for prediction in (baseline, challenger):
        cells = pd.DataFrame({"week": week.to_numpy(), "tp": actual & prediction,
                              "fp": ~actual & prediction, "fn": actual & ~prediction})
        rows.append(cells.groupby("week").sum().to_numpy())
    n = len(rows[0])
    draws = np.random.default_rng(SEED).integers(n, size=(2000, n))
    values = []
    for cells in rows:
        tp, fp, fn = cells[draws].sum(axis=1).T
        denominator = 2 * tp + fp + fn
        values.append(np.divide(2 * tp, denominator, out=np.full(len(tp), np.nan), where=denominator > 0))
    differences = values[1] - values[0]
    differences = differences[np.isfinite(differences)]
    return {"publication_weeks": n, "draws": len(draws),
            "valid_draws": len(differences),
            "f1_difference_percentile_interval": np.quantile(differences, [.025, .975]).tolist()
            if len(differences) else None,
            "note": "Paired by publication week; does not account for residual patent-family overlap."}


def run():
    fit, claims_fit, cal, test, excluded = training_parts()
    print(f"Fitting {len(fit):,} abstracts and {len(claims_fit):,} claims; "
          f"calibrating {len(cal):,}; new test {len(test):,}", flush=True)
    av, ah, ap = train_text(fit, cal, "application_abstract")
    cv, ch, cp = train_text(claims_fit, cal, "claim_text")
    tuning = []
    for weight in WEIGHTS:
        p = blend(ap, cp, weight)
        t = operating_threshold(cal[TARGET], p)
        tuning.append(dict(weight=weight, threshold=t, **metrics(cal[TARGET], p, t)))
    tuning = pd.DataFrame(tuning).sort_values(["f1", "weight"], ascending=[False, True])
    tuning.to_csv(OUT / "calibration_grid.csv", index=False)
    best = tuning.iloc[0]
    weight, threshold = float(best.weight), float(best.threshold)
    path = OUT / "frozen_claims_benchmark.joblib"
    joblib.dump(dict(abstract_vectorizer=av, abstract_head=ah, claims_vectorizer=cv, claims_head=ch,
                     claims_weight=weight, threshold=threshold), path, compress=3)
    write_json(OUT / "freeze.json", {"sha256": fingerprint(path), "claims_weight": weight,
               "threshold": threshold, "abstract_fit_n": len(fit), "claims_fit_n": len(claims_fit),
               "claims_fit_by_year": claims_fit.groupby("year").size().to_dict(), "calibration_n": len(cal),
               "test_n": len(test), "test_exact_prior_abstract_exclusions": excluded,
               "family_overlap_cleared": False})

    a = predict(av, ah, test, "application_abstract")
    c = predict(cv, ch, test, "claim_text")
    p = blend(a, c, weight)
    old = joblib.load(V2 / "frozen_benchmark.joblib")
    old_p = Scorer(old).predict_proba(test.application_abstract)[:, 1]
    a_threshold = float(tuning.loc[tuning.weight.eq(0), "threshold"].iloc[0])
    report = {"v2_benchmark": metrics(test[TARGET], old_p, old["threshold"]),
              "refit_abstract_only": metrics(test[TARGET], a, a_threshold),
              "refit_abstract_plus_claims": metrics(test[TARGET], p, threshold),
              "claims_weight": weight, "new_sample_relative_to_v2": True,
              "family_overlap_cleared": False, "newer_predictions_replaced": False,
              "paired_week_bootstrap_vs_v2":
              paired_week_bootstrap(test, old_p >= old["threshold"], p >= threshold)}
    write_json(OUT / "new_2023_benchmark.json", report)
    test[["pgpub_id", "application_id", "published_date", TARGET]].assign(
        v2_score=old_p, abstract_score=a, claims_only_score=c, combined_score=p,
        v2_label=(old_p >= old["threshold"]).astype(int),
        abstract_label=(a >= a_threshold).astype(int),
        combined_label=(p >= threshold).astype(int),
    ).to_csv(OUT / "new_2023_paired_predictions.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=int, nargs="+")
    args = parser.parse_args()
    if args.extract:
        for year in args.extract:
            extract(year)
    else:
        run()
