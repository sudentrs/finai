"""Test whether AIPD's continuous component scores improve the claims surrogate."""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import Ridge

from aipd_claims_expanded import (OUT as PRIOR, blend, paired_week_bootstrap,
                                  training_parts)
from aipd_extension import (AIPD, CHUNK, COMPONENTS, ROOT, TARGET, fingerprint, metrics,
                            operating_threshold, write_json)

OUT = ROOT / "aipd_claims_soft_teacher"


def read_teacher_scores(ids):
    score_fields = [f"ai_score_{component}" for component in COMPONENTS]
    parts = []
    for chunk in pd.read_csv(AIPD, usecols=["doc_id", "flag_patent", TARGET, *score_fields],
                             dtype={"doc_id": str}, chunksize=CHUNK):
        hit = chunk.loc[chunk.flag_patent.eq(0) & chunk.doc_id.isin(ids)].copy()
        if len(hit):
            scores = hit[score_fields]
            if not (scores.ge(0) & scores.le(1)).all().all():
                raise ValueError("Invalid teacher component scores")
            hit["teacher_max"] = scores.max(axis=1)
            if not hit[TARGET].eq(hit.teacher_max.ge(.93)).all():
                raise ValueError("Official flag differs from corrected maximum-score rule")
            parts.append(hit[["doc_id", TARGET, "teacher_max"]])
    scores = pd.concat(parts, ignore_index=True).set_index("doc_id", verify_integrity=True)
    if set(scores.index) != ids:
        raise ValueError("Not every fitting publication joined to an AIPD score")
    return scores


def matrices(bundle, frame):
    return (bundle["abstract_vectorizer"].transform(frame.application_abstract),
            bundle["claims_vectorizer"].transform(frame.claim_text))


def hard_score(bundle, x):
    xa, xc = x
    return blend(bundle["abstract_head"].predict_proba(xa)[:, 1],
                 bundle["claims_head"].predict_proba(xc)[:, 1], bundle["claims_weight"])


def soft_target(p):
    # Center the teacher's component maximum at its corrected decision boundary.
    return np.clip(logit(np.clip(p, 1e-6, 1 - 1e-6)) - logit(.93), -8, 5)


def soft_score(a, c, matrices_pair):
    xa, xc = matrices_pair
    return expit(.7 * a.predict(xa) + .3 * c.predict(xc))


def main():
    OUT.mkdir(exist_ok=False)
    fit, claims_fit, cal, test, _ = training_parts()
    path = PRIOR / "frozen_claims_benchmark.joblib"
    prior = json.loads((PRIOR / "freeze.json").read_text())
    if fingerprint(path) != prior["sha256"]:
        raise ValueError("Prior frozen benchmark does not match its fingerprint")
    if (len(fit), len(claims_fit), len(cal), len(test)) != (
            prior["abstract_fit_n"], prior["claims_fit_n"], prior["calibration_n"], prior["test_n"]):
        raise ValueError("The historical split changed")
    bundle = joblib.load(path)
    print(f"Joining teacher scores for {len(fit):,} fitting publications", flush=True)
    teacher = read_teacher_scores(set(fit.pgpub_id))
    if not np.array_equal(teacher.loc[fit.pgpub_id, TARGET].to_numpy(), fit[TARGET].to_numpy()):
        raise ValueError("Teacher labels differ from historical fit")
    print("Fitting continuous-score abstract and claims heads", flush=True)
    a = Ridge(alpha=2, solver="lsqr", tol=1e-4).fit(
        bundle["abstract_vectorizer"].transform(fit.application_abstract),
        soft_target(teacher.loc[fit.pgpub_id, "teacher_max"].to_numpy()))
    c = Ridge(alpha=2, solver="lsqr", tol=1e-4).fit(
        bundle["claims_vectorizer"].transform(claims_fit.claim_text),
        soft_target(teacher.loc[claims_fit.pgpub_id, "teacher_max"].to_numpy()))
    x = matrices(bundle, cal)
    hp = hard_score(bundle, x)
    sp = soft_score(a, c, x)
    del x
    options = []
    for weight in (0, .15, .30, .45, .60, .75, 1):
        p = blend(hp, sp, weight)
        cutoff = operating_threshold(cal[TARGET], p)
        options.append(dict(soft_weight=weight, threshold=cutoff, **metrics(cal[TARGET], p, cutoff)))
    grid = pd.DataFrame(options).sort_values(["f1", "soft_weight"], ascending=[False, True])
    grid.to_csv(OUT / "calibration_grid.csv", index=False)
    weight, threshold = map(float, grid.iloc[0][["soft_weight", "threshold"]])
    model_path = OUT / "frozen_challenger.joblib"
    joblib.dump(dict(hard_bundle=bundle, soft_abstract=a, soft_claims=c,
                     soft_weight=weight, threshold=threshold), model_path, compress=3)
    write_json(OUT / "freeze.json", {"sha256": fingerprint(model_path), "soft_weight": weight,
               "threshold": threshold, "fit_n": len(fit), "claims_fit_n": len(claims_fit),
               "calibration_n": len(cal), "fit_target": "centered clipped logit of max AIPD component score",
               "test_reused": True, "family_clearance": False})
    x = matrices(bundle, test)
    hp, sp = hard_score(bundle, x), soft_score(a, c, x)
    p = blend(hp, sp, weight)
    original = pd.read_csv(PRIOR / "new_2023_paired_predictions.csv", dtype={"pgpub_id": str})
    original = original.set_index("pgpub_id", verify_integrity=True)
    if set(original.index) != set(test.pgpub_id) or not np.array_equal(
            hp >= bundle["threshold"], original.loc[test.pgpub_id, "combined_label"].eq(1).to_numpy()):
        raise ValueError("The reference benchmark did not reconcile")
    base = metrics(test[TARGET], hp, bundle["threshold"])
    new = metrics(test[TARGET], p, threshold)
    report = {"hard_label_claims_baseline": base, "continuous_score_challenger": new,
              "soft_weight": weight, "test_reused_exploratory": True, "recent_predictions_replaced": False,
              "paired_week_bootstrap": paired_week_bootstrap(
                  test, hp >= bundle["threshold"], p >= threshold)}
    write_json(OUT / "reused_2023_diagnostics.json", report)
    test[["pgpub_id", "published_date", TARGET]].assign(hard_score=hp, soft_score=sp,
        challenger_score=p, challenger_label=(p >= threshold).astype(int)).to_csv(
            OUT / "reused_2023_predictions.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
