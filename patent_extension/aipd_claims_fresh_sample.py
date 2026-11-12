"""Draw an unused 2023 publication sample for the expanded claims comparison."""
import argparse
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

from aipd_extension import AIPD, ABSTRACTS, CHUNK, FRAME, REVIEW, ROOT, TARGET, fingerprint, normalized_hash, write_json
from aipd_v2_prepare import KEYS, V2

OUT = ROOT / "aipd_claims_expanded"
SEED = 20261111


def prepare(out=OUT, count=12000, seed=SEED, extra_exclusions=()):
    out.mkdir(exist_ok=False)
    prior = pd.read_csv(V2 / "historical_abstracts_labels.csv",
                        usecols=["application_id"], dtype=str)
    reserved = pd.read_csv(REVIEW, usecols=["application_id"], dtype=str)
    excluded = set(prior.application_id) | set(reserved.application_id)
    for path in extra_exclusions:
        excluded.update(pd.read_csv(path, usecols=["application_id"], dtype=str).application_id)
    with sqlite3.connect(FRAME.as_uri() + "?mode=ro", uri=True) as db:
        eligible = pd.read_sql_query(
            """SELECT application_id, pgpub_id, published_date, filing_date FROM applications
               WHERE published_date BETWEEN '2023-01-01' AND '2023-12-31'
               ORDER BY rowid""", db)
    draw = eligible.loc[~eligible.application_id.isin(excluded)].sample(n=count, random_state=seed)
    ids = set(draw.pgpub_id)
    official = []
    cols = ["doc_id", "flag_patent", "appl_id", "pub_dt", TARGET]
    print(f"Drew {len(draw):,} new publications; joining AIPD", flush=True)
    for chunk in pd.read_csv(AIPD, dtype=str, chunksize=CHUNK, usecols=cols):
        official.append(chunk.loc[chunk.flag_patent.eq("0") & chunk.doc_id.isin(ids)])
    draw = draw.merge(pd.concat(official), left_on="pgpub_id", right_on="doc_id", how="left",
                      validate="one_to_one")
    valid = draw.application_id.eq(draw.appl_id) & draw.published_date.eq(draw.pub_dt)
    valid &= draw[TARGET].isin(["0", "1"])
    draw = draw.loc[valid, KEYS + [TARGET]]
    print(f"{len(draw):,} official matches; reading abstracts", flush=True)
    ids = set(draw.pgpub_id)
    parts = []
    with zipfile.ZipFile(ABSTRACTS) as archive:
        member, = [name for name in archive.namelist() if name.endswith(".tsv")]
        with archive.open(member) as stream:
            for i, chunk in enumerate(pd.read_csv(stream, sep="\t", dtype=str, chunksize=CHUNK,
                                                  keep_default_na=False), 1):
                parts.append(chunk.loc[chunk.pgpub_id.isin(ids)])
                if i % 20 == 0:
                    print(f"Scanned {i * CHUNK:,} abstracts", flush=True)
    merged = draw.merge(pd.concat(parts), on="pgpub_id", how="left", validate="one_to_one")
    merged["hash"] = merged.application_abstract.fillna("").map(normalized_hash)
    merged["year"] = 2023
    merged["fresh_benchmark"] = True
    merged.to_csv(out / "new_2023_frame.csv", index=False)
    write_json(out / "draw_audit.json", {"seed": seed, "draw_n": count,
               "official_matches": len(draw), "usable_abstracts": int(merged.hash.ne("").sum()),
               "previous_v2_and_review_application_overlap": len(set(merged.application_id) & excluded),
               "note": "New relative to all listed exclusion files. Family overlap not cleared.",
               "extra_exclusions": {str(path): fingerprint(path) for path in extra_exclusions},
               "input_sha256": {str(path): fingerprint(path) for path in (AIPD, ABSTRACTS, FRAME, REVIEW)}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--count", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    args = parser.parse_args()
    prepare(args.out, args.count, args.seed, args.exclude)
