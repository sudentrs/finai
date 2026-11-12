"""Build v2 patent-only text and labels, with a previously unused 2023 benchmark."""
import argparse
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from aipd_extension import (AIPD, ABSTRACTS, CHUNK, COMPONENTS, FRAME, REVIEW, ROOT,
                            TARGET, fingerprint, normalized_hash, write_json)

V1 = ROOT / "aipd_surrogate_v1"
V2 = ROOT / "aipd_surrogate_v2"
FLAGS = [f"predict93_{component}" for component in COMPONENTS]
KEYS = ["application_id", "pgpub_id", "published_date", "filing_date"]
SEED = 20261110


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    old = pd.read_csv(V1 / "historical_split_manifest.csv", dtype=str)[KEYS]
    reserved = pd.read_csv(REVIEW, usecols=["application_id"], dtype=str)
    with sqlite3.connect(FRAME.as_uri() + "?mode=ro", uri=True) as db:
        latest = pd.read_sql_query(
            """SELECT application_id, pgpub_id, published_date, filing_date FROM applications
               WHERE published_date BETWEEN '2023-01-01' AND '2023-12-31'
               ORDER BY rowid""", db
        )
    excluded = set(old.application_id) | set(reserved.application_id)
    pool = latest[~latest.application_id.isin(excluded)]
    fresh = pool.sample(n=15000, random_state=SEED)
    frame = pd.concat([old.assign(fresh_benchmark=False), fresh.assign(fresh_benchmark=True)])
    ids = set(frame.pgpub_id)
    print(f"Collecting official labels for {len(frame):,} historical documents", flush=True)
    hits = []
    for chunk in pd.read_csv(AIPD, dtype=str, chunksize=CHUNK,
                             usecols=["doc_id", "flag_patent", "appl_id", "pub_dt", TARGET, *FLAGS]):
        hits.append(chunk[chunk.flag_patent.eq("0") & chunk.doc_id.isin(ids)])
    frame = frame.merge(pd.concat(hits), left_on="pgpub_id", right_on="doc_id", how="left",
                        validate="one_to_one")
    valid = frame.application_id.eq(frame.appl_id) & frame.published_date.eq(frame.pub_dt)
    valid &= frame[[TARGET, *FLAGS]].isin(["0", "1"]).all(axis=1)
    join = frame.assign(matched=valid).groupby("fresh_benchmark").matched.agg(["size", "sum"])
    frame = frame.loc[valid, KEYS + ["fresh_benchmark", TARGET, *FLAGS]].copy()
    ids = set(frame.pgpub_id)
    print(f"Official matches: {len(frame):,}; scanning abstract archive", flush=True)
    text = []
    with zipfile.ZipFile(ABSTRACTS) as archive:
        member, = [name for name in archive.namelist() if name.endswith(".tsv")]
        with archive.open(member) as stream:
            for i, chunk in enumerate(pd.read_csv(stream, sep="\t", dtype=str, chunksize=CHUNK,
                                                  keep_default_na=False), 1):
                text.append(chunk[chunk.pgpub_id.isin(ids)])
                if i % 20 == 0:
                    print(f"Scanned {i * CHUNK:,} abstracts", flush=True)
    frame = frame.merge(pd.concat(text), on="pgpub_id", how="left", validate="one_to_one")
    frame["hash"] = frame.application_abstract.fillna("").map(normalized_hash)
    frame["year"] = frame.published_date.str[:4].astype(int)
    frame.to_csv(out / "historical_abstracts_labels.csv", index=False)
    write_json(out / "data_audit.json", {
        "seed": SEED, "join": join.reset_index().to_dict("records"),
        "empty_or_missing_abstract": int(frame.hash.eq("").sum()),
        "fresh_benchmark_draw": 15000,
        "fresh_exclusions": "v1 matched historical manifest plus all independent review sample IDs",
        "claims": "not available locally for comparable historical and deployment coverage",
        "family": "not cleared; exact abstract and application separation only",
        "source_sha256": {str(path): fingerprint(path) for path in (FRAME, REVIEW, AIPD, ABSTRACTS)},
    })
    print(f"Prepared {len(frame):,} historical documents", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=V2)
    prepare(parser.parse_args().out)
