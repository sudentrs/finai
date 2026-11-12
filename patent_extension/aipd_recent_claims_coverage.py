"""Audit recent claims availability without rescoring or filling in missing labels."""
import zipfile

import pandas as pd

from aipd_claims_benchmark import SOURCE
from aipd_extension import ROOT, fingerprint, write_json
from aipd_v2_prepare import V2


def main():
    predictions = pd.read_csv(V2 / "publications_2024_2025.csv",
                              usecols=["pgpub_id", "published_date"], dtype=str)
    results = []
    for year in (2024, 2025):
        cohort = predictions.loc[predictions.published_date.str.startswith(str(year)), "pgpub_id"]
        if cohort.duplicated().any():
            raise ValueError("Repeated recent publication ID")
        ids = set(cohort)
        path = SOURCE / f"pg_claims_{year}.tsv.zip"
        result = {"publication_year": year, "eligible_n": len(ids)}
        if not path.is_file():
            results.append(dict(result, status="source_missing", usable_claims_n=None, coverage=None))
            continue
        seen, usable = set(), set()
        with zipfile.ZipFile(path) as archive:
            member, = [name for name in archive.namelist() if name.endswith(".tsv")]
            with archive.open(member) as stream:
                for chunk in pd.read_csv(stream, sep="\t", dtype=str, chunksize=100_000,
                                         keep_default_na=False, usecols=["pgpub_id", "claim_text"]):
                    selected = chunk.loc[chunk.pgpub_id.isin(ids)]
                    seen.update(selected.pgpub_id)
                    usable.update(selected.loc[selected.claim_text.str.contains(r"\w", regex=True), "pgpub_id"])
        results.append(dict(result, status="checked", joined_n=len(seen), usable_claims_n=len(usable),
                            coverage=len(usable) / len(ids), archive_sha256=fingerprint(path)))
    write_json(ROOT / "recent_pregrant_claims_coverage.json", results)
    print(results, flush=True)


if __name__ == "__main__":
    main()
