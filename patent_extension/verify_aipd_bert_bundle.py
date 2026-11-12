"""Verify the sealed GPU benchmark without opening curator labels."""
import hashlib
import json
import zipfile

import pandas as pd

from aipd_bert_next_prepare import OUT
from aipd_extension import TARGET, fingerprint


def verify():
    audit = json.loads((OUT / "split_audit.json").read_text())
    manifest = pd.read_csv(OUT / "split_manifest.csv", dtype=str)
    if manifest.pgpub_id.duplicated().any() or manifest.groupby("known_group").role.nunique().gt(1).any():
        raise ValueError("Duplicate publication or known relationship crossing splits")
    expected = {"fit": audit["gpu_fit"], "calibration": audit["calibration"], "test": audit["test"]}
    with zipfile.ZipFile(OUT / "sealed_gpu_benchmark.zip") as archive:
        if archive.testzip() is not None or any("CURATOR" in member for member in archive.namelist()):
            raise ValueError("Corrupt archive or curator material in package")
        for role, n in expected.items():
            path = OUT / "gpu_bundle" / f"{role}.csv.gz"
            expected_hash = audit["bundle_sha256"][path.name]
            if (fingerprint(path) != expected_hash or
                    hashlib.sha256(archive.read(f"gpu_bundle/{path.name}")).hexdigest() != expected_hash):
                raise ValueError(f"Unexpected input hash: {path.name}")
            data = pd.read_csv(path, dtype=str)
            if (len(data) != n or data.pgpub_id.duplicated().any() or
                    not set(data.pgpub_id).issubset(manifest.loc[manifest.role.eq(role), "pgpub_id"]) or
                    (TARGET in data) != (role != "test")):
                raise ValueError(f"Incorrect membership or label separation: {role}")
    print("Verified bundle hashes, role membership, record counts and test-label separation.")


if __name__ == "__main__":
    verify()
