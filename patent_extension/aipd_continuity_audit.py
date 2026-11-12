"""Flag recorded U.S. continuity links crossing the claims benchmark splits."""
import json
import zipfile
from collections import defaultdict

import pandas as pd

from aipd_claims_benchmark import SOURCE
from aipd_claims_expanded import OUT, training_parts
from aipd_extension import TARGET, fingerprint, metrics, write_json
from aipd_v2_prepare import V2

PATH = SOURCE / "pg_us_rel_doc.tsv.zip"
TYPES = {"continuation", "continuation-in-part", "division", "us-provisional-application"}


def components(anchors):
    parent = {}

    def root(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in anchors:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    return {node: root(node) for node in parent}


def audit():
    fit, _, cal, test, _ = training_parts()
    frame = pd.concat([part[["pgpub_id", "application_id"]].assign(role=role)
                       for part, role in ((fit, "fit"), (cal, "calibration"), (test, "test"))])
    pub_to_app = frame.set_index("pgpub_id", verify_integrity=True).application_id
    edges, observed, relation_rows = set(), set(), 0
    columns = ["pgpub_id", "related_doc_number", "published_country", "related_doc_type"]
    with zipfile.ZipFile(PATH) as archive:
        member, = [name for name in archive.namelist() if name.endswith(".tsv")]
        with archive.open(member) as stream:
            for chunk in pd.read_csv(stream, sep="\t", usecols=columns, dtype=str, chunksize=150_000,
                                     keep_default_na=False):
                hit = chunk.loc[chunk.pgpub_id.isin(pub_to_app.index)]
                observed.update(hit.pgpub_id)
                hit = hit[hit.published_country.eq("US") & hit.related_doc_type.isin(TYPES)
                          & hit.related_doc_number.str.fullmatch(r"\d{8}")]
                relation_rows += len(hit)
                src = hit.pgpub_id.map(pub_to_app)
                edges.update((a, b) for a, b in zip(src, hit.related_doc_number) if a != b)
    comp = components(edges)
    frame["group"] = frame.application_id.map(lambda app: comp.get(app, app))
    roles = frame.groupby("group").role.agg(lambda entries: frozenset(entries))
    cross = {key for key, value in roles.items() if len(value) > 1}
    test_cross = {key for key in cross if "test" in roles[key]}
    flagged = frame[frame.group.isin(cross)].copy()
    flagged.to_csv(OUT / "known_us_continuity_cross_split.csv", index=False)
    by_role = defaultdict(int)
    for role, count in flagged.groupby("role").size().items():
        by_role[role] = int(count)
    report = {"source_sha256": fingerprint(PATH), "source_publication_vintage": "through 2024",
              "screened_publications": len(frame), "publications_with_any_recorded_relationship": len(observed),
              "qualifying_link_rows": relation_rows, "unique_edges": len(edges),
              "cross_split_components": len(cross), "test_cross_split_components": len(test_cross),
              "test_publications_in_cross_split_components":
                  int((frame.role.eq("test") & frame.group.isin(test_cross)).sum()),
              "flagged_publications_by_role": dict(by_role), "family_overlap_cleared": False,
              "scope": "U.S. 8-digit continuation, division and provisional anchors attached to selected first publications; "
                       "does not follow unsampled publications or establish DOCDB families"}
    predictions = pd.read_csv(OUT / "new_2023_paired_predictions.csv", dtype={"pgpub_id": str})
    exclude = set(frame.loc[frame.role.eq("test") & frame.group.isin(test_cross), "pgpub_id"])
    sensitivity = predictions.loc[~predictions.pgpub_id.isin(exclude)]
    cutoff = json.loads((OUT / "freeze.json").read_text())["threshold"]
    cutoff_v2 = json.loads((V2 / "benchmark_freeze.json").read_text())["threshold"]
    report["sensitivity_excluding_linked_test_publications"] = {
        "claims": metrics(sensitivity[TARGET], sensitivity.combined_score, cutoff),
        "v2": metrics(sensitivity[TARGET], sensitivity.v2_score, cutoff_v2),
        "note": "Existing models and thresholds unchanged; this is not a new family-disjoint fit."
    }
    write_json(OUT / "known_us_continuity_audit.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    audit()
