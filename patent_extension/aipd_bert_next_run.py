"""Run the frozen patent encoder on the sealed benchmark, without test labels."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from aipd_extension import ROOT, TARGET, fingerprint, operating_threshold, write_json
from aipd_patent_bert_pilot import CHECKPOINT, MODEL, REVISION, Encoder

DEFAULT = ROOT / "aipd_bert_next"


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    if fingerprint(args.checkpoint / "model.safetensors") != CHECKPOINT:
        raise ValueError("Unexpected checkpoint hash")
    encoder = Encoder(args.batch_size, args.checkpoint)
    arrays, frames = [], []
    for name in ("fit", "calibration", "test"):
        path = args.data / f"{name}.csv.gz"
        frame = pd.read_csv(path, dtype={"pgpub_id": str}, keep_default_na=False)
        if frame.pgpub_id.duplicated().any() or (name == "test" and TARGET in frame):
            raise ValueError("Repeated IDs or unsealed test")
        blocks = []
        for start in range(0, len(frame), 128):
            blocks.append(encoder.encode(frame.iloc[start:start + 128]))
            if start % 512 == 0:
                print(f"{name}: {min(start + 128, len(frame))}/{len(frame)}", flush=True)
        arrays.append(np.concatenate(blocks))
        frames.append(frame)
        np.save(out / f"{name}_embeddings.npy", arrays[-1])
    x, xc, xt = arrays
    fit, cal, test = frames
    head = LogisticRegression(C=1, solver="liblinear", max_iter=300, random_state=20261113).fit(x, fit[TARGET])
    threshold = operating_threshold(cal[TARGET], head.predict_proba(xc)[:, 1])
    path = out / "frozen_head.joblib"
    joblib.dump({"head": head, "threshold": threshold}, path)
    write_json(out / "freeze.json", {"head_sha256": fingerprint(path), "threshold": threshold,
               "checkpoint_sha256": CHECKPOINT, "checkpoint_revision": REVISION,
               "gpu": encoder.torch.cuda.get_device_name(), "batch_size": args.batch_size,
               "bundle_sha256": {p.name: fingerprint(p) for p in args.data.glob("*.csv.gz")}})
    test[["pgpub_id"]].assign(neural_score=head.predict_proba(xt)[:, 1]).to_csv(
        out / "neural_test_scores.csv", index=False)
    print("Frozen neural scores saved; no test labels loaded.", flush=True)


def speed(args):
    frame = pd.read_csv(ROOT / "aipd_claims_expanded/claims_2021_joined.csv",
                        usecols=["pgpub_id", "claim_text"], dtype=str, keep_default_na=False)
    hist = pd.read_csv(ROOT / "aipd_surrogate_v2/historical_abstracts_labels.csv",
                       usecols=["pgpub_id", "application_abstract"], dtype=str, keep_default_na=False)
    sample = frame.loc[frame.claim_text.ne("")].sample(n=64, random_state=20261113)
    sample = sample.merge(hist, on="pgpub_id", validate="one_to_one")
    if fingerprint(args.checkpoint / "model.safetensors") != CHECKPOINT:
        raise ValueError("Unexpected checkpoint hash")
    enc = Encoder(4, args.checkpoint)
    enc.encode(sample.iloc[:4])
    reports = []
    for batch in (4, 8, 16, 32):
        enc.batch_size = batch
        enc.torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        try:
            enc.encode(sample)
        except enc.torch.cuda.OutOfMemoryError:
            reports.append({"batch": batch, "status": "out_of_memory"})
            enc.torch.cuda.empty_cache()
            break
        seconds = time.monotonic() - start
        reports.append({"batch": batch, "status": "ok", "docs_per_second": len(sample) / seconds,
                        "peak_torch_bytes": enc.torch.cuda.max_memory_allocated()})
        print(reports[-1], flush=True)
    write_json(DEFAULT / "local_batch_speed.json", {"gpu": enc.torch.cuda.get_device_name(),
               "historical_fit_examples_only": len(sample), "benchmarks": reports})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT / "gpu_bundle")
    parser.add_argument("--out", type=Path, default=DEFAULT / "neural_run")
    parser.add_argument("--checkpoint", type=Path, default=MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--speed-only", action="store_true")
    args = parser.parse_args()
    (speed if args.speed_only else run)(args)
