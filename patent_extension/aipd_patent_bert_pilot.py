"""Memory-bounded, frozen BERT-for-Patents pilot against AIPD labels."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from aipd_claims_benchmark import vectorizer
from aipd_claims_expanded import OUT as PREVIOUS, training_parts
from aipd_extension import ROOT, TARGET, fingerprint, metrics, operating_threshold, write_json

MODEL = ROOT / "bert_for_patents_checkpoint"
OUT = ROOT / "aipd_bert_pilot"
REVISION = "148e559789f59c2660cb7e7aaa0812041660c607"
CHECKPOINT = "06b84587f013cc325b939716eea0e39fa33a1aaa93a0ac83379d65f12c4ea278"
SEED = 20261112
MAX_TOKENS = 128


def claim_windows(tokens, width):
    if len(tokens) <= width:
        return [tokens, []]
    # Keep a second non-overlapping view from later in the capped source text.
    offset = max(width, (len(tokens) - width) // 2)
    return [tokens[:width], tokens[offset:offset + width]]


class Encoder:
    def __init__(self, batch_size, checkpoint):
        import torch
        from transformers import BertModel, BertTokenizerFast

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; this pilot will not silently switch to CPU")
        self.torch = torch
        self.tokenizer = BertTokenizerFast.from_pretrained(checkpoint, local_files_only=True,
                                                         do_lower_case=True, model_max_length=10**6)
        self.model = BertModel.from_pretrained(
            checkpoint, local_files_only=True, use_safetensors=True,
            add_pooling_layer=False, torch_dtype=torch.float16
        ).eval().to("cuda")
        self.batch_size = batch_size
        self.width = MAX_TOKENS - 2

    def encode(self, frame):
        tok, torch = self.tokenizer, self.torch
        embeddings = np.zeros((len(frame), 3, self.model.config.hidden_size), dtype=np.float32)
        entries = []
        for i, (abstract, claims) in enumerate(zip(frame.application_abstract, frame.claim_text)):
            abstract_tokens = tok.encode(abstract, add_special_tokens=False, truncation=True, max_length=self.width)
            windows = [abstract_tokens, *claim_windows(tok.encode(claims, add_special_tokens=False), self.width)]
            for j, tokens in enumerate(windows):
                if tokens:
                    entries.append((i, j, [tok.cls_token_id, *tokens, tok.sep_token_id]))
        with torch.inference_mode():
            for start in range(0, len(entries), self.batch_size):
                batch = entries[start:start + self.batch_size]
                model_input = tok.pad({"input_ids": [x[2] for x in batch]},
                                      padding=True, return_tensors="pt").to("cuda")
                mask = model_input.attention_mask
                hidden = self.model(**model_input, return_dict=True).last_hidden_state
                pooled = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
                for (i, j, _), row in zip(batch, pooled.cpu().numpy()):
                    embeddings[i, j] = row
        if not np.isfinite(embeddings).all() or not np.linalg.norm(embeddings[:, 0], axis=1).all():
            raise ValueError("Invalid contextual embeddings")
        a, c1, c2 = np.moveaxis(embeddings, 1, 0)
        divisor = 1 + np.linalg.norm(c2, axis=1, keepdims=True).astype(bool).astype(float)
        c = ((c1 + c2) / divisor).astype(np.float32)
        return np.hstack([normalize(a), normalize(c)])


def matched_sparse(fit, cal, test):
    blocks = [[], [], []]
    for column in ("application_abstract", "claim_text"):
        vec = vectorizer()
        blocks[0].append(vec.fit_transform(fit[column]))
        blocks[1].append(vec.transform(cal[column]))
        blocks[2].append(vec.transform(test[column]))
    x, xc, xt = [hstack(part, format="csr") for part in blocks]
    head = LogisticRegression(C=2, solver="liblinear", max_iter=300, random_state=SEED)
    head.fit(x, fit[TARGET])
    threshold = operating_threshold(cal[TARGET], head.predict_proba(xc)[:, 1])
    return metrics(test[TARGET], head.predict_proba(xt)[:, 1], threshold)


def run(args):
    checkpoint = args.checkpoint
    if fingerprint(checkpoint / "model.safetensors") != CHECKPOINT:
        raise ValueError("Unexpected BERT checkpoint fingerprint")
    args.out.mkdir(exist_ok=False)
    import torch
    encoder = Encoder(args.batch_size, checkpoint)
    _, fit, cal, test, _ = training_parts()
    smoke = fit.sample(n=16, random_state=SEED)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    sample = encoder.encode(smoke)
    seconds = time.monotonic() - started
    speed = len(smoke) / seconds
    budget_n = min(args.max_docs, int(args.budget_seconds * speed * .8))
    audit = {"gpu": torch.cuda.get_device_name(), "cuda_runtime": torch.version.cuda,
             "torch_version": torch.__version__, "checkpoint_revision": REVISION,
             "checkpoint_sha256": CHECKPOINT, "precision": "float16 inference; float32 pooled",
             "token_cap": MAX_TOKENS, "windows": "one abstract and up to two non-overlapping claims views",
             "smoke_docs": len(smoke), "seconds": seconds, "docs_per_second": speed,
             "embedding_dim": sample.shape[1], "estimated_pilot_documents": budget_n,
             "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
             "family_overlap_cleared": False, "test_previously_examined": True}
    write_json(args.out / "runtime_smoke.json", audit)
    print(json.dumps(audit, indent=2), flush=True)
    if args.smoke_only or budget_n < 240:
        return
    remaining = fit[~fit.pgpub_id.isin(smoke.pgpub_id)]
    counts = [int(budget_n * .6), int(budget_n * .18)]
    counts.append(budget_n - sum(counts))
    frames = [part.sample(n=n, random_state=SEED + 1 + i).reset_index(drop=True)
              for i, (part, n) in enumerate(zip((remaining, cal, test), counts))]
    if any(part[TARGET].nunique() != 2 for part in frames):
        raise ValueError("Pilot draw did not contain both classes in each split")
    arrays = []
    for name, part in zip(("fit", "calibration", "test"), frames):
        blocks = []
        for start in range(0, len(part), 32):
            blocks.append(encoder.encode(part.iloc[start:start + 32]))
            if start % 128 == 0:
                print(f"{name} embeddings: {min(start+32, len(part))}/{len(part)}", flush=True)
        array = np.concatenate(blocks)
        np.save(args.out / f"{name}_embeddings.npy", array)
        part[["pgpub_id", "published_date", TARGET]].to_csv(args.out / f"{name}_manifest.csv", index=False)
        arrays.append(array)
    f, c, t = frames
    x, xc, xt = arrays
    head = LogisticRegression(C=1, solver="liblinear", max_iter=300, random_state=SEED).fit(x, f[TARGET])
    threshold = operating_threshold(c[TARGET], head.predict_proba(xc)[:, 1])
    joblib.dump(dict(head=head, threshold=threshold, encoder_revision=REVISION), args.out / "pilot_head.joblib")
    probabilities = head.predict_proba(xt)[:, 1]
    prior = pd.read_csv(PREVIOUS / "new_2023_paired_predictions.csv", dtype={"pgpub_id": str})
    prior = prior.set_index("pgpub_id", verify_integrity=True).loc[t.pgpub_id]
    cutoff = json.loads((PREVIOUS / "freeze.json").read_text())["threshold"]
    results = {"fit_n": len(f), "calibration_n": len(c), "test_n": len(t),
               "frozen_patent_bert": metrics(t[TARGET], probabilities, threshold),
               "matched_training_size_tfidf": matched_sparse(f, c, t),
               "full_training_claims_reference": metrics(t[TARGET], prior.combined_score, cutoff),
               "test_previously_examined": True, "family_overlap_cleared": False,
               "recent_predictions_replaced": False}
    write_json(args.out / "pilot_metrics.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODEL)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--budget-seconds", type=int, default=300)
    parser.add_argument("--max-docs", type=int, default=1200)
    parser.add_argument("--smoke-only", action="store_true")
    run(parser.parse_args())
