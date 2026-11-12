# Frozen BERT-for-Patents GPU pilot

The pilot is a compute and model feasibility check, not confirmatory evidence. It uses the claims-eligible, previously examined historical sample, with no financial, issuer or disclosure inputs. The same random subsample is used for the frozen encoder and matched-size TF-IDF. The full-data sparse claims model is reported separately as the incumbent comparison.

The [public BERT-for-Patents conversion](https://huggingface.co/anferico/bert-for-patents) is pinned to revision `148e559789f59c2660cb7e7aaa0812041660c607`, with a verified Safetensors hash. CUDA PyTorch was installed in `neural_runtime`, an isolated virtual environment, without replacing the system Python runtime. The model ran frozen in float16, with float32 attention-masked mean pooling, one abstract window and up to two nonoverlapping claim windows, each at most 128 tokens. The two claim vectors are averaged, then concatenated with the normalized abstract vector. Only the logistic head is fitted; the cutoff uses calibration records, not test labels.

The local MX550 measured about 1.57 documents per second in the smoke test, with about 676 MiB of GPU allocations reported by PyTorch (not total driver memory). The pilot fitted 226 records, calibrated on 67 and evaluated on 84. F1 was 83.3% for the frozen embeddings, 70.0% for TF-IDF with the same training size, and 87.8% for the full-data sparse claims model on those exact 84 records. There were only 23 reference positives. Do not compare the pilot scores to the incumbent's headline 80.9% on the different, much larger benchmark.

The structured [PatentsView relationship archive](https://zenodo.org/records/15783125) supplies a partial screen for U.S. continuation, continuation-in-part, division and provisional links. This is not a complete patent-family map. Known cross-split links found in the large historical benchmark are reported separately. For a larger trial, put all known connected applications in one role before fitting, document the links' coverage, and seek a broader family crosswalk. Keep a previously unused, representative test; mine errors on development data only. Fine-tuning, a confirmatory trial and recent neural scoring were not performed.

## Reproduce

```powershell
$env:HF_HOME="$PWD/patent_extension/neural_runtime/hf_cache"
& 'patent_extension/neural_runtime/Scripts/python.exe' -u patent_extension/aipd_patent_bert_pilot.py --out patent_extension/aipd_bert_pilot_repeat
python patent_extension/aipd_continuity_audit.py
python patent_extension/aipd_recent_claims_coverage.py
```

The inference budget controls sample size from measured throughput rather than from labels or performance. The local GPU is suitable for a smoke test and small pilot. A larger frozen-embedding run, fine-tuning and scoring the full recent patent universe are separate compute decisions. The 2025 pre-grant claims ZIP is still needed from the [USPTO Open Data Portal](https://data.uspto.gov/bulkdata/datasets/pvpgpubtxt); no 2024-2025 predictions have been replaced.
