# Larger historical BERT benchmark

This package is for a frozen-embedding experiment, not for retraining on recent AIPD-unlabeled patents or replacing 2024-2025 predictions. No job has been submitted to Rick, Morty, or a cloud service.

The test was newly drawn from 2023, excluding the prior historical and review applications and exact text already used. Known U.S. continuation, division and provisional links and exact text hashes were grouped before assigning roles. The graph uses recorded links attached to publications in the candidate and earlier benchmark frames, plus shared relationship anchors; it is not a complete DOCDB or international family map. The test has priority over calibration and fitting, and groups in calibration are removed from fitting. Test labels are stored separately from the GPU bundle.

Both the neural model and a refitted full-data sparse claims model use the same test and calibration cohorts. The neural pilot has a smaller fitting sample for compute reasons; this is a comparison of the actual candidate configurations, not a pure architecture ablation. The neural encoder sees one abstract window and up to two nonoverlapping claim windows, each capped at 128 tokens; it does not see the full claims. Fit a logistic head on frozen embeddings and choose the threshold on calibration only. No hard-case weighting or fine-tuning is used in this first larger run.

## Execute

The scripts in the GPU package can run together from the extracted directory with a CUDA-enabled PyTorch environment. Install PyTorch for the host's driver, plus Transformers (tested with 4.32.1), Safetensors, pandas, numpy, SciPy, scikit-learn, and joblib. Use the same dependency versions as the local run where practical. The checkpoint is the [BERT-for-Patents conversion](https://huggingface.co/anferico/bert-for-patents), revision `148e559789f59c2660cb7e7aaa0812041660c607`; download `config.json`, `vocab.txt` and `model.safetensors` into a local directory. The runner verifies the checkpoint and records the input hashes.

```bash
python -u aipd_bert_next_run.py --data gpu_bundle --checkpoint /path/to/checkpoint --out neural_run --batch-size 8
```

The runner never reads the test labels. Bring back the complete `neural_run` directory. Once both scores and thresholds are frozen, run the evaluator in the project workspace:

```powershell
python patent_extension/aipd_bert_next_evaluate.py --neural path/to/neural_run
```

The evaluator compares precision, recall, F1, average precision and the confusion matrix; it also reports a paired publication-week bootstrap interval. This interval is not family-robust. Do not reuse this test for tuning. The 2025 pre-grant claims source is still separate from this historical test.
