# Reproduction instructions

## Current status

The code is preserved and indexed. Not every historical script is portable: several retain original Windows paths. The publication table entry point supports FINAI_DATA_ROOT and FINAI_TABLES_OUT; the earlier patent paths and downloaded-source layout must be checked before executing other entry points. This branch is not a one-command final replication.

## Table rebuild

Python 3.11 was used with pandas, numpy, scipy, statsmodels, openpyxl (read-only verification) and joblib. Install in an isolated environment. The original working data tree must include the professor panel, detailed annual accounting CSV, standardized Item 1/document files, annual period-date mapping, occurrence-topic assignments and SIC classification rules.

Example on Windows, from this repository:

    $env:FINAI_DATA_ROOT='C:/path/to/authorised/finai-data'
    $env:FINAI_TABLES_OUT='C:/path/to/output/analysis'
    python publication_tables.py
    python publication_examples_tex.py
    python test_publication_tables.py

For workbooks, publication_tables_workbook.mjs uses the host's documented @oai/artifact-tool runtime. It is not a generic npm dependency supplied by this repository. FINAI_TABLES_OUT locates the analytical JSON and FINAI_WORKBOOK_OUT sets the workbook output folder. verify_publication_tables.py expects workbook output under FINAI_DATA_ROOT/outputs/publication_tables_20261112 in the current implementation. Consult the source before relocating outputs.

## Patent BERT benchmark

Follow patent_extension/AIPD_BERT_NEXT.md. Download the pinned checkpoint outside Git and verify the Safetensors hash. The GPU archive is a private/local generated deliverable, not in the source repository. Test labels remain with the evaluator. Do not run the evaluator before candidate scores and thresholds are frozen.

The local pilot used PyTorch 2.7.1+cu118 and Transformers 4.32.1. The runtime must match the target host's driver and Python platform. No scheduler or remote GPU reservation is assumed. Running inside tmux preserves a job through an SSH disconnect, not a host reboot.

## Checks

    python test_publication_tables.py
    python patent_extension/test_aipd_claims_benchmark.py
    python patent_extension/test_aipd_v2.py

These tests use synthetic cases. Other tests, including saved-manifest checks, need the corresponding local inputs. Tests are not evidence of classifier accuracy or validation of financial construct definitions.
