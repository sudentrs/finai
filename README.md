# FINAI research reproducibility

This branch organizes the current disclosure-measurement, validation and patent-classification source code. It is a **referee-preparation branch, not a final publication replication release**. The empirical samples, some specifications and the patent extension remain under review.

## Start here

1. [Pipeline map](docs/PIPELINE_MAP.md): which scripts belong to each task and which are exploratory.
2. [Data and access](docs/DATA_ACCESS.md): inputs, licensing and the collaborator handoff.
3. [Reproduction](docs/REPRODUCE.md): environment, entry points, tests and limits.
4. [Release checklist](docs/REFEREE_RELEASE_CHECKLIST.md): verified steps and unresolved issues.
5. [Meeting decisions](docs/MEETING_DECISIONS.md): the two-sample rule and remaining table/figure work.

## Project areas

| Area | Location |
| --- | --- |
| SEC extraction and filing exports | Root processing and export scripts |
| AI topic construction and occurrence aggregation | Root topic and mutually-exclusive scripts |
| Firm-characteristic and validation tables | Root analysis scripts; publication_tables.py is the November draft rebuild |
| Patent extension | patent_extension/ |
| Independent patent-review protocol | patent_extension/independent_validation_v2/ |
| BERT historical benchmark | patent_extension/aipd_bert_next_*.py |
| Earlier collection notebooks | legacy_snapshot/; newly copied notebooks have outputs removed |

Existing root paths are retained because many historical scripts resolve data relative to their own directory. Moving them into new folders would silently change those paths. The documentation provides the organization without implying all historical scripts have been refactored into one runnable pipeline.

## Two sample versions

Keep **professor-rule sample** and **broader author sample** separate for every table or figure. The current table build reconstructs the professor sample from eligible monthly observations with price **>= $5**, following the written template. The broader version has no added price screen. Do not silently replace these with a third sample or switch to a strict > $5 rule.

## Data and results are separate

No new WRDS/Compustat/CRSP extract, professor panel, Refinitiv transcripts, fitted weights, raw patent archive, virtual environment or new firm-level output is published in this branch. Access to this branch does not grant permission to redistribute licensed data. Authorised collaborators need the separate private data/output package.

The source inventory contains hashes of the newly copied files. Notebook outputs are removed and credential-like literals use environment variables. This source-only branch is published from a root commit with no historical parents, so it does not inherit the original branch's notebook history. The original main branch is unchanged. Any real credential previously committed there still needs review and rotation.

The existing MIT license covers project code where applicable. Third-party papers, patent-model weights, external repositories and data retain their own terms.
