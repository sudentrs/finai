# Expanded pre-grant claims benchmark

## Design

The goal is agreement with USPTO AIPD's corrected `predict93_any_ai` flag for utility pre-grant publications. The inputs are PatentsView publication metadata, abstracts, and pre-grant claims. No firm sample, financial data, 10-K variables, or human pilot labels are read.

The [PatentsView claims snapshot](https://zenodo.org/records/15062183) contains year-specific claim rows keyed by publication ID. Historical ZIPs were checked against the archive's published MD5 digests before use. Claim text is joined by publication ID, sorted by claim sequence and capped at 12,000 characters per publication. Conflicting sequence duplicates fail ingestion rather than being silently combined. A missing claim never becomes a negative AI label.

The benchmark fits the abstract head on available 2015-2022 historical abstracts and the claims head on available 2018-2022 claims. A separate seeded quarter of eligible 2022 records is held out for selecting the blend weight and thresholds. The blend combines scores on the log-odds scale. It is an agreement score, not an independently calibrated probability of true AI invention. The claims weight grid and cutoff rule are implemented in code. The final evaluation compares the pre-existing v2 benchmark, the refitted abstract-only head and the selected blend on identical publications.

The test starts from a newly seeded 12,000-publication draw from 2023. Applications in the v2 historical manifest or the independent review manifest are ineligible, and exact normalized abstracts already present in the v2 historical data are excluded. Claims and abstracts matching the final test are excluded from fitting and calibration; fitting and calibration are also exact-text disjoint. A family crosswalk was not available, so different texts from related applications may remain. The paired bootstrap resamples publication weeks and does not establish family-robust inference.

## Scope of the results

The benchmark is about reproducing AIPD's classifications, not independent human accuracy. The comparison is conditional on matched abstracts and claims. The recent v2 predictions remain unchanged: the 2025 claims archive is still needed before comparable recent coverage can be measured and a common claims-based classifier can be considered for both 2024 and 2025. See the [USPTO pre-grant long-text dataset](https://data.uspto.gov/bulkdata/datasets/pvpgpubtxt).

## Reproduction

```powershell
python patent_extension/aipd_claims_fresh_sample.py
python patent_extension/aipd_claims_expanded.py --extract 2018 2019 2020 2021 2022 2023
python patent_extension/aipd_claims_expanded.py
````

Outputs and source fingerprints are under `aipd_claims_expanded`. The input archives are in `claims_source`. Older benchmark and production files were not overwritten.
