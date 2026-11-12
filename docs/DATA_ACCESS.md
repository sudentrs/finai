# Data access and collaborator package

## Not supplied through GitHub

- Professor's ITEM1_reg.dta monthly panel: use the research team's authorised copy.
- CRSP/Compustat/WRDS downloads: obtain under the applicable institutional licence. A derived firm-level file is not automatically free to redistribute.
- Refinitiv transcripts and call-level results: use authorised access; only old aggregate correlation summaries are currently located.
- SEC processed sentence corpus and standardized linked filing files: large local derived inputs, with financial links requiring separate access.
- Patent abstracts, claims and AIPD: public inputs downloaded separately and fingerprinted by the relevant audit scripts.
- BERT weights and Python/GPU environments: downloaded/installed separately, never versioned as project source.

The machine-readable data_inventory.json lists local-relative paths, roles and availability. It does not contain the datasets.

## Private handoff

An authorised colleague should receive the current two table workbooks, all table CSV/TeX and audit outputs, and the filing-level analysis CSVs if their data permissions allow. Include the methods/review memo and exact input versions. Do not put these files on a public branch merely because it is called a collaborator branch. All branches share the repository's access controls.

For a referee release, request restricted-data access through the publisher or a secure institutional channel. If inputs cannot be shared, provide download/query specifications, schemas, dates and fingerprints. Do not fabricate substitute data and label it real.

## Public sources

- [SEC EDGAR](https://www.sec.gov/edgar)
- [USPTO AIPD](https://www.uspto.gov/ip-policy/economic-research/research-datasets/artificial-intelligence-patent-dataset)
- [USPTO PatentsView](https://www.uspto.gov/ip-policy/economic-research/patentsview)
- [Historical claims snapshot](https://zenodo.org/records/15062183)
- [Historical relationship snapshot](https://zenodo.org/records/15783125)
- [Pinned patent encoder](https://huggingface.co/anferico/bert-for-patents)

The AIPD label is a teacher-model prediction, not human ground truth. Publication-year coverage does not establish complete filing cohorts. The 2025 claims source is still required for a comparable claims-based recent extension.
