# Audit of the earlier prototype

This audit distinguishes established facts from unresolved independence claims. It does not erase prior runs or relabel exploratory results as confirmatory.

| Question | Finding | Consequence |
|---|---|---|
| Did training or threshold selection read the professor's financial panel? | No. `abstract_model.py` used the AIPat research corpus and historical benchmark text/labels for its documented checks and evaluation. | No evidence of direct financial-target leakage in the fitted prototype. |
| Did the classifier use the 10-K model's labels as patent labels? | No. It fitted separate weights using patent research labels. | It is a separate model, but separate weights alone do not prove external validity. |
| Were deployment and recent review population-wide? | No. Predictions and the old 600-case review file were restricted to the existing company-linked publication universe. | Their coverage and diagnostics cannot be presented as USPTO-wide validation. |
| Were training labels manual ground truth? | No. `AI_related == 2` in the third-party research training data was the positive target. | Internal performance is agreement with that target; label provenance and construct alignment require scrutiny. |
| Was the final model a reproduction of AIPD? | No. It used a lexical TF-IDF/logistic architecture and a third-party label target, rather than the USPTO abstract/claims architecture and target. | It must not be described as the USPTO classifier or a verified reproduction. |
| Was historical benchmark family independence established? | No. Exact normalized-text overlap removal found zero matching abstracts, but no global external-family crosswalk was available. | Do not describe the benchmark as proven family-independent. |
| Is the historical benchmark still untouched for future development? | No. Its labels and error examples have been inspected and used to motivate the redesign. | It remains reference evidence, not a fresh confirmatory test for subsequent variants. |
| Is the roughly 0.99 firm-count correlation sufficient? | No. It is aggregate agreement within available linked coverage, not patent-level gold-label accuracy. | Report patent-level validation and scale/zero-sensitive count comparisons separately. |
| Does a new model fill confidential recent applications? | No. Classification acts on public text only. | Publication truncation remains, independent of classification quality. |
| Was the final professor sample fully recovered? | The price-screen count matches, but upper-tail transformations and exact final control construction remain unresolved. | Preserve the term candidate sample until the professor's implementation is reconciled. |

No old input data, existing fitted model, historical metrics or professor files are replaced by this redesign. The old company-linked review file is not used to define membership of the new general-population review draw.

The corrective step is to separate measurement development, independent evaluation, company ownership linkage and economic testing as distinct documented components. Linking a frozen patent measure to a financial sample at the last stage is legitimate; optimizing that measure to obtain the desired financial association is not.
