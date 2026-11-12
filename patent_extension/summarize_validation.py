"""Create diagnostic comparisons, never paper results, for the frozen classifier."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix
from abstract_model import metrics as classification_metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'validation_diagnostics'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(ROOT / 'outputs/abstract_model/evaluation_metrics.csv')
    model = json.loads((ROOT / 'outputs/abstract_model/model_card.json').read_text())
    predictions = pd.read_csv(ROOT / 'outputs/abstract_model/external_benchmark_predictions.csv', dtype={'doc_id': str})
    raw = pd.read_csv(ROOT / 'references/aipat/df_smj_4k_final.csv', dtype={'doc_id': str})
    inspect = raw[['doc_id', 'application_id', 'pub_dt', 'abstract', 'actual']].merge(
        predictions[['doc_id', 'model_probability']], on='doc_id', validate='one_to_one')
    inspect['predicted_ai'] = inspect.model_probability.ge(model['threshold']).astype(int)
    inspect['error_type'] = np.select([inspect.actual.eq(0) & inspect.predicted_ai.eq(1),
                                       inspect.actual.eq(1) & inspect.predicted_ai.eq(0)],
                                      ['false_positive', 'false_negative'], default='agreement')
    examples = [frame.sample(min(25, len(frame)), random_state=42)
                for _, frame in inspect[inspect.error_type.ne('agreement')].groupby('error_type')]
    pd.concat(examples, ignore_index=True).to_csv(OUT / 'external_benchmark_error_examples.csv', index=False)
    correlations = []
    scatter_panels = []
    source = ROOT / 'outputs/integrated_panels'
    publication = pd.read_csv(source / 'linked_applications.csv', dtype={'pgpub_id': str, 'cik': str, 'application_id': str})
    own = pd.read_csv(source / 'author_publication_predictions.csv', dtype={'pgpub_id': str})
    publication = publication.drop(columns=['author_ai', 'author_probability']).merge(own, on='pgpub_id', how='left', validate='many_to_one')
    publication = publication.drop_duplicates('pgpub_id')
    publication['publication_year'] = pd.to_datetime(publication.published_date).dt.year
    annual = publication.groupby('publication_year').agg(n=('pgpub_id', 'size'),
        author_classified=('author_ai', 'count'), official_classified=('official_ai93', 'count'),
        author_ai=('author_ai', 'sum'), official_ai=('official_ai93', 'sum'))
    annual['author_ai'] = annual.author_ai.where(annual.author_classified.gt(0))
    annual['official_ai'] = annual.official_ai.where(annual.official_classified.gt(0))
    # Shares use each source's available labels, with denominators shown explicitly.
    annual['author_share_available'] = annual.author_ai / annual.author_classified.replace(0, np.nan)
    annual['official_share_available'] = annual.official_ai / annual.official_classified.replace(0, np.nan)
    annual.to_csv(OUT / 'publication_year_label_coverage.csv')
    overlap = publication.dropna(subset=['official_ai93', 'author_ai'])
    agreement = [dict(publication_year='all', **classification_metrics(overlap.official_ai93,
                               overlap.author_probability, model['threshold']))]
    for year, frame in overlap.groupby('publication_year'):
        agreement.append(dict(publication_year=int(year), **classification_metrics(frame.official_ai93,
                                   frame.author_probability, model['threshold'])))
    pd.DataFrame(agreement).to_csv(OUT / 'publication_level_aipd_agreement_metrics.csv', index=False)
    matrix = confusion_matrix(overlap.official_ai93, overlap.author_ai, labels=[0, 1])
    pd.DataFrame(matrix, index=['official_not_ai', 'official_ai'], columns=['author_not_ai', 'author_ai']).to_csv(
        OUT / 'publication_agreement_confusion.csv')
    for file in source.glob('*_research_panel.csv'):
        d = pd.read_csv(file, low_memory=False)
        for window, eligible in [('all_observed', pd.Series(True, index=d.index)),
                                  ('mature_18m', d.publication_maturity_18m.eq(True)),
                                  ('official_robustness_fy2015_2021', d.fiscal_year_label.between(2015, 2021))]:
            s = d.loc[eligible].dropna(subset=['author_ai_application_count', 'official93_ai_application_count'])
            x, y = s.author_ai_application_count, s.official93_ai_application_count
            correlations.append({'panel': file.stem, 'window': window, 'n': len(s),
                'pearson': float(pearsonr(x, y).statistic) if len(s)>2 and x.nunique()>1 and y.nunique()>1 else np.nan,
                'spearman': float(spearmanr(x, y).statistic) if len(s)>2 and x.nunique()>1 and y.nunique()>1 else np.nan,
                'pearson_log1p': float(pearsonr(np.log1p(x), np.log1p(y)).statistic) if len(s)>2 and x.nunique()>1 and y.nunique()>1 else np.nan,
                'zero_nonzero_agreement': float(x.gt(0).eq(y.gt(0)).mean()) if len(s) else np.nan,
                'interpretation': 'Agreement on available same-coverage counts, not proof of AI accuracy or causal validation'})
            if window == 'mature_18m':
                scatter_panels.append((file.stem.replace('_research_panel', ''), s[['official93_ai_application_count', 'author_ai_application_count']]))
    pd.DataFrame(correlations).to_csv(OUT / 'filing_level_official_author_count_correlations.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    subset = metrics[metrics.partition.isin(['temporal_family_test', 'external_historical_grants'])]
    x = np.arange(len(subset))
    axes[0].bar(x-.17, subset.precision*100, .34, color='#276B9A', label='Precision')
    axes[0].bar(x+.17, subset.recall*100, .34, color='#C05A40', label='Recall')
    axes[0].set_xticks(x, ['Research-label\nfamily/time holdout', 'Historical grant\nbenchmark'])
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel('Percent')
    axes[0].set_title('A. Frozen-model performance', loc='left', fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    year = annual.loc[annual.index >= 2015]
    axes[1].plot(year.index, year.author_share_available*100, color='#276B9A', marker='o', ms=3, label='Author classifier')
    axes[1].plot(year.index, year.official_share_available*100, color='#C05A40', marker='s', ms=3, label='Official AIPD 93%')
    axes[1].set_title('B. Linked publications with available labels', loc='left', fontsize=10)
    axes[1].set_ylabel('Classified AI (%)')
    axes[1].set_xlabel('Publication year')
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('Patent classifier diagnostics - research candidate, not paper results', fontsize=11)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'classifier_validation_diagnostics.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7), constrained_layout=True)
    for ax, (name, frame) in zip(axes.ravel(), scatter_panels):
        x = np.log1p(frame.official93_ai_application_count)
        y = np.log1p(frame.author_ai_application_count)
        ax.scatter(x, y, s=5, alpha=.16, color='#276B9A', rasterized=True)
        bound = max(x.max(), y.max()) if len(frame) else 1
        ax.plot([0, bound], [0, bound], '--', color='#777777', linewidth=.8)
        sample = 'Broad filings' if name.startswith('broad') else 'Professor price-screen candidate'
        scope = 'Item 1' if 'ITEM_1' in name else 'Document'
        ax.set_title(f'{sample}\n{scope}; paired N = {len(frame):,}', fontsize=9)
        ax.set_xlabel('Official AI application count: log(1 + count)')
        ax.set_ylabel('Author AI application count: log(1 + count)')
    fig.suptitle('Count agreement on mature, jointly classified observations\nDiagnostic only: correlation does not establish classification accuracy', fontsize=10)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'paired_count_agreement.{ext}', dpi=180)
    plt.close(fig)
    gate = {'computational_classification_complete': True, 'paper_ready': False,
            'publication_gate': 'NOT PASSED',
            'reasons': ['Low precision on the historical external benchmark despite strong internal scores',
                       'Recent independent human validation is not complete',
                       'The external benchmark family overlap is not established',
                       'Historical parent/subsidiary matching and professor ratio transformations remain unresolved'],
            'regressions_and_paper_figures': 'Not promoted/generated from this unapproved classifier',
            'new_labels_are': 'Frozen independent-code lexical model trained on AIPat research labels; not an exact AIPD replication'}
    (OUT / 'release_gate.json').write_text(json.dumps(gate, indent=2), encoding='utf-8')
    print(json.dumps(gate, indent=2), flush=True)
    print(annual.tail(6).to_string(), flush=True)


if __name__ == '__main__':
    main()
