"""Read-only source audit; write derived coverage tables separately."""
import json
import zipfile
from pathlib import Path

import pandas as pd


SOURCE = Path(r'C:\Users\sudet\Desktop\finai\ibes_production_output.zip')
OUT = Path(__file__).resolve().parent / 'analysis' / 'ibes_missingness_audit'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE) as z:
        def read(name):
            return pd.read_csv(z.open('ibes_production_output/' + name), low_memory=False)
        d = read('ibes_historical_unique.csv')
        coverage = read('historical_field_coverage.csv')
        status = read('query_status.csv')
        config = json.loads(z.read('ibes_production_output/run_config.json'))
    d['valid'] = d.Value.notna() & d.IsAPIError.eq(0)
    d['year'] = d.Date.str[:4]
    observed = d[d.valid]
    bounds = observed.groupby('RequestedInstrument').Date.agg(first='min', last='max')
    d = d.join(bounds, on='RequestedInstrument')
    d['missing_class'] = 'observed'
    missing = ~d.valid
    d.loc[missing & (d.Date < d['first']), 'missing_class'] = 'before_first_any_field'
    d.loc[missing & (d.Date > d['last']), 'missing_class'] = 'after_last_any_field'
    inside = missing & d.Date.ge(d['first']) & d.Date.le(d['last'])
    d.loc[inside, 'missing_class'] = 'within_any_field_history'
    print('MISSING CLASS\n', d.missing_class.value_counts().to_string())
    summary = d.groupby('missing_class').size().rename('rows').reset_index()
    summary.to_csv(OUT / 'missingness_decomposition.csv', index=False)
    field = d.groupby('Field').agg(rows=('valid', 'size'), valid=('valid', 'sum'))
    field['missing_pct'] = 100 * (1 - field.valid / field.rows)
    field.to_csv(OUT / 'field_coverage.csv')
    year = d.groupby('year').agg(rows=('valid', 'size'), valid=('valid', 'sum'))
    year['missing_pct'] = 100 * (1 - year.valid / year.rows)
    year.to_csv(OUT / 'year_coverage.csv')
    print('YEAR\n', year.to_string())
    print('FIELD\n', field.to_string())
    active = d[d.Date.ge(d['first']) & d.Date.le(d['last'])]
    active_field = active.groupby('Field').agg(rows=('valid', 'size'), valid=('valid', 'sum'))
    active_field['missing_pct'] = 100 * (1 - active_field.valid / active_field.rows)
    active_field.to_csv(OUT / 'field_coverage_within_observed_company_history.csv')
    print('WITHIN HISTORY\n', active_field.to_string())
    bounds['first_year'] = bounds['first'].str[:4]
    bounds.to_csv(OUT / 'instrument_observed_history.csv')
    print('FIRST YEARS\n', bounds.first_year.value_counts().sort_index().to_string())
    empty_pairs = coverage[coverage.NonEmpty.eq(0)]
    print('EMPTY RETURNED SERIES', len(empty_pairs), 'ROWS', empty_pairs.DatesReturned.sum())
    historical = status[status.Mode.eq('historical')].copy()
    historical.to_csv(OUT / 'historical_query_status.csv', index=False)
    print('FAILURE DETAILS\n', historical.loc[historical.Status.eq('API_RESPONSE_ERROR'), ['RequestedInstrument', 'Detail']].head(8).to_string(index=False))
    requested = len(historical) * d.Date.nunique() * len(config['historical_fields'])
    print('EXPECTED GRID', requested, 'RETURNED', len(d), 'UNRETURNED', requested-len(d))
    # Conditional dispersion missingness helps distinguish no estimates from one estimate.
    pivot = d.pivot(index=['RequestedInstrument', 'Date'], columns='Field', values='Value')
    for sd, ne in [('EPS1SD', 'EPS1NE'), ('EPSI1SD', 'EPSI1NE'), ('EPSI2SD', 'EPSI2NE'), ('EPSI3SD', 'EPSI3NE')]:
        n = pd.to_numeric(pivot[ne], errors='coerce')
        print('SD CHECK', sd, 'missing with NE<=1', int((pivot[sd].isna() & n.le(1)).sum()),
              'missing with NE>=2', int((pivot[sd].isna() & n.ge(2)).sum()))
    print('OUTPUT', OUT)


if __name__ == '__main__':
    main()
