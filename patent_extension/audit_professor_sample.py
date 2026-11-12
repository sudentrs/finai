"""Recover candidate sample keys without changing ratios or source data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r'C:\Users\sudet\Desktop\finai\publication pack\ITEM1_reg.dta')


def main():
    out = ROOT / 'outputs' / 'sample_audit'
    out.mkdir(parents=True, exist_ok=True)
    cols = ['company_name', 'cik', 'crsp_permno', 'permco', 'compustat_gvkey',
            'filed_date', 'period_end', 'filed_year', 'filed_month', 'mdate', 'prc',
            'expertiseratio', 'airatio', 'total_sentences', 'ai_sentences_count',
            'ai_expertise', 'ff48', 'common', 'obs_main', 'exch_main', 'primary_sec', 'excntry']
    d = pd.read_stata(SOURCE, columns=cols, convert_categoricals=False)
    for field in ['filed_date', 'period_end']:
        d[field + '_parsed'] = pd.to_datetime(d[field].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
    d['candidate_price_screen'] = d.prc.ge(5)
    d['filed_year_is_panel_year'] = d.filed_year.eq(d.mdate.dt.year)
    d['filed_month_is_panel_month'] = d.filed_month.eq(d.mdate.dt.month)
    summary = {'source': str(SOURCE), 'status': 'candidate sample; professor final transformations not confirmed',
               'rows': len(d), 'months': d.mdate.nunique(), 'permno': d.crsp_permno.nunique(),
               'duplicate_permno_month': int(d.duplicated(['crsp_permno', 'mdate']).sum()),
               'price_below_5': int(d.prc.lt(5).sum()), 'price_missing': int(d.prc.isna().sum()),
               'filed_year_equals_panel_year_all': bool(d.filed_year_is_panel_year.all()),
               'filed_month_equals_panel_month_all': bool(d.filed_month_is_panel_month.all())}
    key = ['cik', 'filed_date', 'period_end']
    counts = d.groupby(key).agg(stock_months=('mdate', 'size'), first_month=('mdate', 'min'),
                                last_month=('mdate', 'max'), eligible_price_months=('candidate_price_screen', 'sum'),
                                expertise_versions=('expertiseratio', 'nunique'), ai_versions=('airatio', 'nunique'),
                                permno_versions=('crsp_permno', 'nunique'), permco_versions=('permco', 'nunique'))
    counts.to_csv(out / 'filing_repetition_audit.csv')
    # This is a candidate filing roster, not an accounting or regression dataset.
    roster = d.sort_values('mdate').drop_duplicates(key)[key + ['company_name', 'compustat_gvkey',
            'filed_date_parsed', 'period_end_parsed', 'total_sentences', 'ai_sentences_count', 'ai_expertise']]
    roster = roster.merge(counts.reset_index(), on=key, validate='one_to_one')
    roster['in_candidate_price_sample'] = roster.eligible_price_months.gt(0)
    roster.to_csv(out / 'candidate_filing_roster.csv', index=False)
    d[key + ['crsp_permno', 'permco', 'mdate', 'prc', 'ff48', 'candidate_price_screen']].to_csv(
        out / 'stock_month_filing_crosswalk.csv', index=False)
    summary['unique_filing_keys'] = len(roster)
    summary['filings_with_varying_ratios'] = int((counts.expertise_versions.gt(1) | counts.ai_versions.gt(1)).sum())
    summary['filings_with_multiple_permco'] = int(counts.permco_versions.gt(1).sum())
    summary['expertise_above_ai_rows'] = int(d.expertiseratio.gt(d.airatio + 1e-7).sum())
    anomalies = d[d.expertiseratio.gt(d.airatio + 1e-7)].copy()
    anomalies.to_csv(out / 'expertise_above_ai_stock_months.csv', index=False)
    anomalies.drop_duplicates(key).to_csv(out / 'expertise_above_ai_filings.csv', index=False)
    for name, frame in [('delivered_panel', d), ('candidate_price_ge5', d[d.candidate_price_screen])]:
        monthly = frame.assign(ai_positive=frame.airatio.gt(0), expertise_positive=frame.expertiseratio.gt(0))
        monthly = monthly.groupby('mdate').agg(n=('crsp_permno', 'size'), ai_mean=('airatio', 'mean'),
                    expertise_mean=('expertiseratio', 'mean'), ai_positive_share=('ai_positive', 'mean'),
                    expertise_positive_share=('expertise_positive', 'mean'))
        monthly.to_csv(out / f'{name}_monthly_diagnostics.csv')
        summary[name] = {'rows': len(frame), 'average_monthly_n': float(monthly.n.mean()),
                         'permno': frame.crsp_permno.nunique(), 'ff48_missing': int(frame.ff48.isna().sum()),
                         'ai_max': float(frame.airatio.max()), 'expertise_max': float(frame.expertiseratio.max())}
    denom = d.total_sentences.replace(0, np.nan)
    summary['expertise_count_ratio_mismatch_rows'] = int(((d.ai_expertise / denom - d.expertiseratio).abs() > 1e-6).sum())
    summary['ai_count_ratio_mismatch_rows'] = int(((d.ai_sentences_count / denom - d.airatio).abs() > 1e-6).sum())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
