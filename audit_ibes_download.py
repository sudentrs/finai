import json
import zipfile
from pathlib import Path

import pandas as pd


def main():
    source = Path(r'C:\Users\sudet\Desktop\finai\ibes_production_output.zip')
    with zipfile.ZipFile(source) as z:
        def read(name):
            return pd.read_csv(z.open('ibes_production_output/' + name), low_memory=False)

        d = read('ibes_historical_unique.csv')
        c = read('historical_field_coverage.csv')
        s = read('query_status.csv')
        m = read('original_input_manifest.csv')
        snap = read('ibes_snapshot_unique.csv')
        print('INPUT', len(m), 'UNIQUE', m.RequestedInstrument.nunique())
        print('HISTORICAL', len(d), 'INSTRUMENTS', d.RequestedInstrument.nunique(),
              'DATE RANGE', d.Date.min(), d.Date.max(), 'DATES', d.Date.nunique(),
              'FIELDS', d.Field.nunique(), 'DUPLICATES',
              d.duplicated(['RequestedInstrument', 'Date', 'Field']).sum())
        print('STATUS\n', s.groupby(['Mode', 'Status']).size().to_string())
        print('COVERAGE\n', c.groupby('Field')[['DatesReturned', 'NonEmpty', 'Missing', 'APIErrorCells']].sum().to_string())
        print('YEAR\n', d.assign(Year=d.Date.str[:4], Valid=d.Value.notna() & d.IsAPIError.eq(0))
              .groupby('Year').agg(rows=('Field', 'size'), instruments=('RequestedInstrument', 'nunique'),
                                  valid=('Valid', 'sum')).to_string())
        print('ERROR VALUES\n', d.loc[d.IsAPIError.ne(0), 'Value'].value_counts().head(10).to_string())
        print('SNAPSHOT\n', snap.groupby('Field').agg(rows=('Field', 'size'), nonmissing=('Value', 'count')).to_string())
        print('NAMES\n', snap.loc[snap.Field.eq('IBNAME'), ['RequestedInstrument', 'Value']].tail(20).to_string(index=False))
        print('PREFIXES\n', m.RequestedInstrument.str.extract(r'^([^:]+:)')[0].fillna('no-colon').value_counts().to_string())
        print('STATUS DETAILS\n', s.loc[~s.Status.eq('HAS_DATA')].head(15).to_string(index=False))
        print('VALID FIRMS', d.loc[d.Value.notna() & d.IsAPIError.eq(0), 'RequestedInstrument'].nunique())


if __name__ == '__main__':
    main()
