"""Extract reference text and metadata without modifying publication sources."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
from pypdf import PdfReader

ROOT = Path(r'C:\Users\sudet\Desktop\finai\publication pack')
OUT = Path(__file__).resolve().parent / 'analysis' / 'publication_pack_review'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(ROOT / 'Manuscript - AI Risk 20261031.docx') as z:
        xml = ET.fromstring(z.read('word/document.xml'))
        paras = [''.join(t.text or '' for t in p.findall('.//w:t', ns))
                 for p in xml.findall('.//w:p', ns)]
        (OUT / 'manuscript.txt').write_text('\n'.join(paras), encoding='utf-8')
        for name in ['word/comments.xml']:
            if name in z.namelist():
                (OUT / 'manuscript_comments.txt').write_text(
                    '\n'.join(ET.fromstring(z.read(name)).itertext()), encoding='utf-8')
    for f in ROOT.glob('*.pdf'):
        reader = PdfReader(f)
        (OUT / (f.stem + '.txt')).write_text(
            '\n\n'.join(f'PAGE {i+1}\n{p.extract_text() or ""}'
                          for i, p in enumerate(reader.pages)), encoding='utf-8')
        print('PDF', f.name, len(reader.pages))
    for f in ROOT.glob('*.xlsx'):
        sheets = pd.read_excel(f, sheet_name=None, header=None)
        text = '\n\n'.join('SHEET ' + name + '\n' + frame.dropna(how='all').to_string(index=False, header=False)
                           for name, frame in sheets.items())
        (OUT / (f.stem + '.txt')).write_text(text, encoding='utf-8')
        print('EXCEL', f.name, [(n, d.shape) for n, d in sheets.items()])
    reader = pd.read_stata(ROOT / 'ITEM1_reg.dta', iterator=True, convert_categoricals=False)
    labels = reader.variable_labels()
    (OUT / 'stata_labels.json').write_text(json.dumps(labels, indent=2), encoding='utf-8')
    print('STATA LABELS', json.dumps(labels))
    print('OUTPUT', OUT)


if __name__ == '__main__':
    main()
