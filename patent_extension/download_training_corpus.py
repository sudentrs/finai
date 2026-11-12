"""Retrieve pinned public research data without running third-party code."""
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent


def main():
    root = ROOT / 'references' / 'aipat'
    manifest = json.loads((root / 'source_manifest.json').read_text())
    name = 'DATA/df_train_v6_af_pu_af_cpc_select_af_family_af_word_select.csv'
    entry = next(x for x in manifest['files'] if x['path'] == name)
    pointer = entry['lfs_pointer']
    expected_hash = next(x.split('sha256:')[1] for x in pointer.splitlines() if x.startswith('oid '))
    target = root / name
    expected = entry['expected_bytes']
    if target.exists() and target.stat().st_size == expected:
        if hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() == expected_hash:
            print('Existing corpus verified', flush=True)
            return
    url = f'https://media.githubusercontent.com/media/{manifest["repository"]}/{manifest["commit"]}/{name}'
    partial = target.with_suffix('.csv.partial')
    previous = time.monotonic()
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        count = 0
        digest = hashlib.sha256()
        with partial.open('wb') as stream:
            for block in response.iter_content(1024 * 1024):
                count += len(block)
                if count > expected:
                    raise ValueError('Response exceeds expected LFS size')
                stream.write(block)
                digest.update(block)
                if time.monotonic() - previous > 20:
                    print(f'Downloaded {count:,}/{expected:,} bytes', flush=True)
                    previous = time.monotonic()
    if count != expected or digest.hexdigest() != expected_hash:
        raise ValueError('Training corpus size/hash verification failed')
    partial.replace(target)
    record = {'url': url, 'commit': manifest['commit'], 'bytes': count,
              'sha256': digest.hexdigest(), 'status': 'verified',
              'license_note': 'No explicit repository license found; no raw-data redistribution authorized by this audit.'}
    (root / 'training_corpus_download.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
