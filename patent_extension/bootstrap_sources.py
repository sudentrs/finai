"""Download bounded research references at a pinned commit; never execute them."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
REPO = 'yusu2022/AIPat-Data-and-Methods-for-Identifying-Artificial-Intelligence-Related-Patents'
LIMIT = 20_000_000


def fetch(url, limit=LIMIT):
    response = requests.get(url, timeout=(15, 90), stream=True)
    response.raise_for_status()
    parts, size = [], 0
    for part in response.iter_content(65536):
        size += len(part)
        if size > limit:
            raise ValueError(f'Download exceeds {limit} bytes: {url}')
        parts.append(part)
    return b''.join(parts)


def main():
    out = ROOT / 'references' / 'aipat'
    out.mkdir(parents=True, exist_ok=True)
    commit = requests.get(f'https://api.github.com/repos/{REPO}/commits/main', timeout=40)
    commit.raise_for_status()
    sha = commit.json()['sha']
    tree = requests.get(f'https://api.github.com/repos/{REPO}/git/trees/{sha}?recursive=1', timeout=40)
    tree.raise_for_status()
    files = [x for x in tree.json()['tree'] if x['type'] == 'blob']
    (out / 'repository_inventory.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    wanted = ['README.md', 'application.ipynb', 'test.ipynb', 'train.ipynb',
              'text_process.py', 'PU_Learning/Utils/Model.py',
              'df_smj_4k_final.csv', 'DATA/df_train_v6_af_pu_af_cpc_select_af_family_af_word_select.csv',
              'model2/CNN_TEXT_CA.pth', 'model2/dictionary.pkl',
              'pgpub_patent_ai_pro_cnn_text_ca_20260629.csv', 'requirements.txt']
    manifest = {'repository': REPO, 'commit': sha,
                'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                'license_files': [x['path'] for x in files if 'license' in x['path'].lower()],
                'files': []}
    for name in wanted:
        url = f'https://raw.githubusercontent.com/{REPO}/{sha}/{name}'
        data = fetch(url)
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {'path': name, 'url': url}
        if data.startswith(b'version https://git-lfs.github.com/spec/v1'):
            pointer = data.decode('utf-8')
            record['lfs_pointer'] = pointer
            expected_size = int(next(x.split()[1] for x in pointer.splitlines() if x.startswith('size ')))
            expected_hash = next(x.split('sha256:')[1] for x in pointer.splitlines() if x.startswith('oid '))
            record['expected_bytes'] = expected_size
            target.with_name(target.name + '.lfs-pointer.txt').write_bytes(data)
            # Only the human benchmark is downloaded now; weights are not loaded.
            if name == 'df_smj_4k_final.csv' and expected_size <= LIMIT:
                url = f'https://media.githubusercontent.com/media/{REPO}/{sha}/{name}'
                try:
                    data = fetch(url)
                    if hashlib.sha256(data).hexdigest() != expected_hash:
                        raise ValueError('LFS content hash mismatch')
                    target.write_bytes(data)
                    record.update(status='downloaded', content_url=url, bytes=len(data), sha256=expected_hash)
                except (requests.RequestException, ValueError) as exc:
                    record.update(status='lfs_download_failed', error=str(exc))
            else:
                record['status'] = 'pointer_only_not_downloaded'
        else:
            target.write_bytes(data)
            record.update(status='downloaded', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        manifest['files'].append(record)
        print(name, record['status'], record.get('expected_bytes', record.get('bytes')), flush=True)
    (out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('LICENSE FILES', manifest['license_files'], flush=True)
    print('PINNED COMMIT', sha, flush=True)


if __name__ == '__main__':
    main()
