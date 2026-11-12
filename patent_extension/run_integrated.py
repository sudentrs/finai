"""One entry point with explicit prerequisites and no silent stage skipping."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / 'outputs'
    output.mkdir(parents=True, exist_ok=True)
    status = {'started_utc': datetime.now(timezone.utc).isoformat(), 'stages': [], 'paper_ready': False}
    stages = ['audit_professor_sample.py', 'audit_external_benchmark.py', 'abstract_model.py',
              'build_patent_panels.py', 'test_pipeline.py', 'summarize_validation.py', 'write_full_run_report.py']
    for stage in stages:
        print('RUN', stage, flush=True)
        result = subprocess.run([sys.executable, str(ROOT / stage)], cwd=ROOT.parent)
        status['stages'].append({'script': stage, 'exit_code': result.returncode})
        (output / 'integrated_run_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        if result.returncode:
            raise SystemExit(result.returncode)
    integration = json.loads((output / 'integrated_panels/integration_status.json').read_text())
    status['integration'] = integration
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    status['remaining_requirements'] = ['Recent independent manual labels', 'Final professor sample transformations',
                                       'Historical assignee-parent ownership validation']
    if integration['text_status'] == 'missing_abstract_source':
        status['remaining_requirements'].insert(0, 'Authenticated USPTO abstract-file download')
    (output / 'integrated_run_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
