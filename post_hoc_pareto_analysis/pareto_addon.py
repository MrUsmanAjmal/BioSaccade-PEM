"""Descriptive Pareto-frontier add-on (added AFTER the locked paper run; see NOTE.md).

Runs PEM_uniform, PEM_greedy, PPAT2026_M1 and PEM_audit_only on exactly the 45 clips, regime and draw
of the paper's kappa / certificate sweeps, reusing the paper run's code, fitted models, selected
settings and geometry. Nothing in the paper experiment folder is written. A reproducibility check also
re-runs PEM_R9_TwinAudit at the selected settings; its rows must equal the paper's PEM_R9_k2 rows.
"""
import json, sys, time
from dataclasses import replace
from pathlib import Path

PAPER = Path(r'<RESULTS_DIR>\results\local_01_91e457e261dc')
SRC = Path(r'<RESULTS_DIR>\src')          # modules of the paper run (hashes checked below)
OUT = Path(r'<RESULTS_DIR>\pareto_addon')
ROOTS = {'DAVIS': (Path(r'<DAVIS_ROOT>\JPEGImages\480p'), '*.jpg'),
         'REDS': (Path(r'<REDS_ROOT>\val_sharp'), '*.png'),
         'REDS_train': (Path(r'<REDS_ROOT>\train_sharp'), '*.png')}
METHODS = ['PEM_uniform', 'PEM_greedy', 'PPAT2026_M1', 'PEM_audit_only']
CHECK = 'PEM_R9_TwinAudit'
sys.path.insert(0, str(SRC))


def run():
    import hashlib
    from concurrent.futures import wait, FIRST_COMPLETED
    import r9_runner as R
    from r9_core import MethodConfig
    from r9_data import _clip
    from r9_engine import Geometry, core_methods

    code = json.loads((PAPER / 'inputs.lock.json').read_text())['code']
    for name, h in code.items():
        assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == h, f'{name} differs from the paper run'
    lock = json.loads((PAPER / 'protocol.lock.json').read_text())
    selected = MethodConfig(**json.loads((PAPER / 'development_selection.json').read_text())['selected'])
    entries = {}
    for j in lock['jobs']:
        if j['study'] in ('kappa', 'certificate'):
            entries[(j['dataset'], j['clip_id'], j['regime'], j['draw'])] = j
    assert len(entries) == 45
    by = {m.name: m for m in core_methods()}
    methods = [by[n] for n in METHODS] + [by[CHECK]]
    (OUT / 'checkpoints').mkdir(parents=True, exist_ok=True)
    payloads = []
    for i, ((ds, cid, regime, draw), j) in enumerate(sorted(entries.items())):
        base, pattern = ROOTS[ds]
        files = sorted((base / cid).glob(pattern))[:j['geometry']['frames']]
        c = _clip(ds, cid, cid, j['role'], files)
        assert c.content_sha256 == j['content_sha256'], f'content differs for {ds}/{cid}'
        c = replace(c, source_group=j['source_group'])
        geo = Geometry(**j['geometry'])
        payloads.append(dict(index=i, job=('pareto_addon', c, regime, draw, geo, selected, methods), key=f'{ds}_{cid}',
                             checkpoint=str(OUT / 'checkpoints' / f'{ds}_{cid}.json'), keep=False, trace_dir=None))
    todo = [p for p in payloads if not Path(p['checkpoint']).exists()]
    print(f'{len(payloads)} clips verified by content hash; {len(todo)} to run', flush=True)
    t = time.perf_counter()
    if todo:
        with R.make_pool(5, PAPER / 'models.joblib') as pool:
            inflight = {pool.submit(R._execute_job, p) for p in todo}; done = 0
            while inflight:
                finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for f in finished:
                    f.result(); done += 1
                    print(f'{done}/{len(todo)} clips, {(time.perf_counter()-t)/60:.1f} min', flush=True)
    rows = [r for p in payloads for r in json.loads(Path(p['checkpoint']).read_text())['rows']]
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / 'addon_sequence_results.csv', index=False)
    print('run finished', flush=True)


if __name__ == '__main__':
    run()
