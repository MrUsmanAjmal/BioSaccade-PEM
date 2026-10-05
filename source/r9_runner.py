"""Experiment runner."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib, importlib.metadata, json, os, platform, sys, time, zipfile
import cv2, joblib, numpy as np, pandas as pd
from r9_data import (DATASETS, Clip, _clip, sha256, digest_json, atomic_json, immutable_json,
    download, extract_dataset, discover_davis, discover_reds,
    validate_manifest, load_clip, synthetic_frames)
from r9_core import MethodConfig, AuditEnsemble
from r9_engine import (Geometry, Method, core_methods, neural_methods, observations, previews, run_sequence)
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from r9_neural import SPECS, load_baselines, baseline_sanity
from r9_stats import export_analysis, export_backbone_comparisons, publication_figures, trace_figures, paper_tables, CONTEXT
from video_core import RiskMemory


@dataclass
class RunConfig:
    root: str
    profile: str = 'smoke'  # smoke, laptop, paper
    experiment_id: str = 'r9_01'
    run_neural: bool = True
    device: str = 'auto'
    download_data: bool = True
    dataset_roots: dict = None
    threads: int = 2
    export_figures: bool = True
    line_dpi: int = 1000
    high_resolution_reds: bool = False
    save_all_traces: bool = False
    workers: int = 0               # CPU processes for the controller/evaluator; 0 = automatic
    data_root: str = None          # downloads/extractions/synthetic smoke data; defaults to root.
    include_video_sr: bool = True  # BasicVSR++ (CVPR 2022) causal + offline reference
    neural_workers: int = 4        # processes computing neural baseline predictions (speed only)
    neural_threads: int = 2        # PyTorch threads per neural worker (speed only)


def environment():
    packages = ['numpy','scipy','pandas','scikit-learn','matplotlib','opencv-python-headless',
                'torch','einops','joblib','psutil']
    versions = {}
    for name in packages:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'not installed'
    result = dict(python=sys.version, platform=platform.platform(), packages=versions)
    try:
        import torch
        result.update(cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            torch_threads=torch.get_num_threads())
    except ImportError: pass
    return result


def _stable_environment(v):
    """Only package versions enter the lock; host/GPU names may change between
    sessions without changing the experiment."""
    return dict(python=v['python'].split()[0],packages=v['packages'])


def source_hashes():
    return {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def _choose(clips, n):
    ordered = sorted(clips, key=lambda c: hashlib.sha256((c.dataset+'/'+c.clip_id).encode()).hexdigest())
    return ordered if n is None else ordered[:n]


def prepare_data(config, split):
    root=Path(config.data_root or config.root); profile=config.profile
    if profile not in ['smoke','laptop','paper']: raise ValueError('Unknown profile')
    if profile=='smoke':
        geo=Geometry(height=64,width=96,fovea=16,frames=12)
        clips=[]
        for role,count,offset in [('fit',4,100),('development',2,200),('external_test',4,300)]:
            for i in range(count):
                name=f'{role}_{i:02d}'; folder=root/'data'/'synthetic'/name
                folder.mkdir(parents=True,exist_ok=True)
                for t,frame in enumerate(synthetic_frames(geo,offset+i,cut=i%2==1)):
                    path=folder/f'{t:04d}.png'
                    pixels=np.rint(frame[...,::-1]*255).astype(np.uint8)
                    # Regenerate the test frames.
                    if not cv2.imwrite(str(path),pixels): raise IOError(path)
                clips.append(_clip('Synthetic',name,name,role,sorted(folder.glob('*.png'))))
        clips,report=validate_manifest(clips)
        return clips,geo,report
    geo=Geometry(frames=100 if profile=='paper' else 32)
    roots=config.dataset_roots or {}
    resolved={}
    for name,entry in DATASETS.items():
        if name in roots:
            p=Path(roots[name]).expanduser().resolve()
            if not p.is_dir(): raise FileNotFoundError(p)
            resolved[name]=p
            continue
        if not config.download_data or not entry.get('url'):
            raise FileNotFoundError(f'Set dataset_roots[{name!r}] to the local {name} folder ({entry["source"]}). No substitute data is used.')
        print('Preparing official dataset:',name,flush=True)
        filename={'DAVIS':'DAVIS-2017-trainval-480p.zip','REDS':'val_sharp.zip'}[name]
        try:
            archive=download(entry['url'],root/'downloads'/filename,entry.get('sha256'))
            resolved[name]=extract_dataset(archive,root/'data'/name,name)
        except Exception as error:
            raise RuntimeError(f'{name} download failed. Obtain the dataset from {entry["source"]}. Completed work is retained; no substitute data will be silently used.') from error
    davis=discover_davis(resolved['DAVIS'],split,geo.frames)
    if profile=='laptop':
        davis=sum([_choose([c for c in davis if c.role==role],n)
                   for role,n in [('fit',12),('development',4),('legacy_test',10)]],[])
    # Both REDS splits must be complete on disk; the laptop pilot then uses the first 10 clips of each.
    limit=None if profile=='paper' else 10
    reds=discover_reds(resolved['REDS'],geo.frames,limit,dataset='REDS',expected=30)
    reds_train=discover_reds(resolved['REDS_train'],geo.frames,limit,dataset='REDS_train',expected=240)
    if profile=='paper':
        if len(reds_train)!=240: raise ValueError('Paper profile requires all 240 REDS train_sharp clips')
        if len(reds)!=30: raise ValueError('Paper profile requires all 30 REDS validation clips')
        if len(davis)!=90: raise ValueError('Legacy DAVIS manifest must retain all 90 original sources')
        if any(len(c.paths)!=100 for c in reds+reds_train): raise ValueError('REDS paper clips must have 100 frames')
    clips,report=validate_manifest(davis+reds+reds_train)
    report['dataset_roots']={k:str(v) for k,v in resolved.items()}
    report['pilot_scope']=('laptop pilot: DAVIS (12 fit, 4 development, 10 legacy test) + first 10 REDS val + first 10 REDS train clips, 32 frames'
                           if profile=='laptop' else 'all configured datasets')
    report['official_sources']=DATASETS
    report['reds_train_role']='REDS train_sharp replaces Vimeo-90K and is used only as unseen external test data: no model in this study is trained on REDS.'
    report['limitation']='Content hashing detects exact duplicates; it cannot certify absence of near-duplicates or pretrained data overlap.'
    report['content_digests']=dataset_digests(clips)
    report['clip_counts']={d:sum(c.dataset==d for c in clips) for d in sorted({c.dataset for c in clips})}
    return clips,geo,report

def dataset_digests(clips):
    """Order-independent content digest per dataset; published so readers can verify
    that the exact same frames (including the unpinned REDS mirror) were evaluated."""
    out={}
    for d in sorted({c.dataset for c in clips}):
        items=sorted(c.clip_id+':'+c.content_sha256 for c in clips if c.dataset==d)
        out[d]=hashlib.sha256('\n'.join(items).encode()).hexdigest()
    return out


def _lockable_clip(c):
    """Clip identity by content, not by filesystem location, so a session that
    re-extracts data to another disk resumes the same locked experiment."""
    d=asdict(c); d.pop('paths'); return d


def fit_models(clips,geo,acfg,out,profile):
    fit=sorted([c for c in clips if c.role=='fit'],key=lambda c:c.clip_id)
    if len(fit)<4: raise ValueError('At least four fitting sources are required')
    cut=len(fit)//2; memory_sources=fit[:cut]; auditor_sources=fit[cut:]
    provenance=dict(memory_sources=[c.source_group for c in memory_sources],
                    auditor_sources=[c.source_group for c in auditor_sources],
                    labels='only selected, noisy foveal observations; no dense clean targets',
                    harvest_policy='uniform',regimes=['nominal','noisy'],
                    memory_auditor_source_disjoint=True, group_oof_mean_folds=min(5,len(auditor_sources)))
    assert not set(provenance['memory_sources'])&set(provenance['auditor_sources'])
    path=out/'models.joblib'; meta=out/'training_provenance.json'
    if path.exists() and meta.exists():
        saved=json.loads(meta.read_text())
        if saved.get('model_sha256')!=sha256(path): raise ValueError('Fitted-model checksum mismatch')
        # Load only this run's locally generated model; never an uploaded pickle.
        return (*joblib.load(path),saved)
    memory_rows=[]
    harvest=Method('training_uniform','uniform','zero',envelope=False)
    for index,c in enumerate(memory_sources):
        frames=load_clip(c,geo)
        for regime in ['nominal','noisy']:
            obs=observations(frames,geo,c.source_group,regime,0)
            _,_,_,mr,_=run_sequence(frames,obs,geo,acfg,c.source_group,regime,0,harvest,collect=True)
            memory_rows.extend(mr)
        print(f'Memory fit harvest {index+1}/{len(memory_sources)}',flush=True)
    X,C,H,R=[np.concatenate([row[k] for row in memory_rows]) for k in range(4)]
    memory=RiskMemory(leaf=4 if profile=='smoke' else 40).fit(X,C,H,R)
    rows=[]
    for index,c in enumerate(auditor_sources):
        frames=load_clip(c,geo)
        for regime in ['nominal','noisy']:
            obs=observations(frames,geo,c.source_group,regime,0)
            _,_,tr,_,_=run_sequence(frames,obs,geo,acfg,c.source_group,regime,0,harvest,memory,collect=True)
            rows.extend(tr)
        print(f'Auditor fit harvest {index+1}/{len(auditor_sources)}',flush=True)
    Z=np.stack([r[0] for r in rows]); d=np.array([r[1] for r in rows]); e=np.array([r[2] for r in rows]); groups=np.array([r[3] for r in rows])
    auditor=AuditEnsemble(trees=16 if profile=='smoke' else 64,leaf=4 if profile=='smoke' else 12).fit(Z,d,e,groups)
    joblib.dump((memory,auditor),path,compress=3)
    provenance.update(memory_training_blocks=len(X),auditor_observations=len(Z),
                      source_oof_proxy_mse=auditor.oof_mse,model_sha256=sha256(path))
    atomic_json(meta,provenance)
    return memory,auditor,provenance


_WORKER = {}


def _worker_init(model_path):
    """Runs once per spawned worker. 'spawn' (not fork) keeps CUDA state out of
    the workers, so GPU baselines in the parent process are safe."""
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'
    cv2.setNumThreads(1)
    memory, auditor = joblib.load(model_path)
    _WORKER.update(memory=memory, auditor=auditor)


def auto_workers(requested=0):
    if requested and requested > 0:
        return int(requested)
    cpus = os.cpu_count() or 1
    try:
        import psutil
        gb = psutil.virtual_memory().available / 2 ** 30
        by_memory = max(1, int((gb - 2) // 1.5))
    except Exception:
        by_memory = cpus
    return max(1, min(cpus, by_memory, 12))


def make_pool(workers, model_path):
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=_worker_init, initargs=(str(model_path),))


def _neural_worker_init(model_path, official_root, device, include_video, threads, lock):
    """Neural worker: the CPU-worker state plus its own copy of the frozen baselines.
    Model loading (which verifies checksums and rewrites provenance files) is
    serialized with a lock so workers never write the same file concurrently."""
    _worker_init(model_path)
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(20260922)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with lock:
        _WORKER['neural'] = load_baselines(official_root, device, include_video)


def make_neural_pool(workers, model_path, official_root, device, include_video, threads):
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'
    ctx = multiprocessing.get_context('spawn')
    return ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_neural_worker_init,
                               initargs=(str(model_path), str(official_root), device, include_video, threads, ctx.Lock()))


def _neural_execute_job(payload):
    """Same computation as the serial path: _neural_predictions, then _execute_job,
    but inside a worker so several neural jobs run at once. Predictions never leave
    the worker, so no large arrays are pickled between processes."""
    preds, times = _neural_predictions(payload['job'], _WORKER['neural'])
    return _execute_job(dict(payload, predictions=preds, neural_times=times))


def _dev_task(args):
    c, geo, acfg, candidates, regime = args
    memory, auditor = _WORKER['memory'], _WORKER['auditor']
    frames = load_clip(c, geo)
    obs = observations(frames, geo, c.source_group, regime, 0)
    methods = [(-1, acfg, Method('greedy', 'greedy', 'learned', envelope=False, adapt_moment=False,
                                 adapt_weights=False, certificate=False))]
    methods += [(i, a, Method('candidate', 'dual_joint', 'online', envelope=False, certificate=True, twin_audit=True))
                for i, a in enumerate(candidates)]
    rows = []
    for i, a, m in methods:
        s, *_ = run_sequence(frames, obs, geo, a, c.source_group, regime, 0, m, memory, auditor)
        rows.append(dict(candidate=i, source=c.source_group, regime=regime, mse=s['mse'],
                         pqv=s['augmented_pqv'], write_pqv=s['write_pqv']))
    return rows


def _execute_job(payload):
    """Run every method of one (clip, regime, draw, geometry) job and checkpoint it.
    Common random numbers: all methods share the preview noise, patch noise and
    gaze uniforms, so method differences are paired within a job."""
    study, c, regime, draw, g, a, ms = payload['job']
    memory, auditor = _WORKER['memory'], _WORKER['auditor']
    frames = load_clip(c, g)
    obs = observations(frames, g, c.source_group, regime, draw)
    predictions = payload.get('predictions') or {}
    neural_times = payload.get('neural_times') or {}
    rows, kept = [], None
    for method in ms:
        keep = payload['keep'] and method.name == 'PEM_R9_TwinAudit'
        s, trace, _, _, example = run_sequence(frames, obs, g, a, c.source_group, regime, draw, method, memory, auditor,
                                               predictions.get(method.neural), neural_times.get(method.neural), keep=keep)
        s.update(dataset=c.dataset, clip_id=c.clip_id, source_group=c.source_group, role=c.role,
                 regime=regime, draw=draw, study=study, method=method.name,
                 height=g.height, width=g.width, factor=g.factor, fovea=g.fovea,
                 preview_kernel=g.preview_kernel, kappa=a.kappa, write_kappa=a.write_kappa,
                 certificate_kappa=a.certificate_kappa)
        rows.append(s)
        if payload.get('trace_dir'):
            Path(payload['trace_dir']).mkdir(exist_ok=True)
            trace.to_csv(Path(payload['trace_dir']) / (payload['key'] + '_' + method.name + '.csv.gz'),
                         index=False, compression='gzip')
        if keep:
            kept = (trace, example)
    atomic_json(payload['checkpoint'], dict(job_hash=payload['key'], rows=rows))
    return payload['index'], rows, kept


def select_on_development(clips,geo,acfg,out,profile,workers=1):
    path=out/'development_selection.json'
    if path.exists():
        report=json.loads(path.read_text());return MethodConfig(**report['selected']),report
    candidates=[replace(acfg,learning_rate=lr,certificate_kappa=ck,write_kappa=wk)
                for lr in ([.05,.15] if profile=='smoke' else [.05,.15,.4])
                for ck in ([1.5] if profile=='smoke' else [1.15,1.5,2.5])
                for wk in ([2.0] if profile=='smoke' else [1.25,2.0,4.0])]
    dev=[c for c in clips if c.role=='development']
    tasks=[(c,geo,acfg,candidates,regime) for c in dev for regime in ['nominal','noisy']]
    rows=[]
    with make_pool(min(workers,len(tasks)),out/'models.joblib') as pool:
        for k,part in enumerate(pool.map(_dev_task,tasks)):
            rows.extend(part)
            print(f'Development selection {k+1}/{len(tasks)}',flush=True)
    df=pd.DataFrame(rows); grouped=df.groupby('candidate')[['mse','pqv','write_pqv']].mean()
    grouped['dual_audit_score']=np.sqrt(np.maximum(grouped.pqv,1e-30)*np.maximum(grouped.write_pqv,1e-30))
    threshold=1.05*grouped.loc[-1,'mse'];eligible=[i for i in range(len(candidates)) if grouped.loc[i,'mse']<=threshold]
    fallback=not eligible
    selected=min(eligible,key=lambda i:(grouped.loc[i,'dual_audit_score'],i)) if eligible else min(range(len(candidates)),key=lambda i:(grouped.loc[i,'mse'],grouped.loc[i,'dual_audit_score'],i))
    report=dict(selected=asdict(candidates[selected]),selected_candidate=selected,
                candidates=[asdict(c) for c in candidates], reconstruction_threshold=float(threshold),
                rule='Lowest geometric mean of primary-audit PQV and twin-write PQV among candidates with MSE <= 1.05 x greedy development MSE; otherwise lowest MSE, then dual-audit score, then candidate index.',
                fallback_to_reconstruction=fallback,development_sources=sorted(df.source.unique()),
                candidate_scores=grouped.reset_index().to_dict('records'))
    df.to_csv(out/'development_scores.csv',index=False);immutable_json(path,report)
    return candidates[selected],report


def _named(method, name):
    return replace(method, name=name)


def make_jobs(clips,geo,acfg,config):
    """Prespecified evaluation plan.

    Study roles (paper profile):
      main          Long natural videos (REDS val 100 frames, DAVIS legacy 480p) -
                    the primary evidence for temporal memory and auditing.
      reds_train_full   All 240 REDS train_sharp clips (100 frames, unseen by
                    every model) - scale / generalisation evidence.
      reds_train_detail Fixed 60-clip REDS train subset with every controlled policy.
      neural_*      Published SR / video-SR backbones in the identical sensor
                    interface, under the checkpoints' native degradation
                    (clean MATLAB bicubic) and under the paper's sensor model.
      timing        Serial (uncontended) latency measurement.
      sensitivity   kappa, certificate, write-kappa, preview factor/kernel,
                    and a deliberately assumption-violating negative control.
    """
    test=[c for c in clips if c.role in ['external_test','legacy_test']]
    profile=config.profile
    m=core_methods(); by={x.name:x for x in m}
    ours=by['PEM_R9_TwinAudit']
    # Internal predecessor 'R7 reserve' is omitted from reporting studies; it is
    # superseded by the certified variant and adds no information for readers.
    # PEM_R9_single_audit is definitionally identical to PEM_R8_certified (same
    # policy, proxy and certificate; the twin audit is only measured), so it is
    # not reported twice.
    controlled=[x for x in m if x.name not in ('PEM_R7_reserve','PEM_R9_single_audit')]
    jobs=[]
    causal=list(SPECS)+(['BasicVSRpp2022_causal'] if config.include_video_sr else [])
    offline=['BasicVSRpp2022_offline'] if config.include_video_sr else []
    neural_list=[by['PEM_uniform'],by['PEM_R6_fixed'],ours]+neural_methods(causal,offline)

    def add(study,cs,regimes,gs,ms,draws_,ac=acfg):
        for c in cs:
            for regime in regimes:
                for draw in draws_:
                    jobs.append((study,c,regime,draw,gs,ac,ms))

    def sensitivity_block(sens,draws_):
        for k in [1.,2.,4.,8.]:
            add('kappa',sens,['nominal'],geo,[_named(ours,f'PEM_R9_k{k:g}')],draws_,replace(acfg,kappa=k))
        for ck in [1.05,1.25,1.5,2.,4.]:
            add('certificate',sens,['nominal'],geo,[_named(ours,f'PEM_R9_cert{ck:g}')],draws_,replace(acfg,certificate_kappa=ck))
        for wk in [1.,1.25,2.,4.,8.]:
            add('write_kappa',sens,['nominal'],geo,[_named(ours,f'PEM_R9_writek{wk:g}')],draws_,replace(acfg,write_kappa=wk))
        transfer=[by['PEM_uniform'],by['PEM_greedy'],by['PEM_R6_fixed'],by['PEM_R8_certified'],ours]
        add('scale8',sens,['nominal'],replace(geo,factor=8),transfer,draws_)
        add('bicubic',sens,['nominal'],replace(geo,preview_kernel='bicubic'),transfer,draws_)
        add('bias_negative_control',sens,['biased_stress'],geo,[ours],draws_)

    if profile=='paper':
        reds_train=[c for c in test if c.dataset=='REDS_train']
        reds=[c for c in test if c.dataset=='REDS']
        davis=[c for c in test if c.dataset=='DAVIS']
        long_clips=reds+davis
        # 1. Primary evidence: long clips, both sensor regimes, two randomisation draws.
        add('main',long_clips,['nominal','drift'],geo,controlled,range(2))
        # Secondary variant (post-pilot amendment): TwinAudit with write_kappa=1. Not the primary
        # method; TwinAudit at the development-selected write_kappa stays the target of all comparisons.
        add('main',long_clips,['nominal','drift'],geo,[_named(ours,'PEM_R9_TwinAudit_strict')],range(2),replace(acfg,write_kappa=1.))
        # 2. Scale: all 240 REDS train clips (unseen by every model).
        add('reds_train_full',reds_train,['nominal'],geo,
            [by['PEM_greedy'],by['PEM_R6_fixed'],by['PPAT2026_M1'],ours],range(1))
        # 3. All controlled policies on a fixed REDS train subset.
        add('reds_train_detail',_choose(reds_train,60),['nominal'],geo,controlled,range(2))
        if config.run_neural:
            nset=reds+_choose(davis,10)+_choose(reds_train,20)
            add('neural_native',nset,['clean'],replace(geo,preview_kernel='bicubic'),neural_list,range(1))
            add('neural_sensor',nset,['nominal'],geo,neural_list,range(1))
            add('timing',_choose(reds,10),['nominal'],geo,
                [by['PEM_greedy'],by['PEM_R6_fixed'],ours]+neural_methods(causal[-1:],offline),range(1))
        else:
            add('timing',_choose(reds,10),['nominal'],geo,[by['PEM_greedy'],by['PEM_R6_fixed'],ours],range(1))
        sensitivity_block(_choose(reds,15)+_choose(davis,15)+_choose(reds_train,15),range(1))
        if config.high_resolution_reds:
            add('high_resolution',reds,['nominal'],replace(geo,height=448,width=768),
                [by['PEM_uniform'],by['PEM_greedy'],by['PEM_R6_fixed'],ours],range(1))
        return jobs

    draws=range(1 if profile=='smoke' else 2)
    add('main',test,['nominal','drift'],geo,controlled,draws)
    # Secondary variant (post-pilot amendment): TwinAudit with write_kappa=1. Not the primary
    # method; TwinAudit at the development-selected write_kappa stays the target of all comparisons.
    add('main',test,['nominal','drift'],geo,[_named(ours,'PEM_R9_TwinAudit_strict')],draws,replace(acfg,write_kappa=1.))
    per=1 if profile=='smoke' else 12
    subset=[]
    for ds in sorted({c.dataset for c in test}):
        subset.extend(_choose([c for c in test if c.dataset==ds],per))
    if config.run_neural:
        add('neural_native',subset,['clean'],replace(geo,preview_kernel='bicubic'),neural_list,range(1))
        add('neural_sensor',subset,['nominal'],geo,neural_list,range(1))
    add('timing',subset[:max(1,min(len(subset),4))],['nominal'],geo,[by['PEM_greedy'],by['PEM_R6_fixed'],ours],range(1))
    sensitivity_block(subset,draws)
    return jobs

def coverage_table(df,out,reps):
    rng=np.random.default_rng(3519);rows=[]
    keys=CONTEXT+['method']
    for context,g in df.groupby(keys):
        a=g.groupby('source_group')[['cs_simultaneous_covered','cs_final_covered','cs_final_width']].mean()
        row=dict(zip(keys,context));row['source_clusters']=len(a)
        for metric in a:
            values=a[metric].to_numpy();boots=[]
            for start in range(0,reps,256):
                ids=rng.integers(0,len(a),(min(256,reps-start),len(a)))
                boots.extend(values[ids].mean(1))
            row[metric]=float(values.mean());row[metric+'_ci_low'],row[metric+'_ci_high']=np.quantile(boots,[.025,.975])
        row['assumptions_hold']=bool(g.cs_valid_assumptions.all())
        row['interval_limitation']='Percentile source bootstrap can collapse when all repetitions have identical coverage; this is not proof of nominal coverage.'
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'coverage_summary.csv',index=False)


def _job_entry(job):
    study,c,regime,draw,g,a,ms=job
    return dict(study=study,dataset=c.dataset,clip_id=c.clip_id,source_group=c.source_group,
                role=c.role,content_sha256=c.content_sha256,regime=regime,draw=draw,
                geometry=asdict(g),audit_config=asdict(a),methods=[asdict(m) for m in ms])


def _seed_torch(*parts):
    import torch
    from video_core import seed
    torch.manual_seed(seed(*parts))


def _neural_predictions(job,neural):
    study,c,regime,draw,g,a,ms=job
    names=sorted({m.neural for m in ms if m.neural})
    if not names: return {},{}
    frames=load_clip(c,g); pv=previews(frames,g,c.source_group,regime,draw)
    preds,times={},{}
    for name in names:
        # The official SCMSR code samples (gumbel_softmax) even at inference, so its output
        # depends on the global PyTorch RNG. Seed it per clip, condition and model so predictions
        # do not depend on the process, job order, parallel scheduling or resume history.
        _seed_torch(g.seed,c.dataset,c.clip_id,regime,draw,g.preview_kernel,g.factor,name)
        pp,tt=neural[name].predict_sequence(pv); preds[name]=list(pp); times[name]=list(tt)
    return preds,times


def run_pipeline(config,split):
    config.root=str(Path(config.root).expanduser().resolve());root=Path(config.root)
    root.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(config.threads)
    if config.data_root: config.data_root=str(Path(config.data_root).expanduser().resolve())
    workers=auto_workers(config.workers)
    if config.run_neural:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        try:
            import torch
        except ImportError:
            if config.profile!='smoke': raise
            torch=None
    if config.run_neural and torch is not None:
        torch.set_num_threads(config.threads)
        torch.manual_seed(20260922)
        # warn_only: inference-only ops are deterministic; this avoids a hard
        # failure on kernels that merely lack a deterministic *backward*.
        torch.use_deterministic_algorithms(True,warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();clips,geo,data_report=prepare_data(config,split)
    versions=environment()
    lock_config={k:v for k,v in asdict(config).items() if k not in ['root','data_root','dataset_roots','threads','device','download_data','export_figures','line_dpi','workers','save_all_traces','neural_workers','neural_threads']}
    base_lock=dict(config=lock_config,geometry=asdict(geo),method_defaults=asdict(MethodConfig()),
        clips=[_lockable_clip(c) for c in clips],code=source_hashes(),environment=_stable_environment(versions),
        official_specs=SPECS if config.run_neural else {},legacy_splits=split)
    fingerprint=digest_json(base_lock)
    out=root/'results'/(config.experiment_id+'_'+fingerprint[:12]);out.mkdir(parents=True,exist_ok=True)
    immutable_json(out/'inputs.lock.json',base_lock);atomic_json(out/'data_integrity.json',data_report)
    print('Experiment:',out.name,'| sources:',len({c.source_group for c in clips}),'|',config.profile,'| CPU workers:',workers,flush=True)
    memory,auditor,training=fit_models(clips,geo,MethodConfig(),out,config.profile)
    acfg,selection=select_on_development(clips,geo,MethodConfig(),out,config.profile,workers)
    neural={}; sanity={}
    if config.run_neural:
        neural=load_baselines(root/'official_models',config.device,config.include_video_sr)
        if config.profile!='smoke':
            dev=sorted([c for c in clips if c.role=='development'],key=lambda c:c.clip_id)[:2]
            sanity=baseline_sanity(neural,[load_clip(c,replace(geo,frames=10)) for c in dev],geo.factor)
            atomic_json(out/'baseline_sanity.json',sanity)
            print('Baseline sanity gate passed:',{k:round(v['mean_psnr_gain_over_bicubic_db'],2) for k,v in sanity.items()},flush=True)
    jobs=make_jobs(clips,geo,acfg,config)
    job_entries=[_job_entry(j) for j in jobs]
    locked=dict(input_fingerprint=fingerprint,training=training,selection=selection,
                official_models={n:m.provenance for n,m in neural.items()},jobs=job_entries,
                expected_jobs=len(jobs),expected_rows=sum(len(j[-1]) for j in jobs),
                inference='Paired source-cluster resampling; equal source weights; primary endpoints reconstruction MSE, memory-benefit audit PQV, and one-step write-value audit PQV.',
                historical_disclosure='The 40 legacy DAVIS test sources were seen in earlier revisions. REDS train_sharp and REDS val_sharp were never used for design, fitting or selection; no model in this study is trained on REDS (PEM models fit on DAVIS fit sources; BasicVSR++ checkpoint trained on the Vimeo-90K training split; NTIRE 2025 ESR checkpoints are still-image models).',
                baseline_disclosure='NTIRE 2025 ESR checkpoints and the BasicVSR++ Vimeo-90K BI checkpoint are external author weights evaluated without fine-tuning. The BasicVSR++ offline variant reads future previews and is a non-causal reference only.',
                negative_control='biased_stress intentionally violates zero-mean noise; no CS guarantee applies.',
                post_pilot_amendment='Added after the laptop pilot (local_01_677c295da848) and before the paper run: paired same-backbone comparisons in the neural studies, backbone + TwinAudit gaze (_memory_R9) vs backbone + memory (_memory) and vs backbone + patch (_patch), for PSNR, MSE, audit PQV and write PQV, using the same source-cluster bootstrap, sign-flip test and Holm adjustment (within study and metric). Also added at the same time: a secondary variant PEM_R9_TwinAudit_strict (TwinAudit with write_kappa = 1, displayed as "TwinAudit-strict (κw = 1)") in the main study only. It is a secondary variant, NOT the primary method: TwinAudit with the development-selected write_kappa = 2 remains the primary method and the target of all paired comparisons, and the development selection was not changed. Speed-only changes: THREADS raised from 2 to 8, and neural baseline predictions computed in parallel worker processes (each job computed by the same functions as before; equivalence verified before the paper run). Reproducibility fix: the official SCMSR code samples gumbel_softmax even at inference, so the global PyTorch RNG is now seeded per clip, condition and model before each baseline prediction (and per model in the sanity gate); SCMSR numbers therefore differ slightly from the laptop pilot, where its random draws depended on call order. No method, selection rule or existing statistic was changed.')
    immutable_json(out/'protocol.lock.json',locked)
    plan={}
    for job in jobs:
        plan[job[0]]=plan.get(job[0],0)+len(job[-1])
    print('Planned method runs:',plan,flush=True)
    frames_planned=sum(len(j[-1])*min(len(j[1].paths),j[4].frames) for j in jobs)
    print(f'Planned frame-method evaluations: {frames_planned:,}',flush=True)
    checkpoint=out/'checkpoints';checkpoint.mkdir(exist_ok=True)
    trace_dir=str(out/'traces') if config.save_all_traces else None
    keep_index=next((i for i,j in enumerate(jobs) if j[0]=='main' and j[2]=='nominal' and j[3]==0
                     and any(m.name=='PEM_R9_TwinAudit' for m in j[-1])),None)
    if (out/'example.npz').exists(): keep_index=None
    results={};pending=[]
    for index,(job,entry) in enumerate(zip(jobs,job_entries)):
        key=digest_json(dict(job=entry,input=fingerprint,training=training['model_sha256'],selection=asdict(acfg)))
        cp=checkpoint/(key+'.json')
        if cp.exists():
            saved=json.loads(cp.read_text())
            if saved['job_hash']!=key or len(saved['rows'])!=len(job[-1]):raise ValueError('Invalid checkpoint')
            results[index]=saved['rows']
        else:
            pending.append(dict(index=index,job=job,key=key,checkpoint=str(cp),keep=index==keep_index,trace_dir=trace_dir))
    print(f'Locked {len(jobs)} jobs / {locked["expected_rows"]} method-sequence runs; {len(results)} jobs already complete. Restarting resumes.',flush=True)
    kept=None;session_start=time.perf_counter();done=0
    def record(res):
        nonlocal kept,done
        i,rows,k=res;results[i]=rows;done+=1
        if k is not None:
            trace,example=k;trace.to_csv(out/'example_trace.csv',index=False)
            np.savez_compressed(out/'example.npz',**example);atomic_json(out/'example_selection.json',job_entries[i])
            kept=k
        if done%25==0 or done==len(pending):
            el=time.perf_counter()-session_start
            print(f'Completed {len(results)}/{len(jobs)} jobs; session {el/3600:.2f} h; rough ETA {el/done*(len(pending)-done)/3600:.2f} h',flush=True)
    needs_neural=lambda p:any(m.neural for m in p['job'][-1])
    serial=[p for p in pending if p['job'][0]=='timing']
    neural_jobs=[p for p in pending if p['job'][0]!='timing' and needs_neural(p)]
    cpu_jobs=[p for p in pending if p['job'][0]!='timing' and not needs_neural(p)]
    if neural_jobs:
        # Neural baselines: each worker computes the SR predictions for one job and runs
        # that job, so several neural jobs proceed in parallel (speed only).
        nw=max(1,min(config.neural_workers,len(neural_jobs)))
        print(f'Neural jobs: {len(neural_jobs)} on {nw} workers x {config.neural_threads} PyTorch threads',flush=True)
        with make_neural_pool(nw,out/'models.joblib',root/'official_models',config.device,
                              config.include_video_sr,config.neural_threads) as pool:
            inflight={pool.submit(_neural_execute_job,p) for p in neural_jobs}
            while inflight:
                finished,inflight=wait(inflight,return_when=FIRST_COMPLETED)
                for f in finished: record(f.result())
    if cpu_jobs:
        with make_pool(workers,out/'models.joblib') as pool:
            inflight={pool.submit(_execute_job,p) for p in cpu_jobs}
            while inflight:
                finished,inflight=wait(inflight,return_when=FIRST_COMPLETED)
                for f in finished: record(f.result())
    if serial:
        # Latency study: executed alone in this process, with no competing workers.
        _WORKER.update(memory=memory,auditor=auditor)
        for p in serial:
            preds,times=_neural_predictions(p['job'],neural) if needs_neural(p) else ({},{})
            record(_execute_job(dict(p,predictions=preds,neural_times=times)))
    df=pd.DataFrame([row for i in range(len(jobs)) for row in results[i]])
    if len(df)!=locked['expected_rows']:raise AssertionError('Incomplete protocol')
    if df.duplicated(CONTEXT+['method','clip_id','draw']).any():raise AssertionError('Duplicate experiment row')
    required=['mse','ssim','augmented_pqv','augmented_squared_error','write_pqv','write_squared_error','online_total_ms_mean']
    if not np.isfinite(df[required].to_numpy()).all():raise AssertionError('Nonfinite scientific result')
    reps=1000 if config.profile=='smoke' else 6000
    summary,paired,controls=export_analysis(df,out,config.profile=='smoke',reps)
    coverage_table(df,out,reps)
    tables=paper_tables(summary,paired,out/'tables',neural_provenance={n:m.provenance for n,m in neural.items()})
    backbone=export_backbone_comparisons(df,out,config.profile=='smoke',reps)
    example_saved=(out/'example.npz').exists()
    if config.export_figures:
        publication_figures(df,summary,paired,controls,out/'figures',config.profile=='smoke',config.line_dpi)
        if example_saved:
            if kept is None:kept=(pd.read_csv(out/'example_trace.csv'),dict(np.load(out/'example.npz',allow_pickle=True)))
            trace_figures(*kept,out/'figures',config.profile=='smoke')
    import psutil
    runtime=dict(elapsed_seconds=time.perf_counter()-start,process_rss_at_end_bytes=psutil.Process().memory_info().rss,
                 cpu_workers=workers,
                 timing_scope="Report latency ONLY from study == 'timing', which runs serially without competing processes. Per-frame times in other studies were measured under parallel load. Times include preview/flow, controller/audit and SR with device transfers; evaluator, disk IO, fitting and downloads are excluded.",
                 reused_neural_predictions='SR predictions are computed once per common preview, but measured SR time is charged to every method using them.',
                 peak_cuda_allocated_bytes=None)
    if config.run_neural:
        try:
            import torch
            if torch.cuda.is_available():runtime['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        except ImportError: pass
    atomic_json(out/'runtime.json',runtime)
    complete_paper=config.profile=='paper' and config.run_neural
    readiness=dict(protocol_complete=True,profile=config.profile,rows=len(df),
        large_scale_protocol_executed=complete_paper,submission_ready=False,
        baselines_executed=sorted(neural),
        blockers=['Scientific superiority and practical gains must be read from the actual tables, including negative results.',
                  'Neural baselines are external checkpoints evaluated in a common one-patch sensor interface (native bicubic and sensor-noise conditions); they are not their original benchmark protocols.',
                  'CRFP (foveated VSR) and AdaGlimpse/AME (active exploration) were not executed: their action spaces or released weights do not fit this fixed-grid interface without retraining. The manuscript must state this and compare against them conceptually and via the fixed-grid uncertainty baselines.',
                  'Manuscript rewrite, English editing and Elsevier IVC formatting remain necessary.']+
                 ([] if complete_paper else ['The complete paper profile on natural videos with neural baselines has not been executed in this run.']))
    atomic_json(out/'readiness.json',readiness)
    archive=root/(out.name+'_evidence.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            rel=p.relative_to(out)
            if p.is_file() and 'checkpoints' not in rel.parts:
                z.write(p,'results/'+str(rel))
        for p in sorted(Path(__file__).parent.glob('*.py')):z.write(p,'source/'+p.name)
        for folder in sorted((root/'official_models').glob('*')) if (root/'official_models').exists() else []:
            for p in folder.glob('*'):
                if p.suffix in ['.json','.py'] or p.name=='LICENSE':z.write(p,'official_models/'+folder.name+'/'+p.name)
    print('FINISHED:',archive,flush=True)
    return dict(output=str(out),archive=str(archive),summary=summary,paired=paired,controls=controls,
                tables=tables,readiness=readiness,sequence_results=df,backbone=backbone)
