"""Implementation checks."""
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
from r9_core import (MethodConfig, constrained_gaze, lower_normalize, audit_gradient,
                     exact_variance, NormalMixtureCS, fixed_pool_ppat, envelope_lower,
                     certificate_lower, policy_distribution, variance_quadratic, best_fixed_mixture,
                     subgaussian_proxy, dual_audit_ray_gaze)
from r9_engine import Geometry, Method, core_methods, observations, run_sequence
from r9_data import synthetic_frames


def dataset_reader_checks():
    """Exercise the DAVIS and REDS (val and train) directory readers with labeled image fixtures."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import cv2
    from r9_data import discover_davis, discover_reds, validate_manifest, load_clip
    cfg=Geometry(height=64,width=96,fovea=16,frames=7)
    with TemporaryDirectory() as directory:
        root=Path(directory)
        for folder,kind,value in [(root/'DAVIS/JPEGImages/480p/a','davis',50),
                                  (root/'REDS/val_sharp/000','reds',90),
                                  (root/'REDS/val_sharp/001','reds',130),
                                  (root/'REDS/train_sharp/000','reds',180),
                                  (root/'REDS/train_sharp/001','reds',220)]:
            folder.mkdir(parents=True)
            for t in range(7):
                name=f'{t:08d}.'+('jpg' if kind=='davis' else 'png')
                assert cv2.imwrite(str(folder/name),np.full((64,96,3),value+t,np.uint8))
        ds=discover_davis(root/'DAVIS',{'fit':['a']},7)
        rs=discover_reds(root/'REDS/val_sharp',frames=7,dataset='REDS',expected=2)
        ts=discover_reds(root/'REDS/train_sharp',frames=7,dataset='REDS_train',expected=2)
        assert [c.clip_id for c in rs]==[c.clip_id for c in ts]==['000','001']
        allclips,report=validate_manifest(ds+rs+ts)
        assert len({c.source_group for c in allclips})==5, 'REDS train and val identities collided'
        for c in allclips:
            x=load_clip(c,cfg);assert x.shape==(7,64,96,3) and np.isfinite(x).all()
        try: discover_reds(root/'REDS/train_sharp',frames=7,dataset='REDS_train',expected=3)
        except ValueError: pass
        else: raise AssertionError('An incomplete external split was accepted')
        try: discover_reds(root/'REDS',frames=7,dataset='REDS')
        except ValueError: pass
        else: raise AssertionError('Mixed REDS train/val folders were accepted as one split')
        try: validate_manifest(allclips+[replace(ds[0],clip_id='alias',role='external_test')])
        except AssertionError: pass
        else: raise AssertionError('Source leakage was accepted')
    return dict(davis_reader=True,reds_val_reader=True,reds_train_reader=True,
                reds_train_val_ids_distinct=True,incomplete_split_rejected=True,
                mixed_reds_splits_rejected=True,source_leakage_rejected=True,
                scope='directory/image software fixtures, not downloaded natural benchmarks')


def mathematical_checks():
    rng = np.random.default_rng(451)
    unbiased=[]; gradient=[]; solver=[]; psd=[]; envelopes=[]; certificates=[]; quadratics=[]; dual_ratios=[]
    for _ in range(20):
        n = 12; r = rng.lognormal(size=n); v = rng.lognormal(size=n)
        lower = .05/n + .15*rng.dirichlet(np.ones(n))
        ref = lower+(1-lower.sum())/n; cap = 2*float(np.sum(v/ref))
        q = constrained_gaze(r, v, lower, 2.)
        opt = minimize(lambda p: -r@p, ref, method='SLSQP', bounds=list(zip(lower,np.ones(n))),
            constraints=[{'type':'eq','fun':lambda p:p.sum()-1},
                         {'type':'ineq','fun':lambda p:1-np.sum(v/p)/cap}],
            options={'ftol':1e-10,'maxiter':500})
        if not opt.success:
            raise AssertionError('Independent solver check failed: '+opt.message)
        solver.append(abs(float(r@q-r@opt.x)))
        assert np.all(q>=lower-1e-12) and np.sum(v/q)<=cap*(1+1e-8)
        d = rng.uniform(-.2,.2,n); M = rng.uniform(-.2,.2,(n,3)); w = np.array([.2,.3,.5]); mu=M@w
        est = mu.mean()+(d-mu)/(n*q)
        unbiased.append(abs(float(q@est-d.mean())))
        g = sum(q[a]*audit_gradient(M,w,q,a,d[a]) for a in range(n))
        numerical = []
        for k in range(3):
            e = np.eye(3)[k]*1e-5
            numerical.append((exact_variance(d,M@(w+e),q,np.zeros(n))-exact_variance(d,M@(w-e),q,np.zeros(n)))/2e-5)
        gradient.append(float(np.max(abs(g-numerical))))
        hessian = 2*(M.T@(M/q[:,None])/n**2-np.outer(M.mean(0),M.mean(0)))
        psd.append(float(np.linalg.eigvalsh(hessian).min()))
        # Analytic envelope cap remains valid even with inaccurate learned moments.
        acfg=MethodConfig(); low=d-rng.uniform(0,.3,n); high=d+rng.uniform(0,.3,n)
        tau=rng.uniform(0,.001,n); l=envelope_lower(mu,low,high,tau,acfg)
        qq=constrained_gaze(r,v,l,acfg.kappa)
        c=np.maximum(abs(low-mu),abs(high-mu))+np.sqrt(tau)
        B=c.sum()/(n*(1-acfg.epsilon)*acfg.envelope_reserve)
        envelopes.append(subgaussian_proxy(qq,mu,low,high,tau)/(2*B*B))
        assert np.max(c/(n*qq))<=B+1e-12 and envelopes[-1]<=1+1e-12
        # R9 hard certificate: independent of learned second-moment calibration.
        cl, bref, gamma = certificate_lower(mu, low, high, tau, acfg)
        cq = constrained_gaze(r, v, cl, acfg.kappa)
        certificates.append(float(np.max(c/(n*cq))/gamma))
        assert certificates[-1] <= 1 + 1e-10
        # R9 simultaneous two-audit ray allocation.  The reference is feasible
        # by construction, and the returned point must satisfy both caps.
        v2=rng.lognormal(size=n)
        q2,alpha,rr1,rr2=dual_audit_ray_gaze(r,v,v2,cl,1.7,2.3)
        assert np.isclose(q2.sum(),1) and np.all(q2>=cl-1e-12)
        assert rr1<=1.7+1e-8 and rr2<=2.3+1e-8 and 0<=alpha<=1
        dual_ratios.append((rr1,rr2,alpha))
        # Quadratic hindsight objective exactly reproduces exact_variance.
        qa,qb,qc = variance_quadratic(M,d,q,np.zeros(n))
        quadratics.append(abs(float(w@qa@w+2*qb@w+qc)-exact_variance(d,M@w,q,np.zeros(n))))
    assert max(solver)<2e-6 and max(unbiased)<1e-12 and max(gradient)<1e-8 and min(psd)>-1e-10
    n = 14; d=rng.uniform(-1,1,n); mu=rng.uniform(-1,1,n)
    result = [fixed_pool_ppat(d,mu,np.ones(n),5,rng) for _ in range(12000)]
    result = np.asarray(result); mean_error=result.mean(0)-d.mean()
    assert np.all(abs(mean_error)<5*result.std(0)/np.sqrt(len(result)))
    census=fixed_pool_ppat(d,mu,np.ones(n),n,rng)
    assert np.allclose(census,d.mean())
    # Regression test for the supplied R7 bug: risk-only must use risk, not benefit.
    cfg=MethodConfig(); benefit=np.linspace(1,10,n); risk=np.linspace(10,1,n)
    dummy=np.ones(n); experts=np.zeros((n,3)); lower=np.full(n,cfg.epsilon/n)
    qrisk=policy_distribution('risk_only',benefit,risk,dummy,experts,lower,cfg)
    assert int(np.argmax(qrisk)) == int(np.argmax(risk)) != int(np.argmax(benefit))
    # Gaussian contrast noise: use identical noise in the two loss terms.
    x=rng.uniform(size=(20000,30)); b=rng.uniform(size=x.shape); m=rng.uniform(size=x.shape)
    noise=rng.normal(0,.05,x.shape)
    err=(((x+noise-b)**2-(x+noise-m)**2)-((x-b)**2-(x-m)**2)).mean(1)
    assert abs(err.mean())<5*err.std()/np.sqrt(len(err))
    return dict(max_solver_objective_gap=max(solver), max_unbiasedness_error=max(unbiased),
                max_gradient_error=max(gradient), min_variance_hessian_eigenvalue=min(psd),
                fixed_pool_ppat_trials=12000, native_ppat_mean_errors=mean_error.tolist(),
                gaussian_noise_cancellation_error=float(err.mean()), max_envelope_bound_ratio=max(envelopes),
                max_hard_certificate_ratio=max(certificates), max_quadratic_identity_error=max(quadratics),
                risk_only_regression_fixed=True,
                max_dual_primary_ratio=max(x[0] for x in dual_ratios),
                max_dual_write_ratio=max(x[1] for x in dual_ratios),
                max_dual_reference_mix=max(x[2] for x in dual_ratios))


def information_boundary_checks(memory=None, auditor=None):
    cfg=Geometry(height=64,width=96,fovea=16,block=8,factor=4,frames=10)
    acfg=MethodConfig(); method=Method('PEM_R9_TwinAudit','dual_joint','online',envelope=False,certificate=True,twin_audit=True)
    frames=synthetic_frames(cfg); obs=observations(frames,cfg,'boundary','nominal',0)
    a=run_sequence(frames,obs,cfg,acfg,'boundary','nominal',0,method,memory,auditor,keep=True)
    changed=frames.copy(); nx=cfg.width//cfg.fovea;s=cfg.fovea
    for t,row in a[1].iterrows():
        act=int(row.action); yy,xx=act//nx*s,act%nx*s
        mask=np.ones((cfg.height,cfg.width),bool);mask[yy:yy+s,xx:xx+s]=False
        changed[t][mask]=1-changed[t][mask]
    b=run_sequence(changed,obs,cfg,acfg,'boundary','nominal',0,method,memory,auditor,keep=True)
    keys=['action','propensity','augmented','write_augmented','weight_zero','weight_energy','weight_learned']
    assert np.allclose(a[1][keys].values,b[1][keys].values,equal_nan=True)
    assert np.array_equal(a[4]['reconstruction'],b[4]['reconstruction'])
    assert abs(a[0]['mse']-b[0]['mse'])>1e-5
    future=frames.copy();future[5:]=1-future[5:]
    future_obs=observations(future,cfg,'boundary','nominal',0)
    c=run_sequence(future,future_obs,cfg,acfg,'boundary','nominal',0,method,memory,auditor)
    assert np.allclose(a[1][keys].values[:5],c[1][keys].values[:5],equal_nan=True)
    for m in core_methods():
        _,t,_,_,_=run_sequence(frames,obs,cfg,acfg,'boundary','nominal',0,m,memory,auditor)
        assert np.isfinite(t[['augmented','mse','ssim','cs_low','cs_high']]).all().all()
        if m.twin_audit and t.write_valid.any():
            assert np.isfinite(t.loc[t.write_valid,['write_augmented','exact_variance_write']]).all().all()
        if m.policy=='uniform': assert np.allclose(t.naive,t.ipw)
        if m.policy=='ppat_2026_m1': assert np.allclose(t.augmented,t.ppat_m1)
    return dict(unqueried_pixels_invariant=True, future_frames_invariant=True,
                predictable_weight_timing=True, all_core_methods_executed=True,
                uniform_ipw_identity=True, native_one_query_ppat_identity=True,
                twin_state_counterfactual_executed=True, dual_audit_budget_checked=True)


def coverage_simulation(trials=400, frames=80):
    """Adaptive synthetic process; validates code, not natural-video performance."""
    rng=np.random.default_rng(1024); good=[]; final=[]
    for _ in range(trials):
        cs=NormalMixtureCS(.05,.01); truth=0.; ok=True
        for t in range(frames):
            # Bounded, time-varying predictable mean plus Gaussian noise.
            mean=.1*np.sin(t/5)+.04*np.tanh(cs.total)
            variance=.03**2*(1+abs(mean))
            y=mean+rng.normal(0,np.sqrt(variance));truth+=mean
            lo,hi,_=cs.update(y,variance)
            ok=ok and lo<=truth/(t+1)<=hi
        good.append(ok);final.append(hi-lo)
    rate=float(np.mean(good))
    assert rate>.90, 'Large coverage deficit in CS implementation simulation'
    return dict(trials=trials, frames=frames, simultaneous_coverage=rate,
                mean_final_width=float(np.mean(final)), nominal_coverage=.95,
                scope='synthetic adaptive Gaussian software diagnostic')


def baseline_interface_checks():
    """Torch-free checks of the SR/VSR interface: MATLAB-bicubic preview and
    strict causality of the windowed video-SR wrapper."""
    from video_core import matlab_bicubic_down, down
    from r9_neural import TemporalWindowSR
    const=np.full((64,112,3),.37,np.float32)
    assert np.allclose(matlab_bicubic_down(const,4),.37,atol=1e-6)
    ramp=np.tile(np.linspace(0,1,448,dtype=np.float32)[None,:,None],(64,1,3))
    assert np.allclose(matlab_bicubic_down(ramp,4)[:,2:-2],down(ramp,4)[:,2:-2],atol=1e-5)
    class Fake(TemporalWindowSR):
        def __init__(self,mode):
            self.mode=mode;self.causal=mode=='causal';self.window=3;self.device='cpu'
        def sync(self):pass
        def _run(self,stack):
            c=np.cumsum(stack,0)[::-1].cumsum(0)[::-1]  # depends on past AND future within the input
            return np.repeat(np.repeat(c,4,1),4,2)
    rng=np.random.default_rng(5)
    a=[rng.uniform(size=(8,8,3)).astype(np.float32) for _ in range(9)]
    b=a[:5]+[1-x for x in a[5:]]
    pa,_=Fake('causal').predict_sequence(a); pb,_=Fake('causal').predict_sequence(b)
    assert all(np.array_equal(x,y) for x,y in zip(pa[:5],pb[:5])), 'Causal VSR leaked future previews'
    oa,_=Fake('offline').predict_sequence(a); ob,_=Fake('offline').predict_sequence(b)
    assert not np.array_equal(oa[0],ob[0]), 'Offline reference should read the whole clip'
    return dict(matlab_bicubic_constant=True,matlab_bicubic_ramp=True,causal_vsr_future_invariant=True,
                offline_vsr_flagged_noncausal=True)
