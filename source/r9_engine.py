"""Sequence experiments."""
from dataclasses import dataclass, replace
import time
import cv2
import numpy as np
from video_core import (seed, down, base_image, detail, blocks, expand, remap, gray,
                        feature_maps, FEATURES, matlab_bicubic_down)
from r9_core import (MethodConfig, PredictableMixture, NormalMixtureCS, contrast_bounds,
    envelope_lower, certificate_lower, policy_distribution, subgaussian_proxy, exact_variance,
    variance_quadratic, best_fixed_mixture, dual_audit_ray_gaze)


@dataclass(frozen=True)
class Geometry:
    height: int = 256
    width: int = 448
    factor: int = 4
    block: int = 8
    fovea: int = 64
    frames: int = 100
    stride: int = 1
    budget: int = 1
    seed: int = 20260922
    preview_kernel: str = 'area'

    def __post_init__(self):
        for dim in [self.height, self.width]:
            assert dim % self.fovea == 0 and dim % self.factor == 0 and dim % self.block == 0
        assert self.fovea % self.factor == 0 and self.fovea % self.block == 0
        assert self.budget == 1 and self.frames >= 2
        assert self.preview_kernel in ['area', 'bicubic']


@dataclass(frozen=True)
class Method:
    name: str
    policy: str = 'joint'
    control: str = 'online'
    memory_mode: str = 'pem'
    neural: str = ''
    envelope: bool = True
    adapt_moment: bool = True
    adapt_weights: bool = True
    certificate: bool = True
    twin_audit: bool = False


def core_methods():
    """Equal-budget controlled baselines plus direct R8 and R9 ablations."""
    return [
        Method('PEM_uniform', 'uniform', 'learned', envelope=False, adapt_moment=False, adapt_weights=False, certificate=False),
        Method('PEM_greedy', 'greedy', 'learned', envelope=False, adapt_moment=False, adapt_weights=False, certificate=False),
        Method('PEM_softmax', 'softmax', 'learned', envelope=False, adapt_moment=False, adapt_weights=False, certificate=False),
        Method('PEM_risk_only', 'risk_only', 'learned', envelope=False, adapt_moment=False, adapt_weights=False, certificate=False),
        Method('PEM_audit_only', 'audit_only', 'learned', envelope=False, adapt_moment=False, adapt_weights=False, certificate=False),
        Method('PEM_R6_fixed', 'joint', 'learned', envelope=False, adapt_moment=False, adapt_weights=False, certificate=False),
        Method('PPAT2026_M1', 'ppat_2026_m1', 'learned', envelope=False, adapt_moment=False, adapt_weights=False, certificate=False),
        Method('PEM_R7_reserve', 'joint', 'online', envelope=True, certificate=False),
        # R8 predecessor.
        Method('PEM_R8_certified', 'joint', 'online', envelope=False, certificate=True, twin_audit=True),
        # Twin audit measured but not used for gaze.
        Method('PEM_R9_single_audit', 'joint', 'online', envelope=False, certificate=True, twin_audit=True),
        # Dual audit with fixed proxy weights.
        Method('PEM_R9_dual_fixed', 'dual_joint', 'learned', envelope=False, adapt_moment=False, adapt_weights=False, certificate=True, twin_audit=True),
        # Full TwinAudit method.
        Method('PEM_R9_TwinAudit', 'dual_joint', 'online', envelope=False, certificate=True, twin_audit=True),
    ]


def neural_methods(causal_names, offline_names=()):
    """Published SR baselines inside the identical one-patch sensor interface.

    _patch: current SR prediction plus the paid foveal patch (uniform gaze).
    _memory: the same plus the identical causal residual memory.
    _memory_ours: the same SR backbone controlled by the proposed dual-audit policy.
    Offline (bidirectional) video SR reads future previews; it is reported only as a
    non-causal reference and never as a deployable competitor.
    """
    out=[]
    for n in causal_names:
        out.extend([
            Method(n+'_patch','uniform','learned','sr',n,False,False,False,False,False),
            Method(n+'_memory','uniform','learned','sr_memory',n,False,False,False,False,True),
            Method(n+'_memory_R9','dual_joint','online','sr_memory',n,False,True,True,True,True),
        ])
    for n in offline_names:
        out.append(Method(n+'_patch','uniform','learned','sr',n,False,False,False,False,False))
    return out

def sensor_parameters(regime, t, total, tile=None):
    if regime == 'clean': return 0., 0., 0., 0.
    if regime == 'nominal': return .005, .01, 0., 0.
    if regime == 'noisy': return .012, .025, 0., 0.
    if regime in ['drift', 'biased_stress']:
        if t < total // 3: p, f, blur = .003, .008, 0.
        elif t < 2 * total // 3: p, f, blur = .030, .060, .8
        else: p, f, blur = .010, .020, 0.
        if tile is not None:
            f *= .5 + (tile % 5) / 4
        bias = .05 if regime == 'biased_stress' and t >= total // 3 else 0.
        return p, f, blur, bias
    raise ValueError(regime)


def preview_frame(frame, cfg, name, regime, draw, t, total):
    """The coarse noisy preview p_t. Shared by the controller and SR baselines so
    every method sees exactly the same low-resolution input."""
    ps, _, blur, _ = sensor_parameters(regime, t, total)
    optical = cv2.GaussianBlur(frame, (0, 0), blur) if blur else frame
    if cfg.preview_kernel == 'area':
        p = down(optical, cfg.factor)
    else:
        # Antialiased MATLAB-style bicubic: the native degradation of BI-trained SR checkpoints.
        p = matlab_bicubic_down(optical, cfg.factor)
    rng = np.random.default_rng(seed(cfg.seed, name, regime, draw, t, 'preview'))
    return (p + rng.normal(0, ps, p.shape).astype(np.float32)).astype(np.float32)


def previews(frames, cfg, name, regime, draw):
    return [preview_frame(f, cfg, name, regime, draw, t, len(frames)) for t, f in enumerate(frames)]


def observations(frames, cfg, name, regime, draw):
    """Only preview pixels enter optical flow and preview-derived features."""
    result, previous = [], None
    for t, frame in enumerate(frames):
        start = time.perf_counter()
        p = preview_frame(frame, cfg, name, regime, draw, t, len(frames))
        b = base_image(p, cfg.factor)
        g = np.clip(cv2.GaussianBlur(gray(b), (0, 0), 1) * 255, 0, 255).astype(np.uint8)
        if previous is None:
            flow = np.zeros((*g.shape, 2), np.float32)
            fb = np.zeros(g.shape, np.float32); innovation = fb.copy()
        else:
            flow = cv2.calcOpticalFlowFarneback(g, previous[0], None, .5, 3, 21, 3, 5, 1.2, 0)
            reverse = cv2.calcOpticalFlowFarneback(previous[0], g, None, .5, 3, 21, 3, 5, 1.2, 0)
            fb = np.sum((flow + remap(reverse, flow)) ** 2, axis=2)
            innovation = np.mean((b - remap(previous[1], flow)) ** 2, axis=2)
        pg = gray(p); rr = pg - cv2.GaussianBlur(pg, (0, 0), .7)
        noise = max(float(np.median(abs(rr - np.median(rr))) / .67448975), .0005)
        result.append(dict(p=p, b=b, flow=flow, fb=fb, innovation=innovation, noise=noise,
                           preview_ms=1000*(time.perf_counter()-start)))
        previous = (g, b)
    return result


class PatchSensor:
    def __init__(self, frames, cfg, name, regime, draw):
        self._frames = frames; self.cfg = cfg; self.name = name
        self.regime = regime; self.draw = draw; self.reads = 0
        self._seen = set()

    def acquire(self, t, action):
        c = self.cfg; s = c.fovea; nx = c.width // s; j = c.height // s * nx
        if t in self._seen or not 0 <= action < j:
            raise ValueError('One valid foveal query per frame is required')
        self._seen.add(t); self.reads += 1
        y, x = action // nx * s, action % nx * s
        sig, bias = sensor_parameters(self.regime, t, len(self._frames), action)[1::2]
        rng = np.random.default_rng(seed(c.seed, self.name, self.regime, self.draw, t, action, 'fovea'))
        clean = self._frames[t, y:y+s, x:x+s]
        # Keep the noisy observation UNCLIPPED for the cancellation identity.
        patch = (clean + bias + rng.normal(0, sig, clean.shape)).astype(np.float32)
        return patch, (y, x)


def tile_features(X, pred, base, risk, cfg):
    h, w, k, s = cfg.height, cfg.width, cfg.block, cfg.fovea
    x = X.reshape(h // s, s // k, w // s, s // k, -1)
    energy = blocks((pred - base) ** 2, s)
    risk = blocks(expand(risk, k), s)
    extra = np.stack([np.log1p(energy / 1e-6), np.log1p(np.maximum(risk, 0) / 1e-6)], axis=-1)
    return np.concatenate([x.mean((1, 3)), x.std((1, 3)), extra], axis=-1).reshape(-1, 2*len(FEATURES)+2)


class Controller:
    """Controller with a one-step counterfactual shadow memory.

    The deployed memory is always updated with the paid patch.  Separately, we
    retain a copy of the *pre-write* state and propagate it for exactly one
    frame.  At t+1 this produces a shadow prediction that answers a concrete
    counterfactual: what would the current prediction have been if the previous
    foveal write had been omitted, holding the executed gaze history and the
    current preview fixed?  The shadow never influences the deployed memory.
    """
    def __init__(self,cfg,method,memory,auditor):
        self.cfg,self.method,self.memory,self.auditor=cfg,method,memory,auditor
        self.bank=np.zeros((cfg.height,cfg.width,3),np.float32)
        self.age=np.full((cfg.height,cfg.width),24.,np.float32)
        self.hist=np.zeros_like(self.age)
        self.anchor=None
        self.shadow_source=None

    def _advance_arrays(self,bank,age,hist,anchor,o):
        bank=remap(bank,o['flow'])
        anchor=remap(anchor,o['flow'])
        age=np.minimum(remap(age,o['flow'])+1,100)
        hist=remap(hist,o['flow'])+np.maximum(o['innovation']-2*o['noise']**2,0)
        return bank,age,hist,anchor

    def _predict_arrays(self,o,bank,age,hist,anchor,sr=None):
        c=self.cfg
        X,hh,H,A,E,G,AE,V=feature_maps(bank,age,anchor,hist,o,c)
        if self.memory is None:
            gain,risk=np.exp(-A/12),G+H
        else:
            gain,risk=self.memory.predict(X,H.ravel()); gain=gain.reshape(H.shape); risk=risk.reshape(H.shape)
        base=np.clip(o['b'],0,1)
        if self.method.memory_mode=='pem':
            pred=np.clip(o['b']+expand(gain,c.block)[...,None]*hh,0,1)
        elif self.method.memory_mode=='sr':
            if sr is None: raise ValueError('Neural prediction missing')
            pred=sr
        elif self.method.memory_mode=='sr_memory':
            if sr is None: raise ValueError('Neural prediction missing')
            pred=np.clip(sr+expand(np.exp(-A/12),c.block)[...,None]*bank,0,1)
        else:
            raise ValueError(self.method.memory_mode)
        return dict(X=X,h=hh,H=H,A=A,risk_map=risk,base=base,pred=pred)

    def prepare(self,obs,sr=None):
        c=self.cfg; o=obs
        if self.anchor is None:
            self.anchor=o['b'].copy()
            shadow=None
        else:
            self.bank,self.age,self.hist,self.anchor=self._advance_arrays(self.bank,self.age,self.hist,self.anchor,o)
            shadow=None
            if self.shadow_source is not None and self.method.twin_audit:
                sb,sa,sh,san=[x.copy() for x in self.shadow_source]
                sb,sa,sh,san=self._advance_arrays(sb,sa,sh,san,o)
                shadow=self._predict_arrays(o,sb,sa,sh,san,sr)
        main=self._predict_arrays(o,self.bank,self.age,self.hist,self.anchor,sr)
        X,hh,A,risk,base,pred=main['X'],main['h'],main['A'],main['risk_map'],main['base'],main['pred']
        z=tile_features(X,pred,base,risk,c)
        lo,hi,energy=contrast_bounds(base,pred,c.fovea)
        if self.auditor is None:
            experts=np.column_stack([energy*0,energy,energy*.5])
            r=energy[:,None]-experts
            cov=np.einsum('jk,jl->jkl',r,r)+1e-10*np.eye(3)[None]
        else:
            experts,cov=self.auditor.predict(z,energy)
        benefit=blocks(expand(risk*(1-np.exp(-A/8))+1e-5*(A/24)**2,c.block),c.fovea).ravel()
        if sr is not None:
            benefit=benefit+blocks((sr-base)**2,c.fovea).ravel()
        tile_risk=blocks(expand(np.maximum(risk,0),c.block),c.fovea).ravel()
        if shadow is None or self.method.memory_mode=='sr':
            shadow_pred=pred.copy(); write_valid=False
            write_lo=np.zeros_like(energy); write_hi=np.zeros_like(energy); write_energy=np.zeros_like(energy)
        else:
            shadow_pred=shadow['pred']; write_valid=True
            write_lo,write_hi,write_energy=contrast_bounds(shadow_pred,pred,c.fovea)
        snapshot=(self.bank.copy(),self.age.copy(),self.hist.copy(),self.anchor.copy())
        return dict(base=base,pred=pred,X=X,h=hh,energy=energy,features=z,experts=experts,covariance=cov,
                    benefit=benefit,risk=tile_risk,lo=lo,hi=hi,sr=sr,shadow_pred=shadow_pred,
                    write_valid=write_valid,write_lo=write_lo,write_hi=write_hi,write_energy=write_energy,
                    _prewrite_snapshot=snapshot)

    def update(self,state,patch,location):
        c=self.cfg; y,x=location; s=c.fovea; sl=np.s_[y:y+s,x:x+s]
        # This frozen copy becomes next frame's one-step no-write counterfactual.
        self.shadow_source=tuple(a.copy() for a in state['_prewrite_snapshot']) if self.method.twin_audit else None
        if self.method.memory_mode=='sr_memory':
            self.bank[sl]=patch-state['sr'][sl]
        else:
            self.bank[sl]=detail(patch,c.factor)
        self.anchor[sl]=base_image(down(patch,c.factor),c.factor)
        self.age[sl]=0; self.hist[sl]=0
        recon=state['pred'].copy(); recon[sl]=np.clip(patch,0,1)
        return recon

def ssim_rgb(x, y):
    """Mean RGB SSIM, 11x11 Gaussian window, valid interior, dynamic range 1."""
    k1, k2 = .01**2, .03**2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    a, b = blur(x), blur(y)
    vx = np.maximum(blur(x*x)-a*a, 0); vy = np.maximum(blur(y*y)-b*b, 0)
    cov = blur(x*y)-a*b
    z = ((2*a*b+k1)*(2*cov+k2))/((a*a+b*b+k1)*(vx+vy+k2))
    return float(z[5:-5, 5:-5].mean())


def run_sequence(frames, obs, cfg, acfg, name, regime, draw, method, memory=None, auditor=None,
                 sr_predictions=None, sr_ms=None, collect=False, keep=False, references=None):
    """Compare methods with common preview noise, patch noise and random uniforms.

    Evaluator references may be changed independently for boundary tests.
    No reference-derived value is returned to Controller or the online learner.
    """
    if len(frames) != len(obs): raise ValueError('Frame/preview length mismatch')
    c = cfg; n = c.height // c.fovea * (c.width // c.fovea)
    control = Controller(c, method, memory, auditor)
    effective = replace(acfg, envelope_reserve=acfg.envelope_reserve if method.envelope else 0.,
                        online_moment_mix=acfg.online_moment_mix if method.adapt_moment else 0.)
    mix = PredictableMixture(effective); cs = NormalMixtureCS(acfg.alpha, acfg.cs_rho)
    write_cs = NormalMixtureCS(acfg.alpha, acfg.cs_rho)
    sensor = PatchSensor(frames, c, name, regime, draw)
    rng = np.random.default_rng(seed(acfg.seed, name, regime, draw, 'paired-uniform'))
    records, training, memory_training, example = [], [], [], None
    true_sum = 0.; write_true_sum = 0.; write_count = 0; previous_recon = previous_ref = None
    oracle_A = np.zeros((3,3), float); oracle_b = np.zeros(3, float); oracle_c = 0.
    previous_seen = np.zeros((c.height, c.width), np.float32)
    max_sig = max(sensor_parameters(regime, t, len(frames), a)[1] for t in range(len(frames)) for a in range(n))
    if max_sig > acfg.sigma_bound + 1e-12:
        raise ValueError('Sensor standard deviation exceeds declared CS bound')
    for t, o in enumerate(obs):
        start = time.perf_counter()
        sr = sr_predictions[t] if sr_predictions is not None else None
        state = control.prepare(o, sr)
        experts = state['experts']
        if method.control == 'online':
            w = mix.weights.copy()
        elif method.control in ['learned', 'energy', 'zero']:
            w = np.eye(3)[{'zero': 0, 'energy': 1, 'learned': 2}[method.control]]
        else: raise ValueError(method.control)
        mu = experts @ w
        if method.policy == 'ppat_2026_m1':
            mu = experts[:, 2] - experts[:, 2].mean()
        moment = mix.moment(state['covariance']) if method.control == 'online' else np.einsum('k,jkl,l->j', w, state['covariance'], w)
        moment = np.maximum(moment, 1e-14)
        noise_bound = 4*acfg.sigma_bound**2*state['energy']/(3*c.fovea**2)
        # Counterfactual one-step write-value audit.  Energy is a fully
        # predictable analytic proxy; the envelope moment is calibration-free.
        write_mu = state['write_energy'].astype(float)
        write_noise_bound = 4*acfg.sigma_bound**2*state['write_energy']/(3*c.fovea**2)
        write_env = np.maximum(abs(state['write_lo']-write_mu), abs(state['write_hi']-write_mu))
        write_moment = np.maximum(write_env**2 + write_noise_bound, 1e-14) if state['write_valid'] else np.zeros(n)
        certificate_ref = np.nan; certificate_gamma = np.nan
        if method.certificate:
            lower, certificate_ref, certificate_gamma = certificate_lower(
                mu, state['lo'], state['hi'], noise_bound, effective)
        elif method.envelope:
            lower = envelope_lower(mu, state['lo'], state['hi'], noise_bound, effective)
        else:
            lower = np.full(n, effective.epsilon/n)
        dual_alpha=np.nan
        if method.policy=='dual_joint':
            q,dual_alpha,ratio,write_ratio=dual_audit_ray_gaze(
                state['benefit'],moment,write_moment,lower,effective.kappa,effective.write_kappa)
        else:
            q=policy_distribution(method.policy,state['benefit'],state['risk'],moment,experts,lower,effective,write_moment)
            qref=lower/lower.sum() if lower.sum()>=1-1e-12 else lower+(1-lower.sum())/n
            ratio=float(np.sum(moment/q)/np.sum(moment/qref))
            write_ratio=(float(np.sum(write_moment/q)/np.sum(write_moment/qref))
                         if state['write_valid'] and write_moment.sum()>0 else 0.)
        assert np.isclose(q.sum(),1) and np.min(q)>=acfg.epsilon/n-1e-12
        if method.policy in ['joint','dual_joint']: assert np.all(q>=lower-1e-12)
        if method.policy=='joint': assert ratio<=acfg.kappa+1e-7
        if method.policy=='dual_joint':
            assert ratio<=acfg.kappa+1e-7
            if state['write_valid']: assert write_ratio<=acfg.write_kappa+1e-7
        variance_proxy=subgaussian_proxy(q,mu,state['lo'],state['hi'],noise_bound)
        write_variance_proxy=(subgaussian_proxy(q,write_mu,state['write_lo'],state['write_hi'],write_noise_bound)
                              if state['write_valid'] else 0.)
        action = min(int(np.searchsorted(np.cumsum(q), rng.random(), side='right')), n-1)
        patch, loc = sensor.acquire(t, action)
        y, x = loc; s = c.fovea; sl = np.s_[y:y+s, x:x+s]
        observed = float(np.mean((patch-state['base'][sl])**2-(patch-state['pred'][sl])**2))
        ipw = observed/(n*q[action])
        estimate = float(mu.mean()+(observed-mu[action])/(n*q[action]))
        ci_lo, ci_hi, center = cs.update(estimate, variance_proxy)
        if state['write_valid']:
            observed_write=float(np.mean((patch-state['shadow_pred'][sl])**2-(patch-state['pred'][sl])**2))
            write_ipw=observed_write/(n*q[action])
            write_estimate=float(write_mu.mean()+(observed_write-write_mu[action])/(n*q[action]))
            write_ci_lo,write_ci_hi,write_center=write_cs.update(write_estimate,write_variance_proxy)
        else:
            observed_write=write_ipw=write_estimate=np.nan
            write_ci_lo=write_ci_hi=write_center=np.nan
        pp_mu = experts[:, 2] - experts[:, 2].mean()
        ppat = float((observed-pp_mu[action])/(n*q[action]))
        if collect:
            training.append((state['features'][action].copy(), observed, state['energy'][action], name))
            k = c.block
            X = state['X'].reshape(c.height//k, c.width//k, -1)[y//k:(y+s)//k, x//k:(x+s)//k].reshape(-1, len(FEATURES))
            h = state['h'][sl]; residual = patch-o['b'][sl]
            memory_training.append((X.copy(), blocks(h*residual,k).ravel(), blocks(h*h,k).ravel(), blocks(residual*residual,k).ravel()))
        recon = control.update(state, patch, loc)
        if method.control == 'online': mix.update(experts, q, action, observed, adapt_weights=method.adapt_weights)
        online_ms = 1000*(time.perf_counter()-start)
        # Everything below this point is evaluator-only.
        ref = (references if references is not None else frames)[t]
        d = blocks((ref-state['base'])**2-(ref-state['pred'])**2, s).ravel().astype(float)
        sigma = np.array([sensor_parameters(regime,t,len(frames),a)[1] for a in range(n)])
        noise_var = 4*sigma*sigma*state['energy']/(3*s*s)
        beta = np.array([sensor_parameters(regime,t,len(frames),a)[3] for a in range(n)])
        mean_observed = d + 2*beta*blocks(state['pred']-state['base'],s).ravel()
        qa, qb, qc = variance_quadratic(experts, mean_observed, q, noise_var)
        oracle_A += qa; oracle_b += qb; oracle_c += qc
        true_sum += float(d.mean()); target = true_sum/(t+1)
        if state['write_valid']:
            write_d=blocks((ref-state['shadow_pred'])**2-(ref-state['pred'])**2,s).ravel().astype(float)
            write_noise_var=4*sigma*sigma*state['write_energy']/(3*s*s)
            write_mean_observed=write_d+2*beta*blocks(state['pred']-state['shadow_pred'],s).ravel()
            write_true_sum+=float(write_d.mean()); write_count+=1
            write_target=write_true_sum/write_count
            exact_write=exact_variance(write_mean_observed,write_mu,q,write_noise_var)
            write_bias=float(write_mean_observed.mean()-write_d.mean())
            write_covered=bool(write_ci_lo<=write_target<=write_ci_hi)
        else:
            write_d=np.zeros(n); write_target=np.nan; exact_write=np.nan; write_bias=np.nan; write_covered=True
        mse = float(np.mean((recon-ref)**2)); bias = float(q@mean_observed-d.mean())
        seen = remap(previous_seen, o['flow']) > .5
        seen[sl] = False
        past_mse = float(np.mean((recon[seen]-ref[seen])**2)) if seen.any() else np.nan
        temporal = np.nan
        if previous_recon is not None:
            temporal = float(np.mean(((recon-previous_recon)-(ref-previous_ref))**2))
        row = dict(frame=t, action=action, propensity=float(q[action]), min_q=float(q.min()),
            effective_actions=float(1/(q@q)), predicted_moment_ratio=ratio,
            write_predicted_moment_ratio=float(write_ratio), dual_reference_mix=float(dual_alpha),
            certificate_reference=float(certificate_ref), certificate_gamma=float(certificate_gamma),
            certified_importance_actual=float(np.max((np.maximum(abs(state['lo']-mu), abs(state['hi']-mu)) + np.sqrt(noise_bound))/(n*q))),
            true_contrast=float(d.mean()), cumulative_true_contrast=target,
            write_valid=bool(state['write_valid']), write_true=float(write_d.mean()) if state['write_valid'] else np.nan,
            cumulative_write_true=float(write_target) if state['write_valid'] else np.nan,
            observed_write=observed_write, write_ipw=write_ipw, write_augmented=write_estimate,
            observed=observed, naive=observed, ipw=ipw, augmented=estimate, plugin=float(mu.mean()), ppat_m1=ppat,
            mse=mse, psnr=-10*np.log10(max(mse,1e-12)), ssim=ssim_rgb(recon, ref),
            preview_mse=float(np.mean((state['base']-ref)**2)),
            pre_mse=float(np.mean((state['pred']-ref)**2)), past_fovea_mse=past_mse,
            past_fovea_pixels=int(seen.sum()), temporal_residual_mse=temporal,
            exact_variance_augmented=exact_variance(mean_observed,mu,q,noise_var),
            exact_variance_ipw=exact_variance(mean_observed,np.zeros(n),q,noise_var),
            exact_variance_energy=exact_variance(mean_observed,experts[:,1],q,noise_var),
            exact_variance_learned=exact_variance(mean_observed,experts[:,2],q,noise_var),
            exact_variance_ppat=exact_variance(mean_observed,pp_mu,q,noise_var),
            exact_variance_write=exact_write, naive_conditional_bias=bias,
            augmented_conditional_bias=float(mean_observed.mean()-d.mean()), write_conditional_bias=write_bias,
            cs_low=ci_lo, cs_high=ci_hi, cs_center=center,
            write_cs_low=write_ci_lo, write_cs_high=write_ci_hi, write_cs_center=write_center,
            write_cs_covered=write_covered, write_cs_width=(max(write_ci_hi-write_ci_lo,0) if state['write_valid'] else np.nan),
            cs_covered=bool(ci_lo<=target<=ci_hi), cs_width=max(ci_hi-ci_lo,0), cs_empty=bool(ci_lo>ci_hi),
            cs_variance_proxy=variance_proxy, weight_zero=w[0], weight_energy=w[1], weight_learned=w[2],
            controller_ms=online_ms, preview_ms=o['preview_ms'], sr_ms=float(sr_ms[t]) if sr_ms is not None else 0.)
        records.append(row)
        previous_seen = remap(previous_seen, o['flow']); previous_seen[sl] = 1
        previous_recon, previous_ref = recon.copy(), ref.copy()
        if keep and t == len(frames)-1:
            example = dict(reference=ref, reconstruction=recon, preview=state['base'],
                gaze=q.reshape(c.height//s,c.width//s), weights=w.copy(), frame=t,
                audit_true=d.reshape(c.height//s,c.width//s), audit_proxy=mu.reshape(c.height//s,c.width//s),
                shadow=state['shadow_pred'], write_true=(write_d.reshape(c.height//s,c.width//s) if state['write_valid'] else None))
    assert sensor.reads == len(frames)
    import pandas as pd
    table = pd.DataFrame(records); total = len(table)
    summary = dict(frames=total, mse=float(table.mse.mean()),
        psnr_frame_mean=float(table.psnr.mean()), psnr_from_mean_mse=float(-10*np.log10(max(table.mse.mean(),1e-12))),
        ssim=float(table.ssim.mean()), preview_mse=float(table.preview_mse.mean()),
        pre_mse=float(table.pre_mse.mean()), temporal_residual_mse=float(table.temporal_residual_mse.mean()),
        past_fovea_mse=float(table.past_fovea_mse.mean()) if table.past_fovea_mse.notna().any() else None,
        acquisition_fraction=1/c.factor**2+c.fovea**2/(c.height*c.width),
        audit_true=float(table.true_contrast.mean()),
        cs_simultaneous_covered=bool(table.cs_covered.all()), cs_final_covered=bool(table.cs_covered.iloc[-1]),
        cs_final_width=float(table.cs_width.iloc[-1]), cs_valid_assumptions=regime!='biased_stress',
        cs_empty_frames=int(table.cs_empty.sum()),
        controller_ms_mean=float(table.controller_ms.mean()),
        online_total_ms_mean=float((table.controller_ms+table.preview_ms+table.sr_ms).mean()),
        online_total_ms_p95=float((table.controller_ms+table.preview_ms+table.sr_ms).quantile(.95)),
        max_predicted_moment_ratio=float(table.predicted_moment_ratio.max()),
        max_write_predicted_moment_ratio=float(table.write_predicted_moment_ratio.max()),
        dual_reference_mix_mean=float(table.dual_reference_mix.dropna().mean()) if table.dual_reference_mix.notna().any() else None,
        min_propensity=float(table.min_q.min()), effective_actions_mean=float(table.effective_actions.mean()))
    for est in ['naive','ipw','augmented','plugin','ppat_m1']:
        value = float(table[est].mean())
        summary[est+'_estimate'] = value
        summary[est+'_error'] = value-summary['audit_true']
        summary[est+'_squared_error'] = (value-summary['audit_true'])**2
    for est in ['augmented','ipw','energy','learned','ppat']:
        # Realized predictable quadratic variation / T^2. Its expectation is
        # the MSE of the martingale audit error; it is NOT the variance
        # conditional on the entire realized future trajectory.
        summary[est+'_pqv'] = float(table['exact_variance_'+est].sum()/total**2)
    wt=table[table.write_valid.astype(bool)]
    if len(wt):
        summary['write_true']=float(wt.write_true.mean())
        summary['write_estimate']=float(wt.write_augmented.mean())
        summary['write_error']=summary['write_estimate']-summary['write_true']
        summary['write_squared_error']=summary['write_error']**2
        summary['write_pqv']=float(wt.exact_variance_write.sum()/len(wt)**2)
        summary['write_cs_simultaneous_covered']=bool(wt.write_cs_covered.all())
        summary['write_cs_final_width']=float(wt.write_cs_width.iloc[-1])
        summary['write_frames']=int(len(wt))
    else:
        summary.update(write_true=0.,write_estimate=0.,write_error=0.,write_squared_error=0.,write_pqv=0.,
                       write_cs_simultaneous_covered=True,write_cs_final_width=0.,write_frames=0)
    summary['naive_mean_conditional_bias'] = float(table.naive_conditional_bias.mean())
    summary['augmented_mean_conditional_bias'] = float(table.augmented_conditional_bias.mean())
    oracle_w, oracle_total = best_fixed_mixture(oracle_A, oracle_b, oracle_c)
    oracle_pqv = oracle_total / total**2
    summary['hindsight_best_fixed_pqv'] = float(oracle_pqv)
    summary['online_mixture_regret_pqv'] = float(summary['augmented_pqv'] - oracle_pqv)
    summary['hindsight_weight_zero'] = float(oracle_w[0])
    summary['hindsight_weight_energy'] = float(oracle_w[1])
    summary['hindsight_weight_learned'] = float(oracle_w[2])
    finite_cert = table.certificate_gamma[np.isfinite(table.certificate_gamma)]
    summary['certificate_gamma_max'] = float(finite_cert.max()) if len(finite_cert) else None
    summary['certified_importance_actual_max'] = float(table.certified_importance_actual.max())
    return summary, table, training, memory_training, example
