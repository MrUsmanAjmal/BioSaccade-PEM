"""Audit estimators and gaze constraints."""
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class MethodConfig:
    epsilon: float = .05
    kappa: float = 2.
    write_kappa: float = 2.0  # simultaneous bound for one-step write-value audit
    learning_rate: float = .15
    envelope_reserve: float = .15  # retained only for the supplied R7 reserve baseline
    certificate_kappa: float = 1.5  # >=1; hard observable-envelope importance cap
    online_moment_mix: float = .25
    moment_rate: float = .05
    alpha: float = .05
    cs_rho: float = .01
    sigma_bound: float = .10
    seed: int = 20260922

    def __post_init__(self):
        assert 0 < self.epsilon < 1 and self.kappa >= 1 and self.write_kappa >= 1
        assert 0 <= self.envelope_reserve < 1 and self.certificate_kappa >= 1
        assert 0 <= self.online_moment_mix <= 1
        assert self.learning_rate >= 0 and 0 < self.alpha < 1 and self.cs_rho > 0
        assert self.sigma_bound >= 0 and 0 < self.moment_rate <= 1


def project_simplex(x):
    """Euclidean projection onto nonnegative weights summing to one."""
    x = np.asarray(x, dtype=float)
    u = np.sort(x)[::-1]
    c = np.cumsum(u) - 1
    good = np.flatnonzero(u - c / np.arange(1, len(x) + 1) > 0)
    theta = c[good[-1]] / (good[-1] + 1)
    w = np.maximum(x - theta, 0)
    return w / w.sum()


def lower_normalize(weights, lower):
    """Water filling with arbitrary, feasible probability lower bounds."""
    w = np.maximum(np.asarray(weights, float), 1e-100)
    lower = np.broadcast_to(np.asarray(lower, float), w.shape).copy()
    if np.any(lower <= 0) or lower.sum() > 1 + 1e-12:
        raise ValueError('Infeasible probability lower bounds')
    if lower.sum() >= 1 - 1e-14:
        return lower / lower.sum()
    fixed = np.zeros(len(w), bool)
    for _ in range(len(w) + 1):
        p = lower.copy()
        free = ~fixed
        p[free] = (1 - lower[fixed].sum()) * w[free] / w[free].sum()
        bad = free & (p < lower - 1e-15)
        if not bad.any():
            return p / p.sum()
        fixed |= bad
    raise RuntimeError('Water filling failed')


def constrained_gaze(benefit, moment, lower, kappa=2.):
    """Maximize r.q subject to sum(v/q) <= kappa * sum(v/q_ref).

    q_ref = lower + (1-sum(lower))/J is always feasible. At constant lower
    bounds this recovers the R6 uniform-reference constraint. KKT bisection
    handles unequal envelope floors; near-degenerate cases return q_ref.
    """
    if kappa < 1:
        raise ValueError('kappa must be at least one')
    r = np.asarray(benefit, float)
    v = np.maximum(np.asarray(moment, float), 1e-16)
    l = np.broadcast_to(np.asarray(lower, float), r.shape).copy()
    if np.any(l <= 0) or l.sum() > 1 + 1e-12:
        raise ValueError('lower must be positive and sum to at most one')
    if l.sum() >= 1 - 1e-12:
        return l / l.sum()
    ref = l + (1 - l.sum()) / len(l)
    v = v / max(v.mean(), 1e-30)
    cap = float(kappa * np.sum(v / ref))
    moment_fn = lambda p: float(np.sum(v / p))
    greedy = l.copy()
    greedy[int(np.argmax(r))] += 1 - l.sum()
    if moment_fn(greedy) <= cap * (1 + 1e-12):
        return greedy
    if np.ptp(r) < 1e-14:
        return lower_normalize(np.sqrt(v), l)
    r = (r - r.min()) / np.ptp(r)
    def dist(nu):
        return lower_normalize(np.sqrt(v / np.maximum(nu - r, 1e-15)), l)
    lo, hi = 1., 2.
    for _ in range(64):
        if moment_fn(dist(hi)) <= cap:
            break
        hi = 1 + 2 * (hi - 1)
    else:
        return ref
    for _ in range(50):
        mid = (lo + hi) / 2
        if moment_fn(dist(mid)) > cap:
            lo = mid
        else:
            hi = mid
    p = dist(hi)
    # Numerical fallback is explicit and preserves feasibility.
    return p if moment_fn(p) <= cap * (1 + 1e-9) else ref


def contrast_bounds(base, prediction, tile):
    """Tight pixelwise extrema of d=2*x*(m-b)+b^2-m^2 for x in [0,1]."""
    from video_core import blocks
    slope = 2 * (prediction - base)
    intercept = base * base - prediction * prediction
    lo = blocks(intercept + np.minimum(slope, 0), tile).ravel().astype(float)
    hi = blocks(intercept + np.maximum(slope, 0), tile).ravel().astype(float)
    energy = blocks((prediction - base) ** 2, tile).ravel().astype(float)
    return lo, hi, energy


def envelope_lower(mu, lo, hi, noise_var_bound, cfg):
    """Reserve a fixed mass using observable residual and noise envelopes."""
    c = np.maximum(abs(lo - mu), abs(hi - mu)) + np.sqrt(noise_var_bound)
    mass = c / c.sum() if c.sum() > 1e-15 else np.full(len(c), 1 / len(c))
    return cfg.epsilon / len(c) + (1 - cfg.epsilon) * cfg.envelope_reserve * mass


def certificate_lower(mu, lo, hi, noise_var_bound, cfg):
    """Hard predictable probability certificate.

    Let c_j bound |d_j-mu_j| plus the declared Gaussian noise scale.  A
    distribution q_safe proportional to c (with the exploration floor) defines
    B_ref=max_j c_j/(J q_safe,j).  Requiring q_j >= c_j/(J*Gamma), with
    Gamma=certificate_kappa*B_ref, guarantees max_j c_j/(J q_j) <= Gamma.
    The guarantee is observable before the action and does not use learned
    residual calibration.  certificate_kappa=1 is the tightest reference cap.
    """
    c = np.maximum(abs(lo - mu), abs(hi - mu)) + np.sqrt(np.maximum(noise_var_bound, 0))
    j = len(c)
    floor = np.full(j, cfg.epsilon / j)
    q_safe = lower_normalize(np.maximum(c, 1e-15), floor)
    b_ref = float(np.max(c / np.maximum(j * q_safe, 1e-30)))
    gamma = max(cfg.certificate_kappa * b_ref, 1e-15)
    lower = np.maximum(floor, c / (j * gamma))
    # q_safe is feasible by construction; absorb only floating-point excess.
    if lower.sum() > 1 + 1e-10:
        raise RuntimeError('Certificate lower bounds became infeasible')
    if lower.sum() > 1:
        lower /= lower.sum()
    return lower, b_ref, gamma


def subgaussian_proxy(q, mu, lo, hi, noise_var_bound):
    """Hoeffding range proxy plus maximum conditional Gaussian variance.

    Gaussian noise must be independent across RGB scalar samples with SD
    no greater than the declared bound. The audit observation is not clipped.
    """
    j = len(q)
    left = mu.mean() + (lo - mu) / (j * q)
    right = mu.mean() + (hi - mu) / (j * q)
    width = float(right.max() - left.min())
    return width * width / 4 + float(np.max(noise_var_bound / (j * q) ** 2))


class NormalMixtureCS:
    def __init__(self, alpha=.05, rho=.01):
        self.alpha, self.rho = alpha, rho
        self.total = self.variance_proxy = 0.
        self.n = 0

    def update(self, estimate, predictable_proxy):
        self.total += float(estimate)
        self.variance_proxy += max(float(predictable_proxy), 0)
        self.n += 1
        v = self.variance_proxy + self.rho
        radius = np.sqrt(v * np.log(v / (self.rho * self.alpha ** 2))) / self.n
        center = self.total / self.n
        # Intersect with the known parameter domain; an empty interval signals
        # a coverage failure and is not silently repaired into a nonempty set.
        return max(-1., center - radius), min(1., center + radius), center


def audit_gradient(experts, weights, q, action, observed):
    """Unbiased stochastic gradient of the conditional design variance at fixed q.

    E[g | pre-action state] = -2/J^2 sum M_j e_j/q_j
                              +2*Mbar*(dbar-mubar).
    Taking a gradient through q would be a different objective; we do not do so.
    """
    j = len(q)
    mu = experts @ weights
    a = action
    return (-2 * experts[a] * (observed - mu[a]) / (j * q[a]) ** 2
            + 2 * experts.mean(0) * (observed / (j * q[a]) - mu.mean()))


class PredictableMixture:
    def __init__(self, cfg):
        self.cfg = cfg
        self.weights = np.array([0., 0., 1.])  # zero, energy, learned
        self.gradient_squared = 1e-12
        self.recent = None

    def moment(self, covariances):
        c = covariances
        if self.recent is not None:
            rate = self.cfg.online_moment_mix
            c = (1 - rate) * c + rate * self.recent[None]
        return np.maximum(np.einsum('k,jkl,l->j', self.weights, c, self.weights), 1e-14)

    def update(self, experts, q, action, observed, adapt_weights=True):
        # Called strictly AFTER storing the current audit and current q.
        if adapt_weights:
            g = audit_gradient(experts, self.weights, q, action, observed)
            self.gradient_squared += float(g @ g)
            step = self.cfg.learning_rate / np.sqrt(self.gradient_squared)
            self.weights = project_simplex(self.weights - step * g)
        residual = observed - experts[action]
        sample = np.outer(residual, residual) / (len(q) * q[action])
        rate = self.cfg.moment_rate
        self.recent = sample if self.recent is None else (1 - rate) * self.recent + rate * sample


class AuditEnsemble:
    """Source-cross-fitted three-expert residual second-moment matrices."""
    def __init__(self, trees=64, leaf=12):
        self.trees, self.leaf = trees, leaf

    def _regressor(self, seed):
        return ExtraTreesRegressor(n_estimators=self.trees, min_samples_leaf=self.leaf,
                                   max_depth=16, n_jobs=1, random_state=seed)

    def fit(self, features, observed, energy, groups):
        if len(np.unique(groups)) < 2:
            raise ValueError('Auditor needs at least two independent training groups')
        oof = np.empty(len(observed), float)
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        for train, valid in cv.split(features, observed, groups):
            oof[valid] = self._regressor(104).fit(features[train], observed[train]).predict(features[valid])
        residual = observed[:, None] - np.column_stack([np.zeros(len(oof)), energy, np.clip(oof, -1, 1)])
        covariance = np.einsum('nk,nl->nkl', residual, residual)
        self.mean = self._regressor(104).fit(features, observed)
        self.second = self._regressor(105).fit(features, covariance.reshape(-1, 9))
        self.floor = max(float(np.quantile(residual[:, 2] ** 2, .10)), 1e-14)
        self.oof_mse = float(np.mean(residual[:, 2] ** 2))
        return self

    def predict(self, features, energy):
        mean = np.clip(self.mean.predict(features), -1, 1)
        experts = np.column_stack([np.zeros(len(mean)), np.clip(energy, 0, 1), mean])
        c = self.second.predict(features).reshape(-1, 3, 3)
        # A common multi-output tree partition averages PSD outer products.
        # Symmetrization and a diagonal ridge absorb floating point noise.
        c = (c + c.transpose(0, 2, 1)) / 2 + self.floor * np.eye(3)[None]
        return experts, c



def dual_audit_ray_gaze(benefit, primary_moment, write_moment, lower, kappa=2., write_kappa=2.):
    """Fast simultaneous two-audit allocation with explicit guarantees.

    The benefit-greedy distribution q0 and the feasible reference q_ref share
    identical positive lower bounds.  We search only the line segment
        q(alpha)=(1-alpha)q0 + alpha q_ref.
    Because each second-moment functional sum_j v_j/q_j is convex in q and
    q_ref is feasible, the feasible alpha values form an interval containing 1.
    Bisection therefore returns the smallest feasible alpha.  Since expected
    reconstruction benefit is linear in q and q0 maximizes it on this ray, this
    is the highest-benefit point on the ray satisfying *both* audit budgets.

    This is intentionally not claimed to solve the globally optimal
    multi-constraint simplex problem; it is a cheap, certifiable controller for
    the very large video benchmark.
    """
    if kappa < 1 or write_kappa < 1:
        raise ValueError('audit kappas must be at least one')
    r=np.asarray(benefit,float); v1=np.maximum(np.asarray(primary_moment,float),1e-16)
    v2=np.maximum(np.asarray(write_moment,float),0.)
    l=np.broadcast_to(np.asarray(lower,float),r.shape).copy()
    if np.any(l<=0) or l.sum()>1+1e-12: raise ValueError('infeasible lower bounds')
    if l.sum()>=1-1e-12:
        q=l/l.sum(); return q,1.0,1.0,1.0
    ref=l+(1-l.sum())/len(l)
    greedy=l.copy(); greedy[int(np.argmax(r))]+=1-l.sum()
    c1_ref=float(np.sum(v1/ref)); cap1=kappa*c1_ref
    active2=float(v2.sum())>1e-18
    c2_ref=float(np.sum(v2/ref)) if active2 else 1.0
    cap2=write_kappa*c2_ref
    def stats(q):
        a=float(np.sum(v1/q))/max(c1_ref,1e-30)
        b=float(np.sum(v2/q))/max(c2_ref,1e-30) if active2 else 0.0
        return a,b
    def feasible(q):
        a,b=stats(q); return a<=kappa*(1+1e-11) and (not active2 or b<=write_kappa*(1+1e-11))
    if feasible(greedy):
        a,b=stats(greedy); return greedy,0.0,a,b
    lo,hi=0.,1.
    for _ in range(60):
        mid=(lo+hi)/2; q=(1-mid)*greedy+mid*ref
        if feasible(q): hi=mid
        else: lo=mid
    q=(1-hi)*greedy+hi*ref
    a,b=stats(q)
    if not feasible(q):
        q=ref; hi=1.; a,b=stats(q)
    return q,float(hi),float(a),float(b)


def policy_distribution(policy, benefit, risk, moment, experts, lower, cfg, write_moment=None):
    n = len(benefit)
    const = np.full(n, cfg.epsilon / n)
    if policy == 'uniform':
        return np.full(n, 1 / n)
    if policy == 'greedy':
        p = const.copy(); p[int(np.argmax(benefit))] += 1 - cfg.epsilon
        return p
    if policy == 'risk_only':
        # True uncertainty/risk baseline; the supplied R7 notebook accidentally
        # reused reconstruction benefit here, making this comparator mislabeled.
        return lower_normalize(np.sqrt(np.maximum(risk, 1e-14)), const)
    if policy == 'softmax':
        z = (benefit - np.min(benefit)) / max(float(np.ptp(benefit)), 1e-12)
        z = np.exp((z - 1) / .25)
        return const + (1 - cfg.epsilon) * z / z.sum()
    if policy == 'audit_only':
        return lower_normalize(np.sqrt(moment), const)
    if policy == 'ppat_2026_m1':
        # Native M=1 centered-proxy PPAT proposal with learned residual
        # variance and surrogate conditional mean equal to the learned proxy.
        score = np.sqrt(moment + float(experts[:, 2].mean()) ** 2)
        return const + (1 - cfg.epsilon) * score / score.sum()
    if policy == 'joint':
        return constrained_gaze(benefit, moment, lower, cfg.kappa)
    if policy == 'dual_joint':
        if write_moment is None:
            raise ValueError('dual_joint requires write_moment')
        return dual_audit_ray_gaze(benefit, moment, write_moment, lower, cfg.kappa, cfg.write_kappa)[0]
    raise ValueError(policy)


def exact_variance(d, mu, q, noise_var):
    e = d - mu
    return max(float(np.sum((e * e + noise_var) / q) / len(q) ** 2 - e.mean() ** 2), 0.)


def variance_quadratic(experts, d, q, noise_var):
    """Return A,b,c for V(w)=w'A w+2 b'w+c at a fixed realized state.

    This is evaluator-only and is used after the trajectory to measure online
    mixture regret against the best *fixed* convex expert mixture in hindsight.
    It never changes the executed actions or estimates.
    """
    M = np.asarray(experts, float)
    d = np.asarray(d, float)
    q = np.asarray(q, float)
    nv = np.asarray(noise_var, float)
    j = len(q)
    mbar = M.mean(0)
    dbar = float(d.mean())
    A = M.T @ (M / q[:, None]) / (j*j) - np.outer(mbar, mbar)
    b = -(M.T @ (d / q)) / (j*j) + dbar * mbar
    c = float(np.sum((d*d + nv) / q) / (j*j) - dbar*dbar)
    A = (A + A.T) / 2
    return A, b, c


def best_fixed_mixture(A, b, c):
    """Convex 3-expert hindsight oracle; diagnostic, not a deployable method."""
    from scipy.optimize import minimize
    A = (np.asarray(A, float) + np.asarray(A, float).T) / 2
    b = np.asarray(b, float)
    fun = lambda w: float(w @ A @ w + 2*b @ w + c)
    jac = lambda w: 2*A @ w + 2*b
    res = minimize(fun, np.full(3, 1/3), jac=jac, method='SLSQP',
                   bounds=[(0.,1.)]*3,
                   constraints={'type':'eq','fun':lambda w: float(w.sum()-1),
                                'jac':lambda w: np.ones(3)},
                   options={'ftol':1e-12,'maxiter':500})
    candidates = [np.eye(3)[k] for k in range(3)]
    if res.success and np.isfinite(res.fun):
        candidates.append(project_simplex(res.x))
    vals = [fun(w) for w in candidates]
    k = int(np.argmin(vals))
    return candidates[k], max(float(vals[k]), 0.)


def fixed_pool_ppat(loss, proxy, scores, sample_count, rng, epsilon=.05):
    """Native finite-pool LURE/PPAT, fixed lambda=1, without replacement.

    Diagnostic only: freeze a state and do not update memory during this assay.
    This extra dense-reference experiment is not a one-patch sensor result.
    """
    n, m = len(loss), sample_count
    if not 1 <= m <= n:
        raise ValueError('Invalid query budget')
    remaining = list(range(n)); vals, residuals = [], []
    centered = proxy - proxy.mean()
    for k in range(1, m + 1):
        score = np.maximum(scores[remaining], 1e-15)
        q = epsilon / len(remaining) + (1 - epsilon) * score / score.sum()
        pos = int(rng.choice(len(remaining), p=q)); a = remaining.pop(pos)
        weight = 1. if m == n else 1 + (n - m) / (n - k) * (1 / ((n - k + 1) * q[pos]) - 1)
        vals.append(weight * loss[a]); residuals.append(weight * (loss[a] - centered[a]))
    return float(np.mean(vals)), float(np.mean(residuals))
