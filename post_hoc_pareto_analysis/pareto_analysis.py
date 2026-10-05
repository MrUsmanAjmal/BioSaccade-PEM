"""Descriptive Pareto analysis (added after the locked paper run; see NOTE.md).

Points: four baselines from the add-on run, and the paper's TwinAudit kappa and certificate sweeps,
all on the same 45 sensitivity clips (nominal regime, draw 0). Means over source clusters with paired
percentile-bootstrap 95% intervals (6000 resamples of source clusters, shared across all points).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = Path(r'<RESULTS_DIR>\results\local_01_91e457e261dc')
OUT = Path(r'<RESULTS_DIR>\pareto_addon')
REPS, SEED = 6000, 20261005
BASE = {'PEM_uniform': ('Uniform gaze', 'o', '#7f7f7f'), 'PEM_greedy': ('Greedy gaze', 's', '#2ca02c'),
        'PPAT2026_M1': ('PPAT (2026), M=1', '^', '#9467bd'), 'PEM_audit_only': ('Audit-only gaze', 'D', '#8c564b')}
DS = {'ALL': 'All 45 clips', 'DAVIS': 'DAVIS (15 clips)', 'REDS': 'REDS val (15 clips)', 'REDS_train': 'REDS train (15 clips)'}

paper = pd.read_csv(PAPER / 'sequence_results.csv')
paper = paper[paper.study.isin(['kappa', 'certificate'])]
addon = pd.read_csv(OUT / 'addon_sequence_results.csv')

# ---- reproducibility check: add-on TwinAudit at the selected settings vs the paper's PEM_R9_k2 rows
a = addon[addon.method == 'PEM_R9_TwinAudit'].set_index(['dataset', 'clip_id']).sort_index()
p = paper[paper.method == 'PEM_R9_k2'].set_index(['dataset', 'clip_id']).sort_index()
cols = [c for c in a.columns if c in p.columns and '_ms' not in c and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(p[c])]
av, pv = a[cols].astype(float), p.loc[a.index, cols].astype(float)
diff = (av - pv).abs()
both_nan = av.isna() & pv.isna()
check = dict(clips=len(a), fields=len(cols), max_abs_difference=float(diff.where(~both_nan).max().max()),
             bit_identical_values=int(((diff == 0) | both_nan).sum().sum()), total_values=int(diff.size))
(OUT / 'reproducibility_check.json').write_text(json.dumps(check, indent=2))
print('reproducibility check:', check)

# ---- per-cluster values for every point
points = {}   # key -> (label, family, order, dataframe with source_group, dataset, psnr, pqv)
def take(df, method):
    g = df[df.method == method].groupby(['source_group', 'dataset'], as_index=False)[['psnr_frame_mean', 'augmented_pqv']].mean()
    return g.rename(columns={'psnr_frame_mean': 'psnr', 'augmented_pqv': 'pqv'})
for m, (label, _, _) in BASE.items():
    points[m] = (label, 'baseline', 0, take(addon, m))
for k in [1, 2, 4, 8]:
    points[f'k{k}'] = (f'κ={k}', 'kappa', k, take(paper, f'PEM_R9_k{k:g}'))
for ck in [1.05, 1.25, 1.5, 2, 4]:
    points[f'c{ck}'] = (f'cκ={ck:g}', 'certificate', ck, take(paper, f'PEM_R9_cert{ck:g}'))
points['c2.5'] = ('cκ=2.5 (selected)', 'certificate', 2.5, take(paper, 'PEM_R9_k2'))   # selected point: kappa=2, cert=2.5
groups = sorted(points['PEM_uniform'][3].source_group)
for key, (_, _, _, d) in points.items():
    assert sorted(d.source_group) == groups, key
ds_of = points['PEM_uniform'][3].set_index('source_group').dataset

def matrix(subset):
    g = [s for s in groups if subset == 'ALL' or ds_of[s] == subset]
    P = {k: d.set_index('source_group').loc[g] for k, (_, _, _, d) in points.items()}
    return g, {k: v.psnr.to_numpy() for k, v in P.items()}, {k: v.pqv.to_numpy() for k, v in P.items()}

rng = np.random.default_rng(SEED)
summary, crossing, boots = [], [], {}
for subset in DS:
    g, PS, PQ = matrix(subset)
    idx = rng.integers(0, len(g), (REPS, len(g)))
    bp = {k: v[idx].mean(1) for k, v in PS.items()}; bq = {k: v[idx].mean(1) for k, v in PQ.items()}
    D = {k: v - PS['PEM_uniform'] for k, v in PS.items()}          # paired PSNR gain over uniform gaze
    bd = {k: v[idx].mean(1) for k, v in D.items()}
    boots[subset] = (PS, PQ, bp, bq)
    for k, (label, fam, order, _) in points.items():
        summary.append(dict(subset=subset, point=k, label=label, family=fam, setting=order, clusters=len(g),
                            psnr=PS[k].mean(), psnr_lo=np.quantile(bp[k], .025), psnr_hi=np.quantile(bp[k], .975),
                            dpsnr=D[k].mean(), dpsnr_lo=np.quantile(bd[k], .025), dpsnr_hi=np.quantile(bd[k], .975),
                            pqv=PQ[k].mean(), pqv_lo=np.quantile(bq[k], .025), pqv_hi=np.quantile(bq[k], .975)))
    # crossing: TwinAudit curve PSNR interpolated (in log PQV) at each baseline's PQV, minus that baseline's PSNR
    for fam in ['kappa', 'certificate']:
        ks = [k for k, v in points.items() if v[1] == fam]
        for b in BASE:
            def delta(psn, pq):
                x = np.log(np.array([pq[k] for k in ks])); y = np.array([psn[k] for k in ks])
                o = np.argsort(x); x, y = x[o], y[o]; xb = np.log(pq[b])
                if xb < x[0] or xb > x[-1]:
                    return np.nan
                return float(np.interp(xb, x, y) - psn[b])
            point_est = delta({k: v.mean() for k, v in PS.items()}, {k: v.mean() for k, v in PQ.items()})
            d = np.array([delta({k: bp[k][r] for k in bp}, {k: bq[k][r] for k in bq}) for r in range(REPS)])
            ok = np.isfinite(d)
            # direct paired comparison with the curve point whose mean PQV is closest at or below the baseline's
            below = [k for k in ks if PQ[k].mean() <= PQ[b].mean()]
            near = max(below, key=lambda k: PQ[k].mean()) if below else None
            dd = (bp[near] - bp[b]) if near else None
            crossing.append(dict(subset=subset, curve=fam, baseline=BASE[b][0], baseline_pqv=PQ[b].mean(),
                curve_pqv_min=min(PQ[k].mean() for k in ks), curve_pqv_max=max(PQ[k].mean() for k in ks),
                psnr_gap_at_equal_pqv=point_est,
                ci_lo=np.quantile(d[ok], .025) if ok.mean() > .5 else np.nan,
                ci_hi=np.quantile(d[ok], .975) if ok.mean() > .5 else np.nan,
                resamples_in_curve_range=float(ok.mean()),
                nearest_curve_point_at_or_below_pqv=points[near][0] if near else None,
                its_psnr_minus_baseline=float(PS[near].mean() - PS[b].mean()) if near else np.nan,
                its_ci_lo=float(np.quantile(dd, .025)) if near else np.nan, its_ci_hi=float(np.quantile(dd, .975)) if near else np.nan))
S = pd.DataFrame(summary); C = pd.DataFrame(crossing)
S.to_csv(OUT / 'pareto_points.csv', index=False); C.to_csv(OUT / 'pareto_crossing.csv', index=False)

# ---- figure: y = paired PSNR gain over uniform gaze (same clips, same noise), x = audit PQV
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
OFF = {'PEM_uniform': (8, 8, 'left'), 'PEM_greedy': (-8, -16, 'right'), 'PPAT2026_M1': (0, -20, 'center'),
       'PEM_audit_only': (-8, -16, 'right')}
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
for ax, subset in zip(axs.ravel(), DS):
    s_ = S[S.subset == subset]
    ax.axhline(0, color='#999999', lw=.8, ls=':')
    for fam, color, name, dy in [('kappa', '#C44E52', 'TwinAudit, κ sweep', 7), ('certificate', '#4C72B0', 'TwinAudit, certificate-κ sweep', -13)]:
        c = s_[s_.family == fam].sort_values('setting')
        ax.errorbar(c.pqv, c.dpsnr, xerr=[c.pqv - c.pqv_lo, c.pqv_hi - c.pqv], yerr=[c.dpsnr - c.dpsnr_lo, c.dpsnr_hi - c.dpsnr],
                    color=color, marker='o', ms=5, lw=1.8, elinewidth=.7, capsize=2, alpha=.95, label=name)
        for r in c.itertuples():
            if r.point in ('c2.5', 'k2', 'k4'):      # k2 = the starred selected point; k4 is labelled together with k8
                continue
            text, xy, ha = r.label, (0, dy), 'center'
            if r.point == 'k8': text, xy, ha = 'κ=4, 8', (10, 0), 'left'
            if r.point == 'k1': text, xy, ha = 'κ=1', (8, -4), 'left'
            ax.annotate(text, (r.pqv, r.dpsnr), xytext=xy, textcoords='offset points', fontsize=7.5, color=color, ha=ha)
    sel = s_[s_.point == 'k2'].iloc[0]
    ax.plot(sel.pqv, sel.dpsnr, marker='*', ms=16, color='gold', mec='black', zorder=5, ls='none', label='TwinAudit, selected setting (κ=2, cκ=2.5)')
    for m, (label, marker, color) in BASE.items():
        b = s_[s_.point == m].iloc[0]
        ax.errorbar(b.pqv, b.dpsnr, xerr=[[b.pqv - b.pqv_lo], [b.pqv_hi - b.pqv]], yerr=[[b.dpsnr - b.dpsnr_lo], [b.dpsnr_hi - b.dpsnr]],
                    color=color, marker=marker, ms=9, mec='black', mew=.6, ls='none', elinewidth=.8, capsize=2, zorder=4, label=label)
        dx, dy, ha = OFF[m]
        ax.annotate(label, (b.pqv, b.dpsnr), xytext=(dx, dy), textcoords='offset points', fontsize=8.5, color=color,
                    fontweight='bold', ha=ha)
    ax.set_xscale('log'); ax.set_title(DS[subset], fontweight='bold')
    ax.set_xlabel('Audit PQV, log scale  (← more reliable audit)')
    ax.set_ylabel('PSNR gain over uniform gaze, dB  (↑ better)')
    ax.grid(True, which='major', alpha=.3)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(.5, -.005))
fig.suptitle('Reconstruction vs audit reliability on the 45 paired sensitivity clips (nominal sensor, draw 0)\n'
             'Descriptive analysis added after the locked paper run. Points: means over source clusters; bars: 95% paired source-cluster bootstrap intervals.\n'
             'Upper left is better: higher PSNR at a more reliable (lower-variance) audit.', fontsize=10.5)
fig.tight_layout(rect=(0, .05, 1, .94))
for ext in ['pdf', 'svg', 'png']:
    fig.savefig(OUT / f'pareto_frontier.{ext}', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)

pd.set_option('display.width', 250)
print(S[['subset', 'label', 'psnr', 'dpsnr', 'dpsnr_lo', 'dpsnr_hi', 'pqv', 'pqv_lo', 'pqv_hi']].to_string(index=False, float_format=lambda x: f'{x:.4g}'))
print(C.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
