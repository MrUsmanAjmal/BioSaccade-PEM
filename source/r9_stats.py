"""Statistics and figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONTEXT = ['dataset','regime','height','width','factor','fovea','preview_kernel','study']
METRICS = ['mse','ssim','augmented_pqv','augmented_squared_error','write_pqv','write_squared_error','cs_final_width','write_cs_final_width','online_total_ms_mean']


DISPLAY = {
    'PEM_R9_TwinAudit': 'TwinAudit (ours)',
    'PEM_R9_TwinAudit_strict': 'TwinAudit-strict (κw = 1)',
    'PEM_R9_dual_fixed': 'Ours w/o online proxy mixture',
    'PEM_R9_single_audit': 'Ours w/o write-audit constraint (duplicate definition)',
    'PEM_R8_certified': 'Ours w/o write-audit constraint',
    'PEM_R7_reserve': 'Envelope-reserve joint (superseded)',
    'PEM_R6_fixed': 'Joint gaze, fixed proxy (prior version)',
    'PEM_uniform': 'Uniform gaze', 'PEM_greedy': 'Greedy gaze', 'PEM_softmax': 'Benefit-softmax gaze',
    'PEM_risk_only': 'Risk-only gaze', 'PEM_audit_only': 'Audit-only gaze',
    'PPAT2026_M1': 'PPAT (2026), M=1',
}
_SR = {'NanoSR2025': 'NanoSR (NTIRE 2025)', 'SPANF2025': 'SPAN-F (NTIRE 2025)', 'SCMSR2025': 'SCMSR (NTIRE 2025)',
       'BasicVSRpp2022_causal': 'BasicVSR++ causal (CVPR 2022)', 'BasicVSRpp2022_offline': 'BasicVSR++ offline, non-causal (CVPR 2022)'}
_SUFFIX = {'_patch': ' + patch', '_memory': ' + patch + memory', '_memory_R9': ' + TwinAudit gaze (ours)'}


def display_name(name):
    """Reader-facing method names; internal revision labels never reach figures or tables."""
    if name in DISPLAY: return DISPLAY[name]
    for suffix in ['_memory_R9', '_memory', '_patch']:
        if name.endswith(suffix) and name[:-len(suffix)] in _SR:
            return _SR[name[:-len(suffix)]] + _SUFFIX[suffix]
    for prefix, label in [('PEM_R9_writek', 'Ours, write kappa='), ('PEM_R9_cert', 'Ours, certificate kappa='),
                          ('PEM_R9_k', 'Ours, kappa=')]:
        if name.startswith(prefix): return label + name[len(prefix):]
    return name


def bootstrap_diff(a,b,rng,reps=6000):
    a,b=np.asarray(a,float),np.asarray(b,float); d=a-b
    idx=rng.integers(0,len(d),(reps,len(d)))
    return float(d.mean()),*np.quantile(d[idx].mean(1),[.025,.975])


def bootstrap_ratio(a,b,rng,reps=6000):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Paired inference needs finite data from at least two source clusters')
    if b.mean()<=1e-25:
        return None
    values=[]
    for start in range(0,reps,256):
        idx=rng.integers(0,len(a),(min(256,reps-start),len(a)))
        den=b[idx].mean(1)
        if np.any(den<=1e-25):
            return None
        values.extend(100*(1-a[idx].mean(1)/den))
    return 100*(1-a.mean()/b.mean()),*np.quantile(values,[.025,.975])


def paired_signflip(a,b,rng,reps=20000):
    """Requires exchangeability/symmetry of paired source effects under H0."""
    delta=np.asarray(b)-np.asarray(a);obs=abs(delta.mean());n=len(delta)
    if n<=14:
        ids=np.arange(2**n,dtype=np.uint32)
        signs=2*((ids[:,None]>>np.arange(n))&1).astype(float)-1
        return float(np.mean(abs(signs@delta/n)>=obs-1e-15))
    count=0
    for start in range(0,reps,256):
        signs=rng.choice([-1,1],(min(256,reps-start),n))
        count+=int(np.sum(abs(signs@delta/n)>=obs-1e-15))
    return (count+1)/(reps+1)


def holm(p):
    p=np.asarray(p,float);order=np.argsort(p)
    adjusted=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    out=np.empty(len(p));out[order]=adjusted;return out


def paired_statistics(df,reps=6000,target='PEM_R9_TwinAudit'):
    rng=np.random.default_rng(27182);rows=[]
    match=['clip_id','source_group','draw']
    for context,g in df.groupby(CONTEXT,dropna=False,sort=True):
        if target not in set(g.method):continue
        for baseline in sorted(set(g.method)-{target}):
            aa=g[g.method==target].set_index(match)
            bb=g[g.method==baseline].set_index(match)
            if set(aa.index)!=set(bb.index):
                raise AssertionError(f'Incomplete paired comparisons: {context}/{baseline}')
            for metric in METRICS:
                a=aa[metric].groupby('source_group').mean().sort_index()
                b=bb[metric].groupby('source_group').mean().sort_index()
                if len(a)<2:continue
                stat=bootstrap_ratio(a.values,b.values,rng,reps)
                if stat is None:continue
                row=dict(zip(CONTEXT,context));row.update(target=target,baseline=baseline,metric=metric,
                    source_clusters=len(a),clips=g.clip_id.nunique(),
                    reduction_pct=stat[0],ci_low=stat[1],ci_high=stat[2],
                    target_mean=float(a.mean()),baseline_mean=float(b.mean()),
                    paired_p=paired_signflip(a.values,b.values,rng),
                    positive_reduction_favors_target=metric!='ssim',effect_type='percent_reduction')
                rows.append(row)
            # PSNR (dB) from source-mean MSE: standard reporting unit in SR literature.
            a=-10*np.log10(np.maximum(aa['mse'].groupby('source_group').mean().sort_index(),1e-12))
            b=-10*np.log10(np.maximum(bb['mse'].groupby('source_group').mean().sort_index(),1e-12))
            if len(a)>=2:
                point,lo,hi=bootstrap_diff(a.values,b.values,rng,reps)
                row=dict(zip(CONTEXT,context));row.update(target=target,baseline=baseline,metric='psnr_db',
                    source_clusters=len(a),clips=g.clip_id.nunique(),reduction_pct=point,ci_low=lo,ci_high=hi,
                    target_mean=float(a.mean()),baseline_mean=float(b.mean()),
                    paired_p=paired_signflip(a.values,b.values,rng),
                    positive_reduction_favors_target=True,effect_type='difference_dB')
                rows.append(row)
    out=pd.DataFrame(rows)
    if len(out):
        out['holm_p']=np.nan
        for _,idx in out.groupby(['study','metric']).groups.items():
            out.loc[idx,'holm_p']=holm(out.loc[idx,'paired_p'].values)
    return out


BACKBONE_METRICS = ['mse','augmented_pqv','write_pqv']


def backbone_statistics(df,reps=6000):
    """Same-backbone gaze comparisons in the neural studies.

    Added after the laptop pilot and before the paper run. For each published
    backbone B: B_memory_R9 (TwinAudit gaze) vs B_memory (uniform gaze, identical
    memory) and vs B_patch (uniform gaze, no memory). Same paired source-cluster
    bootstrap and sign-flip test as paired_statistics; Holm within study and metric.
    Ratios with a zero baseline (write PQV of _patch, which has no write audit)
    are undefined and omitted, as in paired_statistics.
    """
    rng=np.random.default_rng(31415);rows=[]
    match=['clip_id','source_group','draw']
    neural=df[df.study.astype(str).str.startswith('neural')]
    for context,g in neural.groupby(CONTEXT,dropna=False,sort=True):
        methods=set(g.method)
        for target in sorted(m for m in methods if m.endswith('_memory_R9')):
            backbone=target[:-len('_memory_R9')]
            for baseline in [backbone+'_memory',backbone+'_patch']:
                if baseline not in methods:continue
                aa=g[g.method==target].set_index(match)
                bb=g[g.method==baseline].set_index(match)
                if set(aa.index)!=set(bb.index):
                    raise AssertionError(f'Incomplete paired comparisons: {context}/{target}/{baseline}')
                common=dict(zip(CONTEXT,context));common.update(backbone=backbone,target=target,baseline=baseline)
                for metric in BACKBONE_METRICS:
                    a=aa[metric].groupby('source_group').mean().sort_index()
                    b=bb[metric].groupby('source_group').mean().sort_index()
                    if len(a)<2:continue
                    stat=bootstrap_ratio(a.values,b.values,rng,reps)
                    if stat is None:continue
                    rows.append(dict(common,metric=metric,source_clusters=len(a),clips=g.clip_id.nunique(),
                        reduction_pct=stat[0],ci_low=stat[1],ci_high=stat[2],
                        target_mean=float(a.mean()),baseline_mean=float(b.mean()),
                        paired_p=paired_signflip(a.values,b.values,rng),
                        positive_reduction_favors_target=True,effect_type='percent_reduction'))
                a=-10*np.log10(np.maximum(aa['mse'].groupby('source_group').mean().sort_index(),1e-12))
                b=-10*np.log10(np.maximum(bb['mse'].groupby('source_group').mean().sort_index(),1e-12))
                if len(a)>=2:
                    point,lo,hi=bootstrap_diff(a.values,b.values,rng,reps)
                    rows.append(dict(common,metric='psnr_db',source_clusters=len(a),clips=g.clip_id.nunique(),
                        reduction_pct=point,ci_low=lo,ci_high=hi,
                        target_mean=float(a.mean()),baseline_mean=float(b.mean()),
                        paired_p=paired_signflip(a.values,b.values,rng),
                        positive_reduction_favors_target=True,effect_type='difference_dB'))
    out=pd.DataFrame(rows)
    if len(out):
        out['holm_p']=np.nan
        for _,idx in out.groupby(['study','metric']).groups.items():
            out.loc[idx,'holm_p']=holm(out.loc[idx,'paired_p'].values)
        out['target_display']=out.target.map(display_name);out['baseline_display']=out.baseline.map(display_name)
    return out


def export_backbone_comparisons(df,out,synthetic=False,reps=6000):
    """CSV + booktabs LaTeX for backbone_statistics (post-pilot amendment)."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    table=backbone_statistics(df,reps)
    t=table.copy();t['evidence_scope']='SYNTHETIC SOFTWARE CHECK ONLY' if synthetic else 'NATURAL VIDEO WITH SIMULATED SENSING'
    t['amendment']='added after the laptop pilot and before the paper run'
    t.to_csv(out/'backbone_gaze_comparisons.csv',index=False)
    if len(table):
        tab=out/'tables';tab.mkdir(exist_ok=True)
        keep=table.copy()
        keep['Effect']=np.where(keep.effect_type=='difference_dB','dB gain','% reduction')
        keep['metric']=keep.metric.map({'psnr_db':'PSNR','mse':'MSE','augmented_pqv':'Audit PQV','write_pqv':'Write PQV'})
        cols={'dataset':'Dataset','regime':'Regime','target_display':'Ours','baseline_display':'Baseline','metric':'Metric',
              'Effect':'Effect','reduction_pct':'Estimate','ci_low':'95% CI low','ci_high':'95% CI high','holm_p':'Holm p'}
        for study,h in keep.groupby('study'):
            write_latex(h[list(cols)].rename(columns=cols),tab/f'table_backbone_gaze_{study}.tex',
                f'Same-backbone gaze comparisons ({study}): each published backbone with TwinAudit gaze vs the same backbone with uniform gaze (+ memory, + patch). Positive values favour TwinAudit gaze; paired source-cluster 95% percentile bootstrap intervals; Holm-adjusted sign-flip p values. Comparison added after the laptop pilot, before the paper run.',
                f'tab:backbone_gaze_{study}')
    return table


def control_statistics(df,reps=6000):
    """Matched estimator ablations on the very same executed R9 trajectories."""
    rng=np.random.default_rng(9821);rows=[]
    for context,g in df[df.method=='PEM_R9_TwinAudit'].groupby(CONTEXT,sort=True):
        p=g.groupby('source_group').mean(numeric_only=True)
        if len(p)<2:continue
        for est in ['ipw','energy','learned','ppat']:
            stat=bootstrap_ratio(p.augmented_pqv,p[est+'_pqv'],rng,reps)
            if stat is None:continue
            row=dict(zip(CONTEXT,context));row.update(comparator=est,source_clusters=len(p),
                reduction_pct=stat[0],ci_low=stat[1],ci_high=stat[2],
                paired_p=paired_signflip(p.augmented_pqv.values,p[est+'_pqv'].values,rng))
            rows.append(row)
    out=pd.DataFrame(rows)
    if len(out):out['holm_p']=holm(out.paired_p.values)
    return out


def source_summary(df):
    keys=CONTEXT+['method']
    numeric=df.select_dtypes(include=['number','bool']).columns.difference(CONTEXT+['draw'])
    source=df.groupby(keys+['source_group'])[numeric].mean().reset_index()
    group=source.groupby(keys)
    result=group[list(numeric)].mean().reset_index()
    result=result.merge(group.size().rename('source_clusters').reset_index(),on=keys)
    result=result.merge(df.groupby(keys).clip_id.nunique().rename('clips').reset_index(),on=keys)
    return result


def export_analysis(df,out,synthetic=False,reps=6000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary=source_summary(df);paired=paired_statistics(df,reps);controls=control_statistics(df,reps)
    scope='SYNTHETIC SOFTWARE CHECK ONLY' if synthetic else 'NATURAL VIDEO WITH SIMULATED SENSING'
    for table in [summary]:
        if 'method' in table: table['display_name']=table.method.map(display_name)
    if len(paired):
        paired['target_display']=paired.target.map(display_name); paired['baseline_display']=paired.baseline.map(display_name)
    for name,table in [('sequence_results',df),('source_summary',summary),('paired_comparisons',paired),('same_gaze_ablation',controls)]:
        table=table.copy();table['evidence_scope']=scope;table.to_csv(out/(name+'.csv'),index=False)
    return summary,paired,controls


def publication_figures(df,summary,paired,controls,out,synthetic=False,line_dpi=1000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);paths=[];captions=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name,caption,dpi=line_dpi):
        if synthetic:
            fig.suptitle('Synthetic software check - not benchmark evidence',fontsize=11,y=1.015)
        fig.tight_layout()
        for ext in ['pdf','svg','png']:
            fig.savefig(out/(name+'.'+ext),dpi=dpi,bbox_inches='tight',facecolor='white')
        paths.append(str(out/(name+'.png')));plt.close(fig)
        captions.append(name+'. '+caption)
    # A result-free graphical abstract: exact computation and feedback paths.
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig,ax=plt.subplots(figsize=(11,4.5));ax.set_xlim(0,11);ax.set_ylim(0,4.5);ax.axis('off')
    nodes=[(.25,2.65,'Observe preview','Warp predictive memory'),
           (4.,2.65,'Plan gaze','Benefit + audit budget'),
           (7.75,2.65,'Acquire one patch','Log its probability'),
           (7.75,.6,'Audit memory','Correct selective sampling'),
           (4.,.6,'Update mixture','Learn from paid observation'),
           (.25,.6,'Carry memory forward','Next frame only')]
    for x,y,title,sub in nodes:
        ax.add_patch(FancyBboxPatch((x,y),3,.95,boxstyle='round,pad=.08',facecolor='#EDF3F8',edgecolor='#365F82',linewidth=1.2))
        ax.text(x+1.5,y+.61,title,ha='center',va='center',fontsize=11,fontweight='bold')
        ax.text(x+1.5,y+.26,sub,ha='center',va='center',fontsize=8.5)
    for start,end in [((3.35,3.1),(3.85,3.1)),((7.1,3.1),(7.6,3.1)),
                      ((9.25,2.5),(9.25,1.7)),((7.6,1.1),(7.1,1.1)),
                      ((3.85,1.1),(3.35,1.1)),((1.75,1.7),(1.75,2.5))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,color='#365F82',linewidth=1.3))
    ax.text(5.5,4.1,'BioSaccade-PEM TwinAudit',ha='center',fontsize=13,fontweight='bold')
    ax.text(5.5,.15,'Current audit uses pre-query weights. Dense reference images enter evaluation only.',ha='center',fontsize=9)
    save(fig,'00_graphical_abstract','Computation order and causal feedback. This schematic contains no empirical performance claim.')
    main=summary[summary.study.isin(['main','reds_train_full','reds_train_detail'])]
    if len(main):
        for (st,ds,reg),g in main.groupby(['study','dataset','regime']):
            fig,axs=plt.subplots(1,2,figsize=(10,4.4))
            for ax,metric,label in [(axs[0],'mse','RGB reconstruction MSE'),(axs[1],'augmented_pqv','Audit quadratic variation / T²')]:
                gg=g.sort_values(metric);y=np.arange(len(gg))
                colors=['#C44E52' if m=='PEM_R9_TwinAudit' else '#4C72B0' for m in gg.method]
                ax.barh(y,gg[metric],color=colors)
                ax.set_yticks(y,[display_name(m) for m in gg.method],fontsize=8)
                ax.set_xlabel(label);ax.set_title(f'{ds} ({st}) - {reg}')
                if metric=='augmented_pqv' and (gg[metric]>0).all():ax.set_xscale('log')
            save(fig,f'01_controlled_policy_{st}_{ds}_{reg}',
                 'Equal sensing budgets and common PEM backbone. Sources are averaged before aggregation; lower is better in both panels. The right panel is evaluator-only predictable quadratic variation, not an online variance certificate.')
    neural=summary[summary.study.astype(str).str.startswith('neural')]
    if len(neural):
        for (st,ds,reg),g in neural.groupby(['study','dataset','regime']):
            fig,ax=plt.subplots(figsize=(9,4.5))
            g=g.sort_values('mse');y=np.arange(len(g))
            colors=['#C44E52' if ('_R9' in m or m=='PEM_R9_TwinAudit') else ('#999999' if 'offline' in m else '#4C72B0') for m in g.method]
            ax.barh(y,g.mse,color=colors)
            ax.set_yticks(y,[display_name(m) for m in g.method],fontsize=8)
            ax.set_title(ds+' - '+('native clean bicubic input' if st=='neural_native' else 'sensor-noise input'));ax.set_xlabel('RGB reconstruction MSE (lower is better)')
            save(fig,f'02_published_backbones_{st}_{ds}_{reg}',
                 'Frozen published checkpoints (NTIRE 2025 ESR; BasicVSR++ CVPR 2022, Vimeo-90K BI) in the common one-patch sensor interface. Patch: SR output plus the paid patch; memory: identical causal residual wrapper; ours: the same backbone with the proposed gaze. Grey: offline BasicVSR++ reads future frames and is a non-causal reference only.')
    if len(controls):
        p=controls[controls.study=='main'].reset_index(drop=True)
        if len(p):
            fig,ax=plt.subplots(figsize=(9,max(3.2,.32*len(p))))
            y=np.arange(len(p));ax.hlines(y,p.ci_low,p.ci_high,color='#4C72B0');ax.scatter(p.reduction_pct,y,color='#4C72B0')
            ax.set_yticks(y,[f'{r.dataset} / {r.regime} / {r.comparator}' for r in p.itertuples()],fontsize=8)
            ax.axvline(0,color='black',linestyle='--',linewidth=1);ax.set_xlabel('R9 primary-audit variance reduction on identical trajectories (%)')
            save(fig,'03_same_gaze_proxy_ablation',
                 'Paired source-cluster 95% percentile bootstrap intervals for ratios of means. Negative values are retained. Holm adjustment applies to p values in the separate table, not these intervals.')
    sweep=summary[summary.study=='kappa']
    if len(sweep):
        for (ds,reg),g in sweep.groupby(['dataset','regime']):
            fig,ax=plt.subplots(figsize=(6,4))
            g=g.sort_values('kappa');ax.plot(g.mse,g.augmented_pqv,'o-',color='#C44E52')
            for row in g.itertuples():ax.annotate(f'κ={row.kappa:g}',(row.mse,row.augmented_pqv),xytext=(4,5),textcoords='offset points',fontsize=8)
            ax.set_xlabel('RGB reconstruction MSE');ax.set_ylabel('Audit quadratic variation / T²');ax.set_yscale('log');ax.set_title(ds+' - '+reg)
            save(fig,f'04_tradeoff_{ds}_{reg}','Prespecified budget sweep. The curve is an empirical trade-off and is not asserted to be a Pareto frontier or monotone in realized reconstruction.')
    main_r9=summary[(summary.study=='main') & (summary.method=='PEM_R9_TwinAudit')]
    if len(main_r9):
        fig,ax=plt.subplots(figsize=(7,4.5))
        labels=[f'{r.dataset} / {r.regime}' for r in main_r9.itertuples()]
        x=np.arange(len(main_r9)); width=.36
        ax.bar(x-width/2,main_r9.augmented_pqv,width,label='memory-vs-preview audit')
        ax.bar(x+width/2,main_r9.write_pqv,width,label='one-step write-value audit')
        ax.set_yscale('log');ax.set_ylabel('Predictable quadratic variation / T²')
        ax.set_xticks(x,labels,rotation=25,ha='right');ax.legend(fontsize=8)
        save(fig,'07_twin_audit_reliability','Full R9 reports two distinct design-based audits from the same paid observations: overall memory benefit and the one-step counterfactual value of the previous foveal write.')
    wk=summary[summary.study=='write_kappa']
    if len(wk):
        for (ds,reg),g in wk.groupby(['dataset','regime']):
            fig,ax=plt.subplots(figsize=(6.2,4.2));g=g.sort_values('write_kappa')
            ax.plot(g.mse,g.write_pqv,'o-')
            for row in g.itertuples():ax.annotate(f'κw={row.write_kappa:g}',(row.mse,row.write_pqv),xytext=(4,5),textcoords='offset points',fontsize=8)
            ax.set_xlabel('RGB reconstruction MSE');ax.set_ylabel('Twin-write audit PQV / T²');ax.set_yscale('log')
            ax.set_title(ds+' - '+reg)
            save(fig,f'08_write_tradeoff_{ds}_{reg}','Prespecified twin-write budget sweep; lower κw enforces stronger auditability of the one-step memory-write effect.')
    (out/'captions.txt').write_text('\n\n'.join(captions),encoding='utf-8')
    return paths


def trace_figures(trace,example,out,synthetic=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(3,1,figsize=(8,8),sharex=True)
    t=trace.frame.to_numpy()+1
    axs[0].fill_between(t,trace.cs_low.to_numpy(),trace.cs_high.to_numpy(),alpha=.2,color='#4C72B0',label='95% anytime confidence sequence')
    axs[0].plot(t,trace.cs_center,label='Audit estimate',color='#4C72B0')
    axs[0].plot(t,trace.cumulative_true_contrast,label='Evaluator reference',color='black',linestyle='--')
    axs[0].set_ylabel('Cumulative average contrast');axs[0].legend(fontsize=8)
    valid=trace.write_valid.astype(bool).to_numpy()
    if valid.any():
        axs[1].plot(t[valid],trace.loc[valid,'cumulative_write_true'],label='Evaluator write value',linestyle='--')
        axs[1].plot(t[valid],trace.loc[valid,'write_cs_center'],label='Twin-write estimate')
        axs[1].fill_between(t[valid],trace.loc[valid,'write_cs_low'],trace.loc[valid,'write_cs_high'],alpha=.2)
        axs[1].legend(fontsize=8)
    axs[1].set_ylabel('One-step write value')
    for key in ['zero','energy','learned']:axs[2].plot(t,trace['weight_'+key],label=key)
    axs[2].set_xlabel('Frame');axs[2].set_ylabel('Predictable mixture weight');axs[2].legend()
    if synthetic:fig.suptitle('Synthetic software check - not benchmark evidence')
    fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(out/('05_online_audit.'+ext),dpi=1000,bbox_inches='tight')
    plt.close(fig)
    if example is not None:
        fig,axs=plt.subplots(1,4,figsize=(12,3.2))
        for ax,k,title in zip(axs,['reference','preview','reconstruction','gaze'],['Reference','Coarse preview','Reconstruction','Gaze probabilities']):
            ax.imshow(example[k],cmap='viridis' if k=='gaze' else None);ax.set_title(title);ax.axis('off')
        if synthetic:fig.suptitle('Synthetic software check - not benchmark evidence')
        fig.tight_layout()
        for ext in ['pdf','png']:fig.savefig(out/('06_example.'+ext),dpi=600,bbox_inches='tight')
        plt.close(fig)
    with (out/'captions.txt').open('a',encoding='utf-8') as f:
        f.write('\n\n05_online_audit. Preselected example; conservative range-based confidence sequence for the running on-trajectory mean. Weights used at frame t are fixed before its foveal query.\n\n06_example. Preselected final frame with identical display scaling; the reference is evaluator-only. A high-DPI export does not add detail to the source image.\n')


def _latex_escape(x):
    x=str(x)
    for a,b in [('\\','\\textbackslash{}'),('&','\\&'),('%','\\%'),('_','\\_'),('#','\\#')]:
        x=x.replace(a,b)
    return x


def write_latex(table,path,caption,label):
    cols=list(table.columns)
    lines=['\\begin{table}[t]','\\centering','\\caption{'+_latex_escape(caption)+'}','\\label{'+label+'}',
           '\\small','\\begin{tabular}{'+'l'*2+'r'*(len(cols)-2)+'}','\\toprule',
           ' & '.join(_latex_escape(c) for c in cols)+' \\\\','\\midrule']
    for row in table.itertuples(index=False):
        cells=[]
        for v in row:
            if isinstance(v,(float,np.floating)):
                cells.append('--' if not np.isfinite(v) else (f'{v:.3g}' if abs(v)<1e-2 and v!=0 else f'{v:.4f}' if abs(v)<10 else f'{v:.2f}'))
            else: cells.append(_latex_escape(v))
        lines.append(' & '.join(cells)+' \\\\')
    lines+=['\\bottomrule','\\end{tabular}','\\end{table}']
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def paper_tables(summary,paired,out,neural_provenance=None):
    """Journal-ready result tables (CSV + booktabs LaTeX). Values are means over
    source clusters; PSNR is the source-averaged per-frame PSNR."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True);written={}
    cols={'display_name':'Method','regime':'Regime','source_clusters':'Sources','clips':'Clips',
          'psnr_frame_mean':'PSNR (dB)','ssim':'SSIM','augmented_pqv':'Audit PQV','write_pqv':'Write PQV'}
    summary=summary.copy()
    if 'write_frames' in summary:
        # Methods without a twin audit have no write-value estimate: show '--', not 0.
        summary.loc[summary.write_frames<=0,'write_pqv']=np.nan
    for study,g in summary.groupby('study'):
        for ds,h in g.groupby('dataset'):
            t=h.sort_values(['regime','psnr_frame_mean'],ascending=[True,False])[list(cols)].rename(columns=cols)
            name=f'table_{study}_{ds}'
            t.to_csv(out/(name+'.csv'),index=False)
            write_latex(t,out/(name+'.tex'),f'{ds}: {study} study. Means over source clusters. Audit PQV is the evaluator-side predictable quadratic variation of the memory-benefit audit divided by T^2 (lower is more reliable).',f'tab:{study}_{ds}')
            written[name]=t
    tim=summary[summary.study=='timing']
    if len(tim):
        t=tim[['display_name','online_total_ms_mean','online_total_ms_p95','controller_ms_mean']].rename(columns={
            'display_name':'Method','online_total_ms_mean':'Mean ms/frame','online_total_ms_p95':'p95 ms/frame','controller_ms_mean':'Controller ms/frame'})
        if neural_provenance:
            params={}
            for n,p in neural_provenance.items(): params[n]=p.get('parameter_count')
            t['SR parameters']=[next((params[k] for k in params if m.startswith(_SR.get(k,'\0'))),np.nan) for m in t.Method]
        t.to_csv(out/'table_latency.csv',index=False)
        write_latex(t,out/'table_latency.tex','Serial per-frame latency (no competing processes): preview/flow, controller and audit, and SR including device transfers.','tab:latency')
        written['table_latency']=t
    if len(paired):
        keep=paired[paired.metric.isin(['psnr_db','mse','augmented_pqv','write_pqv'])].copy()
        keep['Effect']=np.where(keep.effect_type=='difference_dB','dB gain','% reduction')
        keep['metric']=keep.metric.map({'psnr_db':'PSNR','mse':'MSE','augmented_pqv':'Audit PQV','write_pqv':'Write PQV'})
        t=keep[['study','dataset','regime','baseline_display','metric','Effect','reduction_pct','ci_low','ci_high','holm_p']].rename(columns={
            'baseline_display':'Baseline','reduction_pct':'Estimate','ci_low':'95% CI low','ci_high':'95% CI high','holm_p':'Holm p'})
        t.to_csv(out/'table_paired_vs_ours.csv',index=False)
        for study,h in t.groupby('study'):
            write_latex(h.drop(columns=['study']),out/f'table_paired_{study}.tex',
                        f'Paired source-cluster comparisons of TwinAudit (ours) against each baseline ({study}). Positive values favour ours; 95% percentile bootstrap intervals; Holm-adjusted sign-flip p values.',f'tab:paired_{study}')
        written['table_paired_vs_ours']=t
    return written
