"""Additional analyses supplied by the authors; use the original reference pool."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from figures import KEYS, OBJECTIVES, STYLE, save
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def pair_ids(table):
    return set(table[KEYS].itertuples(index=False,name=None))

def run_supplementary(state, directory):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    pairs=state['resultados_pareto'][state['CENARIO_REF']][0].copy()
    if len(pairs)!=126: raise ValueError('Expected 126 reference eligible pairs')
    front_fn=state['calcular_fronteira_pareto'];rank_fn=state['score_pesos'];weights=state['PESOS_BASE']
    ref=front_fn(pairs,OBJECTIVES);ref_ids=pair_ids(ref)
    if len(ref)!=10:raise ValueError('Expected ten reference efficient pairs')
    # Environmental shared-zero encoding: reconstruct original numeric Gower first.
    columns=state['features_gower_ambiental'];climate=list(state['colunas_clima_total'])
    base=state['_df_gower_base'][columns].copy();values=base.to_numpy(dtype=float)
    if len(base)!=111 or len(climate)!=20 or len(columns)!=24 or not np.isfinite(values).all():
        raise ValueError('Expected 111 records and 24 finite environmental descriptors')
    binary=base[climate].to_numpy(dtype=float)
    if not np.isin(binary,[0,1]).all():raise ValueError('Climate descriptors must be binary')
    numerator=np.zeros((111,111))
    for col in columns:
        v=base[col].to_numpy(dtype=float);span=np.ptp(v)
        if span<=0:raise ValueError(f'Constant descriptor: {col}')
        numerator+=np.abs(v[:,None]-v[None,:])/span
    if not np.allclose(numerator/24,state['matriz_distancia_ambiental'],atol=1e-6,rtol=0):
        raise ValueError('Numeric Gower reconstruction failed')
    zero=(binary==0).astype(int);denom=24-zero@zero.T
    alternative_d=numerator/denom
    records=state['df']
    if records['especie'].duplicated().any():raise ValueError('Duplicate names: use record IDs')
    positions=dict(zip(records['especie'],records['pos_matriz'].astype(int)))
    alt=pairs.copy();alt['compatibilidade_ambiental']=[round(float(np.clip(1-alternative_d[positions[g],positions[l]],0,1)),6) for g,l in alt[KEYS].itertuples(index=False,name=None)]
    alt_rank=rank_fn(front_fn(alt,OBJECTIVES),weights);alt_ids=pair_ids(alt_rank)
    jac=len(ref_ids&alt_ids)/len(ref_ids|alt_ids)
    rows=[]
    for name,table,rank in [('Original',pairs,rank_fn(ref,weights)),('Exclude shared zeros',alt,alt_rank)]:
        lead=rank.iloc[0];rows.append({'version':name,'eligible':len(table),'efficient':len(rank),'grass':lead[KEYS[0]],'legume':lead[KEYS[1]],'preference_score':lead['score_preferencia']})
    pd.DataFrame(rows).to_csv(out/'sensibilidade_cenv_resumo.csv',index=False)
    alt.to_csv(out/'sensibilidade_cenv_pares.csv',index=False);alt_rank.to_csv(out/'sensibilidade_cenv_fronteira.csv',index=False)
    # Min-max ranges come from ALL 126 eligible pairs, not just the frontier.
    minima=pairs[OBJECTIVES].min();maxima=pairs[OBJECTIVES].max();spans=maxima-minima
    if (spans<=0).any():raise ValueError('Constant objective')
    pd.DataFrame({'Mínimo':minima,'Máximo':maxima,'Amplitude':spans}).to_csv(out/'mc_normalizado_limites.csv')
    norm=pairs.copy();norm[OBJECTIVES]=(pairs[OBJECTIVES]-minima)/spans
    norm_front=front_fn(norm,OBJECTIVES)
    if pair_ids(norm_front)!=ref_ids:raise ValueError('Normalisation changed frontier')
    original=ref.sort_values(KEYS).reset_index(drop=True);normalised=norm_front.sort_values(KEYS).reset_index(drop=True)
    F=original[OBJECTIVES].to_numpy(dtype=float);N=normalised[OBJECTIVES].to_numpy(dtype=float)
    W=np.random.default_rng(20260903).dirichlet(np.ones(3),size=10000)
    a,b=100*(W@F.T),100*(W@N.T)
    for scores in (a,b):
        if (np.isclose(scores,scores.max(axis=1,keepdims=True),atol=1e-10,rtol=0).sum(axis=1)>1).any():
            raise ValueError('Sampled maximum ties require explicit handling')
    ia,ib=np.argmax(a,axis=1),np.argmax(b,axis=1)
    ca,cb=np.bincount(ia,minlength=10),np.bincount(ib,minlength=10)
    mc=original[KEYS].copy();mc['Vitórias — original']=ca;mc['Top-1 original (%)']=ca/100;mc['Vitórias — normalizada']=cb;mc['Top-1 normalizada (%)']=cb/100;mc['Diferença (pontos percentuais)']=(cb-ca)/100
    mc.sort_values('Vitórias — normalizada',ascending=False).to_csv(out/'mc_normalizado_comparacao.csv',index=False)
    w=np.asarray(weights,dtype=float);w/=w.sum();leaders=[]
    for name,matrix in [('Original',F),('Normalised',N)]:
        scores=100*(matrix@w);i=int(np.argmax(scores));leaders.append({'version':name,KEYS[0]:original.iloc[i][KEYS[0]],KEYS[1]:original.iloc[i][KEYS[1]],'score':scores[i]})
    pd.DataFrame(leaders).to_csv(out/'mc_normalizado_lider_baseline.csv',index=False)
    # Individual weights change joint suitability; screening and post-Pareto weights stay fixed.
    ep=[]
    for label,we,wp in [('50/50',.5,.5),('60/40',.6,.4),('70/30',.7,.3)]:
        scenario=state['CENARIOS'][state['CENARIO_REF']]
        assessed=state['avaliar_cenario'](scenario,w_eco=we,w_prod=wp)
        all_pairs,_=state['montar_pares'](assessed,None,scenario.get('arranjo','consorcio_pastejo'))
        eligible=all_pairs[all_pairs['status_par']=='apto'].copy()
        efficient=front_fn(eligible,OBJECTIVES);ids=pair_ids(efficient);ranking=rank_fn(efficient,weights)
        if len(eligible)!=126:raise ValueError('Individual weighting changed eligible pool')
        if label=='60/40' and ids!=ref_ids:raise ValueError('60/40 frontier mismatch')
        lead=ranking.iloc[0];ep.append({'weights':label,'eligible':len(eligible),'efficient':len(efficient),'jaccard':len(ids&ref_ids)/len(ids|ref_ids),'grass':lead[KEYS[0]],'legume':lead[KEYS[1]],'joint_suitability':lead['suitability_media'],'preference_score':lead['score_preferencia']})
        ranking.to_csv(out/f"sensibilidade_ecoprod_fronteira_{label.replace('/','_')}.csv",index=False)
    pd.DataFrame(ep).to_csv(out/'sensibilidade_ecoprod_resumo.csv',index=False)
    summary={'shared_zero_frontier_jaccard':jac,'shared_zero_leader':rows[1],'monte_carlo_seed':20260903,'monte_carlo_samples':10000,'same_sampled_winner_percent':float(np.mean(ia==ib)*100),'ecoprod':ep}
    (out/'supplementary_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

def scenario_figure(state,directory):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    configs=[('Baseline (Cerrado, pH 5.0, sem calagem)','Cerrado without liming',(126,10)),('Baseline (Cerrado, pH 5.0, com calagem)','Cerrado with liming',(131,15)),('Tropical ácido (pH 4.5, sem calagem)','Acidic tropical',(41,6)),('Subtropical (pH 6.0)','Subtropical',(10,6))]
    data=[]
    for name,title,expected in configs:
        pairs,front,_,_=state['resultados_pareto'][name]
        for table in (pairs,front):
            if table.empty or table[KEYS].isna().any().any() or table[KEYS].duplicated().any() or not np.isfinite(table[OBJECTIVES].to_numpy(dtype=float)).all():raise ValueError(f'Invalid scenario table: {name}')
        p=pairs.set_index(KEYS)[OBJECTIVES];f=front.set_index(KEYS)[OBJECTIVES]
        if not f.index.isin(p.index).all() or not np.allclose(p.loc[f.index],f,atol=1e-12,rtol=0):raise ValueError(f'Frontier mismatch: {name}')
        if (len(pairs),len(front))!=expected:raise ValueError(f'Scenario counts mismatch: {name}')
        mask=pairs.set_index(KEYS).index.isin(f.index);data.append((title,pairs,pairs.loc[~mask],pairs.loc[mask]))
    def limits(v):
        lo,hi=np.min(v),np.max(v);m=max((hi-lo)*.08,.01);return lo-m,hi+m
    x=np.concatenate([p[OBJECTIVES[0]] for _,p,_,_ in data]);y=np.concatenate([p[OBJECTIVES[1]] for _,p,_,_ in data])
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
        for i,(ax,(title,pairs,dom,eff)) in enumerate(zip(axes.flat,data)):
            ax.scatter(dom[OBJECTIVES[0]],dom[OBJECTIVES[1]],s=25,color='#999999',alpha=.6,edgecolors='none')
            ax.scatter(eff[OBJECTIVES[0]],eff[OBJECTIVES[1]],s=50,color='#D55E00',edgecolors='black',linewidths=.4)
            ax.set(title=f'({chr(97+i)}) {title}\n{len(pairs)} eligible; {len(eff)} Pareto-efficient',xlabel=r'$S_{\mathrm{pair}}$',ylabel=r'$C_{\mathrm{env}}$',xlim=limits(x),ylim=limits(y));ax.grid(linestyle=':',alpha=.35);ax.set_axisbelow(True)
        handles=[Line2D([],[],marker='o',linestyle='none',color='#999999',label='Dominated'),Line2D([],[],marker='o',linestyle='none',markerfacecolor='#D55E00',markeredgecolor='black',label='Pareto-efficient')]
        fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.94));save(fig,out,'scenario_fronts')
