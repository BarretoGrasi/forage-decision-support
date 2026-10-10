"""Figures computed from the consolidated execution; no hard-coded silhouettes."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

KEYS = ['Gramínea (Principal)', 'Leguminosa (Acompanhante)']
OBJECTIVES = ['suitability_media', 'compatibilidade_ambiental', 'complementaridade_funcional']
STYLE = {'font.family': 'serif', 'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42}

def save(fig, folder, name):
    for ext in ('png', 'pdf'):
        fig.savefig(folder / f'{name}.{ext}', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def validate_matrix(matrix):
    d = np.asarray(matrix, dtype=float)
    if d.shape != (111, 111) or not np.isfinite(d).all():
        raise ValueError(f'Expected a finite 111 x 111 matrix; got {d.shape}')
    if not np.allclose(d, d.T, atol=1e-7) or not np.allclose(np.diag(d), 0, atol=1e-7):
        raise ValueError('Dissimilarity matrix must be symmetric with zero diagonal')
    return d

def generate_figures(state, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    pairs, front, _, _ = state['resultados_pareto'][state['CENARIO_REF']]
    for table in (pairs, front):
        if table.empty or table[KEYS].isna().any().any() or table[KEYS].duplicated().any():
            raise ValueError('Empty table, missing pair identity, or duplicate pair')
        if not np.isfinite(table[OBJECTIVES].to_numpy(dtype=float)).all():
            raise ValueError('Non-finite pair objectives')
    p = pairs.set_index(KEYS)[OBJECTIVES]; f = front.set_index(KEYS)[OBJECTIVES]
    if not f.index.isin(p.index).all() or not np.allclose(p.loc[f.index], f, rtol=0, atol=1e-12):
        raise ValueError('Frontier identities/objectives disagree with eligible pairs')
    mask = pairs.set_index(KEYS).index.isin(f.index)
    dominated, efficient = pairs.loc[~mask], pairs.loc[mask]
    counts = (len(pairs), len(efficient), len(dominated))
    if counts != (126, 10, 116):
        raise ValueError(f'Consolidated reference mismatch: {counts}; expected (126, 10, 116)')
    projections = [(0,1), (0,2), (1,2)]
    labels = [r'$S_{\mathrm{pair}}$', r'$C_{\mathrm{env}}$', r'$C_{\mathrm{fun}}$']
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1,3,figsize=(12,4))
        for i, (ax, (x,y)) in enumerate(zip(axes,projections)):
            ax.scatter(dominated[OBJECTIVES[x]],dominated[OBJECTIVES[y]],s=25,color='#999999',alpha=.6,edgecolors='none')
            ax.scatter(efficient[OBJECTIVES[x]],efficient[OBJECTIVES[y]],s=50,color='#D55E00',edgecolors='black',linewidths=.4)
            ax.set(xlabel=labels[x],ylabel=labels[y]); ax.grid(linestyle=':',alpha=.35);ax.set_axisbelow(True);ax.margins(.08)
            ax.text(.03,.97,f'({chr(97+i)})',transform=ax.transAxes,va='top',fontweight='bold')
        handles=[Line2D([],[],marker='o',linestyle='none',color='#999999',label=f'Dominated ({len(dominated)})'),Line2D([],[],marker='o',linestyle='none',markerfacecolor='#D55E00',markeredgecolor='black',label=f'Pareto-efficient ({len(efficient)})')]
        fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.9));save(fig,directory,'pareto_objective_space')
    general = validate_matrix(state['matriz_distancia']); environmental = validate_matrix(state['matriz_distancia_ambiental'])
    scores=[]
    for k in range(2,7):
        cluster=AgglomerativeClustering(n_clusters=k,metric='precomputed',linkage='complete').fit_predict(general)
        scores.append(silhouette_score(general,cluster,metric='precomputed'))
    pd.DataFrame({'k':range(2,7),'silhouette':scores}).to_csv(directory/'silhouette_values.csv',index=False)
    best=int(np.argmax(scores));ks=list(range(2,7))
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(6.5,4));ax.plot(ks,scores,color='#0072B2',marker='o',linewidth=1.8)
        ax.annotate(f'$k={ks[best]}$\n{scores[best]:.4f}',xy=(ks[best],scores[best]),xytext=(18,-18),textcoords='offset points',arrowprops={'arrowstyle':'->'},va='top')
        ax.set(xlabel='Number of clusters ($k$)',ylabel='Mean silhouette coefficient',xticks=ks,xlim=(1.8,6.2));ax.grid(axis='y',linestyle=':',alpha=.4);fig.tight_layout();save(fig,directory,'silhouette_111_records')
    idx=np.triu_indices(len(general),k=1);x,y=general[idx],environmental[idx]
    r=float(pearsonr(x,y).statistic);rho=float(spearmanr(x,y).statistic)
    pd.DataFrame([{'records':len(general),'unique_pairs':len(x),'pearson_r':r,'spearman_rho':rho}]).to_csv(directory/'gower_correlations.csv',index=False)
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(6.5,5));ax.scatter(x,y,s=8,color='#0072B2',alpha=.25,edgecolors='none',rasterized=True)
        ax.plot([0,1],[0,1],'--',color='#555555',label='Identity line');ax.set(xlabel='General Gower dissimilarity (27 descriptors)',ylabel='Environmental Gower dissimilarity (24 descriptors)',xlim=(0,1),ylim=(0,1));ax.set_aspect('equal')
        ax.text(.04,.96,f'111 records; {len(x):,} unique pairs\nPearson $r$ = {r:.4f}\nSpearman $\\rho$ = {rho:.4f}',transform=ax.transAxes,va='top',bbox={'facecolor':'white','edgecolor':'none','alpha':.9});ax.legend(loc='lower right',frameon=False);fig.tight_layout();save(fig,directory,'gower_comparison_111_records')
    return {'eligible':counts[0],'efficient':counts[1],'dominated':counts[2],'best_k':ks[best],'silhouette':scores[best],'pearson_r':r,'spearman_rho':rho}
