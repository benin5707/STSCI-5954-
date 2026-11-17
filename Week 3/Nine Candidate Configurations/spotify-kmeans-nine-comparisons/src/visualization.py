"""Create elbow, nine-candidate comparison, PCA, and group-profile figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

def make_figures(project_root):
    root = Path(project_root)
    R = root/"output"/"tables"
    F = root/"output"/"figures"
    F.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "figure.dpi":150,"savefig.bbox":"tight"})
    colors=["#347B98","#E29D41","#44A78A","#7362A9","#CF6A77","#526D82"]
    grid=pd.read_csv(R/"validation_grid.csv")
    short=pd.read_csv(R/"model_comparison.csv")
    elbow=pd.read_csv(R/"elbow_diagnostics.csv")
    df=pd.read_csv(root/"data"/"processed"/"songs_with_clusters.csv")
    bundle=joblib.load(root/"output"/"models"/"playlist_model.joblib")
    x=df[bundle["features"]].copy()
    x.loc[x.tempo<=0,"tempo"]=np.nan
    z=bundle["pipeline"][:-1].transform(x)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for j,representation in enumerate(['original','pca80','pca90']):
        a=grid[(grid.family=='KMeans') & (grid.representation==representation)].sort_values('k')
        axs[0].plot(a.k,a.silhouette,marker='o',label=representation,color=colors[j])
    axs[0].set(xlabel='Number of clusters',ylabel='Validation silhouette',xticks=[5,6,7],
               title='Nine candidates: validation silhouette')
    axs[0].legend(frameon=False)
    names=[f'{r.family} / {r.representation} / k={r.k}' for r in short.itertuples()]
    y=np.arange(len(short))
    axs[1].barh(y,short.stability_ARI_mean,color=[colors[0] if a>=.8 else '#B4BAC1' for a in short.stability_ARI_mean])
    axs[1].set_yticks(y,labels=names,fontsize=8)
    axs[1].invert_yaxis()
    axs[1].axvline(.8,color='#777777',linestyle='--',linewidth=1)
    axs[1].set(xlim=(0,1.05),xlabel='Mean adjusted Rand index',title='All nine candidates: five subsamples each')
    fig.tight_layout(w_pad=3)
    fig.savefig(F/'model_comparison.png')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.5))
    display_names={'original':'Original standardized 9D',
                   'pca80':'PCA80: 6 training components',
                   'pca90':'PCA90: 7 training components'}
    for j,representation in enumerate(['original','pca80','pca90']):
        e=elbow[elbow.representation.eq(representation)].sort_values('k')
        ax.plot(e.k,e.relative_inertia,marker='o',markersize=4,
                color=colors[j],label=display_names[representation])
    ax.axvspan(5,7,color='#E4EAF0',alpha=.6,zorder=-1)
    ax.set(xlabel='Number of clusters (k)',ylabel='Training inertia / own inertia at k=1',
           title='Elbow diagnostic: focus on k=5, 6, 7',xticks=range(1,11))
    ax.legend(frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(F/'elbow_diagnostic.png')
    plt.close(fig)
    
    feat=bundle['features']
    profiles=df.groupby('cluster')[feat].mean()
    standard=bundle['pipeline'].named_steps['scaler']
    heat=(profiles-standard.mean_)/standard.scale_
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[3,1]})
    im=axs[0].imshow(heat,cmap='RdBu_r',vmin=-2,vmax=2,aspect='auto')
    axs[0].set_xticks(np.arange(len(feat)),labels=feat,rotation=45,ha='right',fontsize=9)
    axs[0].set_yticks(np.arange(6),labels=[f'Group {i}' for i in range(1,7)])
    axs[0].set_title('Audio profiles: mean relative to the dataset')
    fig.colorbar(im,ax=axs[0],label='Standard deviations',shrink=.8)
    sizes=df.groupby('cluster').size()
    axs[1].barh(np.arange(6),sizes,color=colors)
    axs[1].set_yticks(np.arange(6),labels=[f'Group {i}' for i in range(1,7)])
    axs[1].invert_yaxis()
    axs[1].set(xlabel='Songs',title='Group sizes')
    fig.tight_layout()
    fig.savefig(F/'group_profiles.png')
    plt.close(fig)
    
    variance=pd.read_csv(R/'pca_variance.csv')
    fig,ax=plt.subplots(figsize=(6,3.7))
    ax.plot(variance.PC,variance.cumulative,marker='o',color=colors[0])
    ax.axhline(.8,color=colors[1],linestyle='--',label='80%')
    ax.axhline(.9,color=colors[2],linestyle='--',label='90%')
    ax.set(xlabel='Number of principal components',ylabel='Cumulative explained variance',
           ylim=(0,1.05),title='Training PCA: two components retain only 41.0%')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(F/'pca_variance.png')
    plt.close(fig)
    
    # The PCA visualization is descriptive only. Two dimensions were not used to fit.
    from sklearn.decomposition import PCA
    p2=PCA(n_components=2).fit(z)
    sample=np.random.default_rng(42).choice(len(df),min(4000, len(df)),replace=False)
    zz=p2.transform(z[sample])
    fig,ax=plt.subplots(figsize=(7,4.8))
    for i in range(1,7):
        mask=df.iloc[sample].cluster.to_numpy()==i
        ax.scatter(zz[mask,0],zz[mask,1],s=7,alpha=.4,color=colors[i-1],label=f'Group {i}')
    ax.set(xlabel='PC1',ylabel='PC2',title='Overlapping groups in a two-dimensional projection')
    ax.legend(frameon=False,markerscale=2,bbox_to_anchor=(1.01,1),loc='upper left')
    fig.tight_layout()
    fig.savefig(F/'pca_group_projection.png')
    plt.close(fig)
    
    print(f"Figures saved to {F}")


