"""Draw figures for the fixed six-group K-means analysis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

def make_figures(project_root):
    root=Path(project_root)
    folder=root/"output"/"figures"
    folder.mkdir(parents=True,exist_ok=True)
    tables=root/"output"/"tables"
    summary=json.loads((root/"output"/"summary.json").read_text())
    profiles=pd.read_csv(tables/"cluster_profiles.csv").sort_values("cluster")
    repeats=pd.read_csv(tables/"stability_repeats.csv")
    model=joblib.load(root/"output"/"models"/"playlist_model.joblib")
    features=model["features"]
    scaler=model["pipeline"].named_steps["scaler"]
    means=profiles[[f"{f}_mean" for f in features]].to_numpy()
    heat=(means-scaler.mean_)/scaler.scale_
    colors=["#347B98","#E29D41","#44A78A","#7362A9","#CF6A77","#526D82"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "figure.dpi":150,"savefig.bbox":"tight"})
    elbow = pd.read_csv(tables/"elbow_diagnostics.csv").sort_values("k")
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(elbow.k, elbow.relative_inertia, "o-", color=colors[0],
            label="Original standardized 9D")
    chosen = elbow[elbow.selected_specification]
    ax.scatter(chosen.k, chosen.relative_inertia, s=90, color=colors[1],
               zorder=3, label="Retained specification: k=6")
    ax.axvline(6, color=colors[1], linestyle="--", linewidth=1, alpha=.7)
    ax.set(xlabel="Number of clusters (k)",
           ylabel="Training inertia / inertia at k=1",
           title="Training-only elbow diagnostic", xticks=range(1,11))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(folder/"elbow_diagnostic.png")
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={"width_ratios":[3,1]})
    im=axs[0].imshow(heat,cmap="RdBu_r",vmin=-3,vmax=3,aspect="auto")
    axs[0].set_xticks(np.arange(len(features)),labels=features,rotation=45,ha="right",fontsize=9)
    group_names=[f"Group {int(g)}" for g in profiles.cluster]
    axs[0].set_yticks(np.arange(6),labels=group_names)
    axs[0].set_title("Full-data audio profiles")
    fig.colorbar(im,ax=axs[0],label="Standard deviations",shrink=.8)
    axs[1].barh(np.arange(6),profiles.songs,color=colors)
    axs[1].set_yticks(np.arange(6),labels=group_names)
    axs[1].invert_yaxis()
    axs[1].set(xlabel="Songs",title="Group sizes")
    fig.tight_layout()
    fig.savefig(folder/"group_profiles.png")
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(np.arange(len(repeats)),repeats.ARI,color=colors[0])
    ax.axhline(summary["stability"]["ARI_mean"],color=colors[1],linestyle="--",
               label=f'Mean ARI = {summary["stability"]["ARI_mean"]:.4f}')
    ax.set_xticks(np.arange(len(repeats)),[f"Seed {s}" for s in repeats.seed])
    ax.set(ylim=(0,1.02),ylabel="Adjusted Rand index",
           title="Six-group K-means: five 80% training subsamples")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder/"stability.png")
    plt.close(fig)

    corr=pd.read_csv(tables/"audio_correlation_train.csv",index_col=0)
    fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap="RdBu_r")
    ax.set_xticks(range(9),features,rotation=45,ha="right")
    ax.set_yticks(range(9),features)
    ax.set_title("Training audio-feature correlations")
    fig.colorbar(im,ax=ax,label="Correlation",shrink=.8)
    fig.tight_layout()
    fig.savefig(folder/"feature_correlation.png")
    plt.close(fig)
    print(f"Figures saved to {folder}")

