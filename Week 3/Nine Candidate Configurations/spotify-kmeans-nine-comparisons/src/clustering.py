"""Nine K-means comparisons with training-only elbow and PCA diagnostics.

All selection uses training/validation songs only. The test set is evaluated
once after selection. Audio geometry is assessed in the same standardized
9-feature space for every candidate, including PCA candidates.
"""
from pathlib import Path
import argparse
import json
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             davies_bouldin_score, calinski_harabasz_score)
from threadpoolctl import threadpool_limits

FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
SEED = 42


def audio_frame(df):
    x = df[FEATURES].apply(pd.to_numeric, errors='coerce').copy()
    # Zero BPM is treated as an unavailable tempo measurement (one song).
    # Other valid zero audio measurements, including instrumentalness, stay zero.
    x.loc[x.tempo <= 0, 'tempo'] = np.nan
    return x.replace([np.inf, -np.inf], np.nan)


def preprocessing(representation):
    steps = [('imputer', SimpleImputer(strategy='median')),
             ('scaler', StandardScaler())]
    if representation != 'original':
        steps.append(('pca', PCA(n_components=float(representation[3:]) / 100,
                                  svd_solver='full', whiten=False)))
    return Pipeline(steps)


def estimator(family, k, seed=SEED):
    if family != 'KMeans' or int(k) not in (5,6,7):
        raise ValueError('Candidate comparison is restricted to KMeans and k=5,6,7.')
    return KMeans(n_clusters=int(k), n_init=20, random_state=seed)


def quality(common_x, labels, distances=None):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return dict(silhouette=np.nan, db=np.nan, ch=np.nan)
    return dict(silhouette=float(silhouette_score(
                    distances if distances is not None else common_x,
                    labels, metric='precomputed' if distances is not None else 'euclidean')),
                db=float(davies_bouldin_score(common_x, labels)),
                ch=float(calinski_harabasz_score(common_x, labels)))


def build_pipeline(row, x, seed=SEED):
    prep = preprocessing(row['representation'])
    z = prep.fit_transform(x)
    model = estimator(row['family'], row['k'], seed)
    with warnings.catch_warnings(record=True) as ws:
        model.fit(z)
    pipe = Pipeline([*prep.steps, ('cluster', model)])
    return pipe, [str(w.message) for w in ws]


def main(csv_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    processed = output/'data'/'processed'
    tables = output/'output'/'tables'
    models = output/'output'/'models'
    for folder in (processed, tables, models):
        folder.mkdir(parents=True, exist_ok=True)

    def destination(filename):
        if filename in {'songs_with_clusters.csv', 'split_manifest.csv'}:
            return processed/filename
        if filename == 'playlist_model.joblib':
            return models/filename
        if filename == 'summary.json':
            return output/'output'/filename
        return tables/filename
    # The supplied catalog is already cleaned: preserve every input row.
    df = pd.read_csv(csv_path)
    required = {'track_id', 'track_name', 'track_artist', 'track_album_name', *FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df.track_id.isna().any() or df.track_id.duplicated().any():
        raise ValueError('The cleaned input must contain unique, nonmissing track_id values.')
    x = audio_frame(df)
    trainval, test = train_test_split(np.arange(len(df)), test_size=.2, random_state=SEED)
    train, val = train_test_split(trainval, test_size=.25, random_state=SEED)
    splits = np.full(len(df), 'train', dtype=object)
    splits[val], splits[test] = 'validation', 'test'
    df[['track_id']].assign(split=splits).to_csv(destination('split_manifest.csv'), index=False)
    # One fixed validation sample, shared by every candidate.
    rng = np.random.default_rng(SEED)
    ve = rng.choice(len(val), size=min(2500, len(val)), replace=False)
    common_prep = preprocessing('original').fit(x.iloc[train])
    common_val = common_prep.transform(x.iloc[val])
    cv = common_val[ve]
    distances = squareform(pdist(cv))
    pca = PCA().fit(common_prep.transform(x.iloc[train]))
    pd.DataFrame({'PC':np.arange(1,len(FEATURES)+1),
                  'variance_ratio':pca.explained_variance_ratio_,
                  'cumulative':np.cumsum(pca.explained_variance_ratio_)}).to_csv(destination('pca_variance.csv'),index=False)
    pd.DataFrame(pca.components_.T, index=FEATURES,
                  columns=[f'PC{i+1}' for i in range(len(FEATURES))]).to_csv(destination('pca_axes.csv'))
    pd.DataFrame(common_prep.transform(x.iloc[train]),columns=FEATURES).corr().to_csv(destination('audio_correlation_train.csv'))
    records, fitted, elbow_records, representation_records = [], {}, [], []
    for representation in ['original', 'pca80', 'pca90']:
        prep = preprocessing(representation).fit(x.iloc[train])
        zt, zv = prep.transform(x.iloc[train]), prep.transform(x.iloc[val])
        explained = (1.0 if representation == 'original' else
                     float(prep.named_steps['pca'].explained_variance_ratio_.sum()))
        representation_records.append(dict(representation=representation,
            dimensions=zt.shape[1], retained_training_variance=explained))
        # Exactly these three k values enter validation comparison and selection.
        candidate_models = {}
        for k in [5,6,7]:
            start = time.time()
            model = estimator('KMeans',k)
            with warnings.catch_warnings(record=True) as ws:
                model.fit(zt)
            candidate_models[k] = model
            labels = model.predict(zv)
            counts = np.bincount(labels,minlength=k)
            train_counts = np.bincount(model.predict(zt),minlength=k)
            row = dict(family='KMeans',representation=representation,k=k,
                dimensions=zt.shape[1],**quality(cv,labels[ve],distances),
                val_min_share=float(counts.min()/len(val)),
                train_min_share=float(train_counts.min()/len(train)),
                seconds=round(time.time()-start,3),warnings=' | '.join(str(w.message) for w in ws))
            row['eligible'] = row['val_min_share'] >= .01 and row['train_min_share'] >= .01
            records.append(row)
            fitted[(representation,k)] = Pipeline([*prep.steps,('cluster',model)])
            print(f"Candidate {representation}, k={k}: silhouette={row['silhouette']:.4f}",flush=True)

        # Wider training-only curve supplies context for the focused candidate set.
        # k outside 5,6,7 receives no validation/test score and cannot be selected.
        inertias = []
        for elbow_k in range(1,11):
            diagnostic = candidate_models.get(elbow_k)
            if diagnostic is None:
                diagnostic = KMeans(n_clusters=elbow_k,n_init=20,random_state=SEED).fit(zt)
            inertias.append(float(diagnostic.inertia_))
        for index,inertia in enumerate(inertias):
            elbow_records.append(dict(representation=representation,k=index+1,
                dimensions=zt.shape[1],inertia=inertia,
                relative_inertia=inertia/inertias[0],
                relative_gain_percent=(None if index==0 else
                    100*(inertias[index-1]-inertia)/inertias[index-1]),
                comparison_candidate=(index+1 in [5,6,7])))
    results = pd.DataFrame(records).sort_values('silhouette',ascending=False)
    results.to_csv(destination('validation_grid.csv'),index=False)
    pd.DataFrame(elbow_records).to_csv(destination('elbow_diagnostics.csv'),index=False)
    pd.DataFrame(representation_records).to_csv(destination('representation_summary.csv'),index=False)

    stability, repeat_records = [], []
    # Run the same stability protocol for all nine candidates for a complete table.
    for _,row in results.iterrows():
        ref = fitted[(row.representation,int(row.k))].predict(x.iloc[val])
        aris,sizes = [],[]
        for seed in range(100,105):
            sub = np.random.default_rng(seed).choice(train,size=int(.8*len(train)),replace=False)
            pipe,ws = build_pipeline(row,x.iloc[sub],seed)
            labels = pipe.predict(x.iloc[val])
            ari = float(adjusted_rand_score(ref,labels))
            minimum_share = float(np.bincount(labels,minlength=int(row.k)).min()/len(val))
            aris.append(ari)
            sizes.append(minimum_share)
            repeat_records.append(dict(representation=row.representation,k=int(row.k),
                seed=seed,training_songs=int(len(sub)),ARI=ari,
                min_cluster_share=minimum_share,warnings=' | '.join(ws)))
        stability.append({**row.to_dict(),'stability_ARI_mean':float(np.mean(aris)),
            'stability_ARI_min':float(min(aris)),'stability_ARI_sd':float(np.std(aris)),
            'resample_min_cluster_share':float(min(sizes)),'ARI_repeats':json.dumps(aris)})
    comparison = pd.DataFrame(stability).sort_values('silhouette',ascending=False)
    stable = comparison.loc[comparison.eligible & (comparison.stability_ARI_mean>=.8)]
    if stable.empty:
        raise RuntimeError('No candidate passed the size and stability criteria.')
    chosen = stable.iloc[0]
    comparison['selected'] = ((comparison.representation==chosen.representation)
                               & (comparison.k==chosen.k))
    comparison.to_csv(destination('model_comparison.csv'),index=False)
    pd.DataFrame(repeat_records).to_csv(destination('stability_repeats.csv'),index=False)
    selected_train_pipe = fitted[(chosen.representation,int(chosen.k))]
    # Freeze selected specification; test never used for candidate selection.
    evaluation_pipe, fit_warnings = build_pipeline(chosen,x.iloc[trainval])
    test_labels = evaluation_pipe.predict(x.iloc[test])
    eval_common_prep = preprocessing('original').fit(x.iloc[trainval])
    common_test = eval_common_prep.transform(x.iloc[test])
    te = np.random.default_rng(SEED).choice(len(test),size=min(2500,len(test)),replace=False)
    test_quality = quality(common_test[te],test_labels[te])
    test_quality['cluster_counts'] = np.bincount(test_labels,minlength=int(chosen.k)).tolist()
    test_quality['ARI_to_train_only_model'] = float(adjusted_rand_score(
        selected_train_pipe.predict(x.iloc[test]),test_labels))
    # Production fit: after the single held-out evaluation, refit on all songs.
    final_pipe, final_warnings = build_pipeline(chosen,x)
    final_labels = final_pipe.predict(x)
    final_model = final_pipe.named_steps['cluster']
    # Canonical labels ordered by mean energy, then valence for an easier handoff.
    order = (df.assign(old=final_labels).groupby('old')[['energy','valence']]
               .mean().sort_values(['energy','valence']).index)
    mapping = {int(old):new+1 for new,old in enumerate(order)}
    df['cluster'] = [mapping[int(c)] for c in final_labels]
    df['split'] = splits
    df['spotify_url'] = 'https://open.spotify.com/track/' + df.track_id
    # Display-only missing metadata: retains useful audio rows without inventing identity.
    for col in ['track_name','track_artist','track_album_name']:
        df[col] = df[col].fillna('[missing]')
    df.to_csv(destination('songs_with_clusters.csv'),index=False)
    profile = df.groupby('cluster')[FEATURES].agg(['mean','median'])
    profile.columns = ['_'.join(c) for c in profile.columns]
    profile.insert(0,'songs',df.groupby('cluster').size())
    profile.insert(1,'share',profile.songs/len(df))
    profile.to_csv(destination('cluster_profiles.csv'))
    transformed = final_pipe[:-1].transform(x)
    centers = final_model.cluster_centers_
    # Distance to own model center; representative is geometric, not validated mood.
    df['distance_to_center'] = np.linalg.norm(transformed-centers[final_labels],axis=1)
    representative = (df.loc[df.track_name.ne('[missing]') & df.track_artist.ne('[missing]')]
                        .sort_values('distance_to_center')
                        .drop_duplicates(['cluster','track_artist'])
                        .groupby('cluster',sort=True).head(10)
                        .sort_values(['cluster','distance_to_center']))
    representative[['cluster','track_id','track_name','track_artist','spotify_url',
                    'distance_to_center',*FEATURES]].to_csv(destination('representative_songs.csv'),index=False)
    # Boundary tracks for client listening, drawn from the two nearest centers.
    distances_to_centers = np.linalg.norm(transformed[:,None,:]-centers[None,:,:],axis=2)
    two = np.sort(distances_to_centers,axis=1)[:,:2]
    df['relative_center_gap'] = (two[:,1]-two[:,0])/np.maximum(two[:,1],1e-12)
    (df.loc[df.track_name.ne('[missing]')].sort_values('relative_center_gap')
       .groupby('cluster').head(3).sort_values(['cluster','relative_center_gap'])
       [['cluster','track_id','track_name','track_artist','spotify_url','relative_center_gap']]
       .to_csv(destination('boundary_songs.csv'),index=False))
    representative[['cluster','track_name','track_artist','spotify_url']].assign(
        proposed_name='',fit_rating_1_to_5='',suggested_other_cluster='',notes=''
    ).to_csv(destination('client_listening_review.csv'),index=False)
    joblib.dump({'pipeline':final_pipe,'features':FEATURES,'cluster_mapping':mapping,
                 'specification':{k:chosen[k] for k in ['family','representation','k']}},
                destination('playlist_model.joblib'))
    summary = dict(source_file=Path(csv_path).name,raw_rows=len(df),unique_songs=len(df),duplicates_removed=0,
                   input_already_cleaned=True,missing_metadata_songs=int((df.track_name=='[missing]').sum()),
                   zero_tempo_songs=int((df.tempo<=0).sum()),
                   splits={s:int((splits==s).sum()) for s in ['train','validation','test']},
                   features=FEATURES,seed=SEED,model_grid_candidates=len(results),
                   elbow_diagnostic_k=list(range(1,11)),
                   elbow_scope='Training-only; outside-range diagnostic fits are not comparison candidates',
                   selection=chosen.to_dict(),test=test_quality,
                   train_pca_cumulative=np.cumsum(pca.explained_variance_ratio_).tolist(),
                   label_mapping=mapping,fit_warnings=fit_warnings,final_warnings=final_warnings,
                   versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,
                                 numpy=np.__version__,pandas=pd.__version__))
    (destination('summary.json')).write_text(json.dumps(summary,indent=2,default=lambda v:v.item()),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',required=True)
    parser.add_argument('--output',default='.', help='Project root for organized outputs')
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.csv,args.output)


