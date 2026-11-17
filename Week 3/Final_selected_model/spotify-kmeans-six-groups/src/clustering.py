"""Fit and evaluate one fixed six-group K-means specification."""
from pathlib import Path
import argparse
import json
import hashlib
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
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
N_CLUSTERS = 6

def audio_frame(df):
    x = df[FEATURES].apply(pd.to_numeric, errors='raise').copy()
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('The cleaned input must have finite, complete numeric audio features.')
    # One supplied track still has tempo=0; retain its row and impute this field.
    x.loc[x.tempo <= 0, 'tempo'] = np.nan
    return x

def preprocessing():
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())])

def quality(x, labels):
    if len(np.unique(labels)) < 2:
        return dict(silhouette=np.nan, db=np.nan, ch=np.nan)
    return dict(silhouette=float(silhouette_score(x, labels)),
                db=float(davies_bouldin_score(x, labels)),
                ch=float(calinski_harabasz_score(x, labels)))

def build_pipeline(x, seed=SEED):
    prep = preprocessing()
    z = prep.fit_transform(x)
    model = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=seed)
    with warnings.catch_warnings(record=True) as ws:
        model.fit(z)
    return Pipeline([*prep.steps, ('cluster', model)]), [str(w.message) for w in ws]

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
    # Consume the provided clean table directly; no deduplication or row filtering.
    df = pd.read_csv(csv_path)
    required = ['track_id','track_name','track_artist','track_album_name',*FEATURES]
    missing_columns = set(required)-set(df.columns)
    if missing_columns:
        raise ValueError(f'Missing required columns: {sorted(missing_columns)}')
    if df.track_id.isna().any() or df.track_id.duplicated().any():
        raise ValueError('Expected the cleaned input to have nonmissing, unique track_id values.')
    input_rows, input_column_count = df.shape
    input_sha256 = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    x = audio_frame(df)
    trainval, test = train_test_split(np.arange(len(df)), test_size=.2, random_state=SEED)
    train, val = train_test_split(trainval, test_size=.25, random_state=SEED)
    splits = np.full(len(df), 'train', dtype=object)
    splits[val], splits[test] = 'validation', 'test'
    df[['track_id']].assign(split=splits).to_csv(destination('split_manifest.csv'), index=False)
    # The specification is fixed; no model search is performed.
    selected_train_pipe, training_warnings = build_pipeline(x.iloc[train])
    # Context only: training inertia for other k values does not select a model.
    training_z = selected_train_pipe[:-1].transform(x.iloc[train])
    inertia_values = []
    for diagnostic_k in range(1, 11):
        if diagnostic_k == N_CLUSTERS:
            diagnostic_model = selected_train_pipe.named_steps['cluster']
        else:
            diagnostic_model = KMeans(n_clusters=diagnostic_k, n_init=20,
                                      random_state=SEED).fit(training_z)
        inertia_values.append(float(diagnostic_model.inertia_))
    pd.DataFrame([
        dict(k=i+1, inertia=value, relative_inertia=value/inertia_values[0],
             relative_gain_percent=(None if i==0 else
                 100*(inertia_values[i-1]-value)/inertia_values[i-1]),
             selected_specification=(i+1 == N_CLUSTERS),
             training_songs=len(train))
        for i, value in enumerate(inertia_values)
    ]).to_csv(destination('elbow_diagnostics.csv'), index=False)
    validation_labels = selected_train_pipe.predict(x.iloc[val])
    ve = np.random.default_rng(SEED).choice(len(val), size=min(2500,len(val)), replace=False)
    common_val = selected_train_pipe[:-1].transform(x.iloc[val])
    validation = quality(common_val[ve], validation_labels[ve])
    validation['sample_size'] = int(len(ve))
    validation['cluster_counts'] = np.bincount(validation_labels,minlength=N_CLUSTERS).tolist()
    pd.DataFrame([{'family':'KMeans','representation':'original','k':N_CLUSTERS,
                   'dimensions':len(FEATURES), **{k:v for k,v in validation.items()
                   if k!='cluster_counts'}}]).to_csv(destination('validation_metrics.csv'),index=False)
    pd.DataFrame(selected_train_pipe[:-1].transform(x.iloc[train]),
                 columns=FEATURES).corr().to_csv(destination('audio_correlation_train.csv'))

    repeats = []
    for seed in range(100,105):
        sub = np.random.default_rng(seed).choice(train,size=int(.8*len(train)),replace=False)
        pipe, repeat_warnings = build_pipeline(x.iloc[sub],seed)
        labels = pipe.predict(x.iloc[val])
        repeats.append(dict(seed=seed, training_songs=int(len(sub)),
                            ARI=float(adjusted_rand_score(validation_labels,labels)),
                            min_cluster_share=float(np.bincount(labels,minlength=N_CLUSTERS).min()/len(val)),
                            warnings=' | '.join(repeat_warnings)))
    pd.DataFrame(repeats).to_csv(destination('stability_repeats.csv'),index=False)
    aris = [r['ARI'] for r in repeats]
    stability = dict(ARI_mean=float(np.mean(aris)), ARI_min=float(min(aris)),
                     ARI_sd=float(np.std(aris)), repeats=len(repeats),
                     sample_fraction=.8, sampling='without replacement',
                     minimum_cluster_share=float(min(r['min_cluster_share'] for r in repeats)))
    pd.DataFrame([stability]).to_csv(destination('stability_summary.csv'),index=False)
    chosen = {'family':'KMeans','representation':'original','k':N_CLUSTERS}

    # Freeze selected specification; test never used for candidate selection.
    evaluation_pipe, fit_warnings = build_pipeline(x.iloc[trainval])
    test_labels = evaluation_pipe.predict(x.iloc[test])
    eval_common_prep = preprocessing().fit(x.iloc[trainval])
    common_test = eval_common_prep.transform(x.iloc[test])
    te = np.random.default_rng(SEED).choice(len(test),size=min(2500,len(test)),replace=False)
    test_quality = quality(common_test[te],test_labels[te])
    test_quality['cluster_counts'] = np.bincount(test_labels,minlength=N_CLUSTERS).tolist()
    test_quality['ARI_to_train_only_model'] = float(adjusted_rand_score(
        selected_train_pipe.predict(x.iloc[test]),test_labels))
    # Production fit: after the single held-out evaluation, refit on all songs.
    final_pipe, final_warnings = build_pipeline(x)
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
    pd.DataFrame([dict(split='test', **test_quality)]).drop(columns='cluster_counts').to_csv(
        destination('test_metrics.csv'),index=False)
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
    summary = dict(input_file=Path(csv_path).name,input_sha256=input_sha256,
                   input_rows=input_rows,input_columns=input_column_count,
                   unique_songs=len(df),precleaned_input=True,deduplication_applied=False,
                   missing_metadata_songs=int((df.track_name=='[missing]').sum()),
                   zero_tempo_songs=int((df.tempo<=0).sum()),
                   splits={s:int((splits==s).sum()) for s in ['train','validation','test']},
                   features=FEATURES,seed=SEED,model_configurations=1,
                   elbow_diagnostic_k=list(range(1,11)),
                   elbow_scope='Training-only context; no automatic model selection',
                   specification={**chosen,'dimensions':len(FEATURES),'n_init':20},
                   validation=validation,stability=stability,test=test_quality,
                   label_mapping=mapping,training_warnings=training_warnings,
                   fit_warnings=fit_warnings,final_warnings=final_warnings,
                   versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,
                                 numpy=np.__version__,pandas=pd.__version__))
    destination('summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

