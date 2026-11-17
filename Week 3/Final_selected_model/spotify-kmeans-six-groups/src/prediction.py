"""Assign new songs with audio measurements, using the saved fitted model.

The command-line entry point is scripts/04_predict_new_songs.py.
The input must contain all nine audio columns. Missing individual values use
the saved model medians; rows with no usable audio measurements are rejected.
"""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd


def predict(csv_path, model_path, output_path):
    bundle = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    missing = set(bundle['features'])-set(df.columns)
    if missing:
        raise ValueError(f'Missing audio columns: {sorted(missing)}')
    x = df[bundle['features']].apply(pd.to_numeric,errors='coerce')
    x.loc[x.tempo<=0,'tempo'] = np.nan
    x = x.replace([np.inf,-np.inf],np.nan)
    if x.isna().all(axis=1).any():
        raise ValueError('At least one row has no usable audio measurements.')
    bounded = ['danceability','energy','speechiness','acousticness',
               'instrumentalness','liveness','valence']
    if ((x[bounded]<0)|(x[bounded]>1)).any().any():
        raise ValueError('Bounded audio measurements must lie in [0,1].')
    labels = bundle['pipeline'].predict(x)
    df['cluster'] = [bundle['cluster_mapping'][int(v)] for v in labels]
    df['imputed_audio_fields'] = x.isna().sum(axis=1)
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output_path,index=False)
    return df

