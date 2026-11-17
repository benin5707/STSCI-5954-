"""Write English reports for the fixed six-group K-means specification."""
from pathlib import Path
import html
import json
import pandas as pd

DESCRIPTIONS=["Low energy and higher acousticness",
              "Stronger spoken-word or rap characteristics",
              "High danceability and higher audio valence",
              "Higher instrumentalness","Higher liveness",
              "Higher energy and faster tempo"]

def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |",
                      "| "+" | ".join(["---"]*len(headers))+" |",
                      *["| "+" | ".join(str(v) for v in row)+" |" for row in rows]])

def generate_report(project_root):
    root=Path(project_root)
    folder=root/"reports"
    folder.mkdir(parents=True,exist_ok=True)
    s=json.loads((root/"output"/"summary.json").read_text())
    profiles=pd.read_csv(root/"output"/"tables"/"cluster_profiles.csv")
    repeats=pd.read_csv(root/"output"/"tables"/"stability_repeats.csv")
    groups=table(["Group","Tracks","Share","Audio description"],
                 [[int(r.cluster),f"{r.songs:,}",f"{r.share:.1%}",DESCRIPTIONS[int(r.cluster)-1]]
                  for r in profiles.itertuples()])
    metrics=table(["Metric","Validation","Test"],
                  [[label,f"{s['validation'][key]:.4f}",f"{s['test'][key]:.4f}"]
                   for key,label in [("silhouette","Silhouette (higher is better)"),
                                     ("db","Davies-Bouldin (lower is better)"),
                                     ("ch","Calinski-Harabasz (higher is better)")]])
    stability=table(["Seed","Training songs","ARI"],
                    [[r.seed,r.training_songs,f"{r.ARI:.4f}"] for r in repeats.itertuples()])
    report=f"""# Six-Group K-Means: Analysis Report

## Fixed specification

This package implements one model specification: median imputation, standardization
of nine original continuous audio features, and K-means with k=6, n_init=20,
and random_state=42. The specification was fixed before running this package.
There is no representation search, algorithm search, or cluster-count selection.
A separate training-only elbow diagnostic supplies context without changing k=6.

The model is fitted at different stages for validation, test evaluation, stability
checks, and full-data delivery. Those repeated fits use the same specification;
they are not alternative candidate configurations.

## Training-only elbow diagnostic

![Training-only elbow curve](../output/figures/elbow_diagnostic.png)

The curve uses original standardized nine-dimensional training features only.
For k=1 through 10, K-means uses n_init=20 and seed 42. The vertical axis is
inertia divided by its value at k=1. The orange marker identifies the retained
six-group specification; it is not an automatically detected knee.

The curve gradually flattens and has no uniquely sharp elbow. It provides
supporting context, while the earlier nine-scheme comparison used validation
silhouette and stability to retain original standardized features with k=6.
This single-model package does not repeat that candidate selection. Other k
values appear only in this diagnostic and receive no validation or test scores.
The table is [elbow_diagnostics.csv](../output/tables/elbow_diagnostics.csv).

## Data and preprocessing

The supplied tracks_clean.csv has {s['input_rows']:,} rows and
{s['input_columns']} columns, with {s['unique_songs']:,} unique track IDs.
It is consumed directly in the supplied row order. No deduplication, averaging,
or row filtering is performed. Input checks verify required fields, unique
nonmissing IDs, and finite numeric audio values, raising an error if the input
does not meet the clean-table contract.

The source is already deduplicated but is not standardized. It has no playlist
columns, so no original playlist-membership table is reconstructed or exported.
The exact supplied file is stored unchanged in data/raw/tracks_clean.csv;
the raw directory denotes the input as supplied, not an uncleaned source.

The nine features are {", ".join(s['features'])}.
Popularity, IDs, and identifying metadata do not enter clustering.
Four tracks with missing metadata remain eligible for audio modeling.
The one nonpositive tempo value is treated as missing. Other valid zeros and
extreme but valid measurements are retained. The input has no missing audio
measurements apart from the nonpositive tempo that is marked unavailable.
Malformed or nonfinite audio input is rejected rather than silently cleaned.

SimpleImputer uses fitted feature medians; StandardScaler uses fitted feature
means and standard deviations. The feature space is not reduced.

## Data partitions and fitting order

| Partition | Tracks | Role |
| --- | ---: | --- |
| Training | {s['splits']['train']:,} | Fit preprocessing and the reference K-means model |
| Validation | {s['splits']['validation']:,} | Describe fixed-model grouping quality |
| Test | {s['splits']['test']:,} | Evaluate the fixed specification after refitting |

The seed is 42. The two-stage split reproduces the earlier song-level split.
Each track_id belongs to exactly one partition.

1. Fit medians, scaler, and K-means on training songs.
2. Predict validation assignments and calculate validation metrics.
3. Check stability using five 80% training subsamples.
4. Refit preprocessing and K-means on training plus validation.
5. Predict the held-out test songs and calculate test metrics.
6. Fit the delivery model on all tracks and export assignments and profiles.

Validation does not select among candidate specifications in this package.
The test data do not participate in the earlier fitting stages.
Full-data delivery outputs are not independent test evidence.
Splitting by track_id does not guarantee separation of recording families,
duplicate releases, alternate versions, or artists.

## Grouping quality

{metrics}

Each metric uses a fixed random sample of 2,500 songs from its partition and the
appropriate fitted standardized nine-dimensional feature space.
The assignment ARI between the training-only model and the model refitted on
training plus validation is {s['test']['ARI_to_train_only_model']:.4f}, calculated
on all {s['splits']['test']:,} test songs.

Silhouette around 0.15 indicates substantial overlap. ARI describes agreement
between two groupings, not emotion-classification accuracy.

## Stability

Each repeat samples 80% of the training songs without replacement, refits
imputation, scaling, and K-means, and predicts the same full validation set.
ARI compares each repeat with the reference training model.
The initialization seed also changes across repeats.

{stability}

Mean stability ARI is {s['stability']['ARI_mean']:.4f}.
This is subsampling rather than bootstrap sampling.
A stable result does not prove that groups match listener moods or activities.

## Full-data delivery groups

{groups}

Delivery group numbers are ordered by mean energy, then valence, after the
full-data fit. They differ from raw internal estimator labels.
Group profiles include each feature's mean and median.

Higher instrumentalness does not prove calmness or a pure instrumental recording.
Higher liveness does not confirm a live recording. Audio valence is a measured
feature, not a human-verified mood target. Client listening is needed before
assigning final mood or activity names.

## Listening examples and prediction

The representative-song table contains up to ten close-to-center songs per
group, with different artists where possible. The boundary table contains up to
three songs per group with small relative gaps between the nearest two centers.
These are geometric examples, not confidence-calibrated emotion predictions.

The listening-review table is blank and awaits client input. No listening
feedback is claimed. Track links are provided without checking regional
playability.

The prediction script requires all nine feature columns and reuses the saved
delivery pipeline. It fills individual missing values using saved medians,
rejects rows with no usable audio measurements, and checks bounded features.
The supplied six-song example is drawn from the existing catalog; it is a
functional example rather than a new independent test.

## Figures

![Group profiles and sizes](../output/figures/group_profiles.png)

![Stability repeats](../output/figures/stability.png)

![Training-feature correlations](../output/figures/feature_correlation.png)

## Reproduction and files

See [README.md](../README.md) for the directory tree, numbered scripts,
dependencies, reproduction commands, and file descriptions.

The supplied clean CSV, client brief, processed data, model, metrics, profiles, example
songs, and figures are included. Song and playlist names retain their original
source language; documentation, comments, and filenames are in English.

## Sources

- [Supplied clean CSV](../data/raw/tracks_clean.csv).
- [Supplied client brief](../docs/client_brief.pdf).
- https://scikit-learn.org/stable/modules/clustering.html#k-means
- https://scikit-learn.org/stable/common_pitfalls.html

The client brief requests natural song groupings and listening examples.
The preprocessing choices, split proportions, and fixed model settings are
project decisions, not instructions quoted from the brief.
"""
    (folder/"analysis_report.md").write_text(report,encoding="utf8")
    summary_text=f"""Fixed-model workflow

Read tracks_clean.csv directly: 28,356 rows and 19 columns, with unique IDs.
No deduplication, row filtering, or playlist reconstruction is performed.
Use nine original continuous audio features.
Split into training (17,013), validation (5,671), and test (5,672).
Fit median imputation, StandardScaler, and six-group K-means on training.
Add a training-only k=1 to 10 elbow curve for context, without selecting another k.
Compute validation metrics and five 80% subsampling stability repeats.
Refit on training plus validation, then evaluate the held-out test set.
Refit on all tracks for delivery and save the model and song assignments.

Validation silhouette: {s['validation']['silhouette']:.4f}
Test silhouette: {s['test']['silhouette']:.4f}
Mean stability ARI: {s['stability']['ARI_mean']:.4f}

Only one fixed model specification is implemented.
No automatic mood labels or supervised classifier are implemented.
"""
    (folder/"workflow_summary.txt").write_text(summary_text,encoding="utf8")
    items=[("Analysis report","analysis_report.md"),("README","../README.md"),
           ("Training-only elbow data","../output/tables/elbow_diagnostics.csv"),
           ("Validation metrics","../output/tables/validation_metrics.csv"),
           ("Test metrics","../output/tables/test_metrics.csv"),
           ("Stability repeats","../output/tables/stability_repeats.csv"),
           ("Group profiles","../output/tables/cluster_profiles.csv"),
           ("All assignments","../data/processed/songs_with_clusters.csv")]
    links="".join(f'<li><a href="{url}">{label}</a></li>' for label,url in items)
    images="".join(f'<figure><img src="../output/figures/{f}.png" alt="{name}"><figcaption>{name}</figcaption></figure>'
                   for f,name in [("elbow_diagnostic","Training-only elbow diagnostic"),
                                  ("group_profiles","Group profiles and sizes"),
                                  ("stability","Stability repeats"),
                                  ("feature_correlation","Training feature correlations")])
    page=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Six-Group K-Means</title><style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;color:#203047}}img{{max-width:100%}}figure{{margin:28px 0}}a{{color:#19639a}}</style>
</head><body><h1>Six-Group K-Means</h1><p>Nine original standardized audio features.
Validation silhouette: {s['validation']['silhouette']:.4f};
test silhouette: {s['test']['silhouette']:.4f};
mean stability ARI: {s['stability']['ARI_mean']:.4f}.</p>
<h2>Results</h2><ul>{links}</ul><h2>Figures</h2>{images}</body></html>"""
    (folder/"index.html").write_text(page,encoding="utf8")
    print(f"English reports saved to {folder}")

