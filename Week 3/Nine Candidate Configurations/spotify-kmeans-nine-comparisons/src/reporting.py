"""Generate English documentation for the nine-candidate K-means comparison."""
from pathlib import Path
import json
import pandas as pd

NAMES={"original":"Original standardized 9D","pca80":"PCA80","pca90":"PCA90"}
GROUPS=["Low energy and higher acousticness",
        "Stronger spoken-word or rap characteristics",
        "High danceability and higher audio valence",
        "Higher instrumentalness","Higher liveness","Higher energy and faster tempo"]

def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |",
                      "| "+" | ".join(["---"]*len(headers))+" |",
                      *["| "+" | ".join(str(v) for v in row)+" |" for row in rows]])

def generate_report(project_root):
    root=Path(project_root)
    reports=root/"reports"
    reports.mkdir(parents=True,exist_ok=True)
    t=root/"output"/"tables"
    summary=json.loads((root/"output"/"summary.json").read_text())
    comparison=pd.read_csv(t/"model_comparison.csv")
    variance=pd.read_csv(t/"pca_variance.csv")
    representation=pd.read_csv(t/"representation_summary.csv")
    profiles=pd.read_csv(t/"cluster_profiles.csv")
    selection=summary["selection"]
    test=summary["test"]
    ordered=comparison.assign(order=comparison.representation.map({"original":0,"pca80":1,"pca90":2})).sort_values(["order","k"])
    rows=[]
    for r in ordered.itertuples():
        status=("Selected" if r.selected else
                "Eligible" if r.eligible and r.stability_ARI_mean>=.8 else "Not eligible")
        rows.append([NAMES[r.representation],int(r.k),int(r.dimensions),
                     f"{r.silhouette:.4f}",f"{r.db:.4f}",f"{r.ch:.2f}",
                     f"{r.stability_ARI_mean:.4f}",status])
    comparison_table=table(["Representation","k","Dimensions","Silhouette ↑","DB ↓","CH ↑","Mean stability ARI ↑","Status"],rows)
    pca_table=table(["Representation","Dimensions","Retained training variance"],
        [[NAMES[r.representation],int(r.dimensions),f"{r.retained_training_variance:.2%}"] for r in representation.itertuples()])
    component_table=table(["PC","Individual variance","Cumulative variance"],
        [[int(r.PC),f"{r.variance_ratio:.2%}",f"{r.cumulative:.2%}"] for r in variance.itertuples()])
    group_table=table(["Group","Tracks","Share","Descriptive audio profile"],
        [[int(r.cluster),f"{r.songs:,}",f"{r.share:.1%}",GROUPS[int(r.cluster)-1]] for r in profiles.itertuples()])
    test_table=table(["Test metric","Value"],
        [["Silhouette",f"{test['silhouette']:.4f}"],
         ["Davies-Bouldin",f"{test['db']:.4f}"],
         ["Calinski-Harabasz",f"{test['ch']:.2f}"],
         ["Assignment ARI: training-only vs refitted",f"{test['ARI_to_train_only_model']:.4f}"]])
    readme=f"""# Spotify K-Means: Nine Comparisons

A focused comparison for the STSCI 5954 playlist project:

- Three representations: original standardized nine features, PCA80, PCA90.
- One algorithm: K-means.
- Three cluster counts: k=5, 6, 7.
- Exactly nine configurations enter validation scoring and final selection.

**Selected configuration: {NAMES[selection['representation']]} + K-means,
k={int(selection['k'])}.** Validation silhouette:
{selection['silhouette']:.4f}; mean stability ARI:
{selection['stability_ARI_mean']:.4f}; test silhouette:
{test['silhouette']:.4f}.

All source data, code, results, figures, reports, and the selected delivery model
are included. Documentation, comments, and filenames are in English.

## Nine-scheme comparison

All rows below use the same 2,500-song validation sample and the same original
standardized nine-dimensional evaluation space, even when the model is fitted
in a PCA representation. This keeps geometric scores comparable.

{comparison_table}

Silhouette is the primary ranking metric after the group-size and stability
requirements. DB and CH are reported as supporting diagnostics. They need not
rank candidates in the same order.

All nine candidates have at least 1% membership in each training and validation
group and mean stability ARI at least 0.8. The original nine-dimensional k=6
configuration has the highest validation silhouette. The k=5 candidates are
slightly more stable, but the k=6 choice has better silhouette while remaining
stable. At k=6, neither PCA option improves on the original representation.
This does not establish that PCA is always worse or that the small differences
are statistically significant.

![Nine-candidate comparison](output/figures/model_comparison.png)

Detailed numerical files:
[validation_grid.csv](output/tables/validation_grid.csv),
[model_comparison.csv](output/tables/model_comparison.csv), and
[stability_repeats.csv](output/tables/stability_repeats.csv).
The comparison table contains all nine configurations; the repeat table contains
five runs for each configuration, for 45 stability records.

## Elbow diagnostic and candidate range

The training-only elbow plot gives context for the focused k=5, 6, 7 comparison.
The curves flatten gradually and do not have a sharp unique knee. The compact
range is a project decision informed by that diagnostic, the earlier exploration,
and the project discussion about at least five groups. It is not a mathematically
unique range returned by an automated elbow detector.

![Training-only elbow curves](output/figures/elbow_diagnostic.png)

For a useful reference curve, the elbow diagnostic evaluates training inertia
for k=1 through 10 in each representation. These are diagnostic fits only.
They are separate from the nine formally compared configurations: k outside
5, 6, 7 receives no validation or test score and cannot be selected.

Inertia is measured in each representation's own fitted space. The plot divides
each curve by its own k=1 inertia so its flattening is easier to inspect;
curve heights do not show that one representation has better separation than
another. Validation metrics use the shared nine-dimensional space instead.

The table [elbow_diagnostics.csv](output/tables/elbow_diagnostics.csv) includes
30 training-only diagnostic records and explicitly flags the nine comparison
candidates. Candidate fits are reused when generating the diagnostic.

This is a focused follow-up to an earlier analysis of the same data. It does
not retrospectively claim that this narrower range was the original search
plan, or that the elbow plot independently proves k=6 is optimal.

## PCA results

PCA is fitted after median imputation and standardization, using training data
only during comparison.

{pca_table}

{component_table}

PCA80 means retaining at least 80% of training variance; it requires six principal
components here and retains 84.39%. PCA90 requires seven components and retains
91.34%. The first two components explain only 40.97%, so two dimensions are not
used to fit the compared clustering models.

The number of principal components and the number of clusters are different:
six components do not imply six groups. The nine comparisons vary k independently.

![Training PCA variance](output/figures/pca_variance.png)

The full loadings are in [pca_axes.csv](output/tables/pca_axes.csv).
A loading's sign is arbitrary; inspect large absolute loadings to see which
original features contribute to each axis. PCA maximizes retained variance,
rather than directly maximizing cluster separation.

The [two-dimensional group projection](output/figures/pca_group_projection.png)
is descriptive: it fits a separate two-component PCA to the standardized
full-data delivery representation and plots 4,000 songs. Its axes are not the
training PCA used for the explained-variance table or model selection.

## Data preparation and evaluation

The supplied tracks_clean.csv contains 28,356 tracks and 19 columns.
It is read directly without repeating deduplication or raw-data cleaning.
Track IDs are checked for uniqueness; all input rows are retained.
Playlist membership fields are absent, so no playlist-membership export is made.
Different track IDs remain even when title and artist match.

The nine continuous input features are:

```text
danceability, energy, loudness, speechiness, acousticness,
instrumentalness, liveness, valence, tempo
```

Nonpositive tempo and unusable numeric values are treated as missing.
Other valid zeros and extreme values remain. SimpleImputer uses fitted medians;
StandardScaler uses fitted means and standard deviations.
Four tracks with missing identifying metadata remain in audio modeling but are
excluded from representative examples where their name or artist is missing.
Popularity, playlist genre, and identifying metadata are not clustering inputs.

| Partition | Tracks | Purpose |
| --- | ---: | --- |
| Training | 17,013 | Fit preprocessing, PCA, and candidate models |
| Validation | 5,671 | Compare the nine candidates and their stability |
| Test | 5,672 | Reproduce the frozen selected-model evaluation |

Seed 42 reproduces the original two-stage split. No track_id crosses partitions.
Recording-family and artist overlap may remain across the track-ID split.

### Selection and stability

All candidate fits use KMeans(n_init=20, random_state=42).
Each representation fits its own training preprocessing. Candidate eligibility
requires every group to contain at least 1% of training and validation songs.
For each of the nine configurations, five runs sample 80% of training songs
without replacement, refit preprocessing and K-means, and predict the same full
validation set. Seeds 100 through 104 vary both subsampling and initialization.

Mean ARI must be at least 0.8. Among candidates passing the size and stability
criteria, choose the highest validation silhouette. These cutoffs are project
rules, not general statistical laws. No true mood labels are used.

### Test and full-data delivery

After selection, refit preprocessing and clustering on training plus validation.
Compute silhouette, DB, and CH on a fixed 2,500-song test sample, and assignment
ARI on all 5,672 test songs. Then fit the delivery model on all 28,356 songs.

{test_table}

This rerun reproduces the prior held-out evaluation on the same split; it is
not a newly collected independent test. The code preserves the separation of
training, validation, test evaluation, and full-data delivery.
High ARI measures reproducibility. Silhouette around 0.15 still indicates
overlapping groups. Neither is mood-classification accuracy.

### Full-data group profiles

{group_table}

Group numbering is ordered by full-data mean energy, then valence. The saved
model includes the mapping from raw estimator labels to delivery groups.
Client listening is still required before naming groups for moods or activities.

![Group profiles](output/figures/group_profiles.png)

## Project structure

```text
spotify-kmeans-nine-comparisons/
├── README.md
├── requirements.txt
├── .gitignore
├── run_all.py
├── src/
│   ├── __init__.py
│   ├── clustering.py
│   ├── visualization.py
│   ├── reporting.py
│   └── prediction.py
├── scripts/
│   ├── 01_fit_clusters.py
│   ├── 02_make_figures.py
│   ├── 03_generate_report.py
│   └── 04_predict_new_songs.py
├── data/
│   ├── raw/
│   │   └── tracks_clean.csv
│   ├── processed/
│   │   ├── songs_with_clusters.csv
│   │   └── split_manifest.csv
│   └── examples/
│       └── sample_new_songs.csv
├── docs/
│   └── client_brief.pdf
├── output/
│   ├── summary.json
│   ├── tables/
│   │   ├── validation_grid.csv
│   │   ├── model_comparison.csv
│   │   ├── stability_repeats.csv
│   │   ├── elbow_diagnostics.csv
│   │   ├── representation_summary.csv
│   │   ├── pca_variance.csv
│   │   ├── pca_axes.csv
│   │   ├── audio_correlation_train.csv
│   │   ├── cluster_profiles.csv
│   │   ├── representative_songs.csv
│   │   ├── boundary_songs.csv
│   │   └── client_listening_review.csv
│   ├── figures/
│   │   ├── elbow_diagnostic.png
│   │   ├── model_comparison.png
│   │   ├── pca_variance.png
│   │   ├── pca_group_projection.png
│   │   └── group_profiles.png
│   ├── models/
│   │   └── playlist_model.joblib
│   └── examples/
│       └── sample_predictions.csv
├── reports/
│   ├── analysis_report.md
│   ├── workflow_summary.txt
│   ├── index.html
│   └── verification.md
└── temp/
    └── .gitkeep
```

## Code structure

| Stage | Script | Reusable code and responsibility |
| --- | --- | --- |
| 1 | scripts/01_fit_clusters.py | src/clustering.py: load cleaned data/split, training PCA and elbow diagnostics, nine-candidate comparison, 45 stability repeats, final evaluation and delivery |
| 2 | scripts/02_make_figures.py | src/visualization.py: render five figures from saved outputs |
| 3 | scripts/03_generate_report.py | src/reporting.py: generate this README, Markdown report, English summary, and HTML results page |
| 4 | scripts/04_predict_new_songs.py | src/prediction.py: assign compatible song rows with the selected delivery model |
| All | run_all.py | Execute stages 1–3 in order; optionally run the stage-4 demonstration |

In src/clustering.py, audio_frame prepares numeric features; preprocessing
builds imputation, scaling, and optional PCA; estimator creates K-means only
for candidate k values 5, 6, 7; quality computes comparable metrics;
build_pipeline refits one configuration; main coordinates diagnostics,
selection, evaluation, and output exports.

Only the selected full-data model is saved as a reusable model bundle.
All nine candidate results and stability repeats are saved as tables.
The code can reproduce candidate models from the included data and settings.

## Reproduction

Use Python 3.12 and the pinned requirements. The included model was fitted using
scikit-learn 1.8.0.

```bash
python -m venv .venv
```

Activate with the appropriate command:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\\Scripts\\Activate.ps1
```

```bash
python -m pip install -r requirements.txt
python run_all.py --include-demo
```

The full run recomputes the nine comparisons, training elbow diagnostics,
stability, selected-model test metrics, delivery outputs, figures, and English
documentation. Script 01 creates a fresh blank listening-review form, so
preserve completed client feedback before refitting.

To regenerate figures and documentation from included results:

```bash
python run_all.py --skip-fit
```

This leaves the model, comparison tables, assignments, and listening-review
table untouched.

To run individual stages:

```bash
python scripts/01_fit_clusters.py
python scripts/02_make_figures.py
python scripts/03_generate_report.py
python scripts/04_predict_new_songs.py
```

To predict another song file:

```bash
python scripts/04_predict_new_songs.py --csv path/to/new_songs.csv --output output/new_song_predictions.csv
```

To write a separate reproduction:

```bash
python run_all.py --project-root temp/reproduction --include-demo
```

Default inputs resolve from the source repository, regardless of the shell's
working directory. Use --csv to select another compatible source dataset.

## Output file guide

| File | Contents |
| --- | --- |
| output/tables/validation_grid.csv | Exactly nine candidate rows: representation, k, dimensions, validation metrics, group sizes, warnings |
| output/tables/model_comparison.csv | Those nine candidates with stability summaries and the selected flag |
| output/tables/stability_repeats.csv | All 45 individual subsampling runs |
| output/tables/elbow_diagnostics.csv | Thirty training-only inertia records, normalized curves, gains, candidate flags |
| output/tables/representation_summary.csv | Actual fitted dimensions and retained training variance |
| output/tables/pca_variance.csv | Individual and cumulative training variance for all PCs |
| output/tables/pca_axes.csv | Training PCA loadings |
| output/tables/audio_correlation_train.csv | Training-feature correlations |
| output/tables/cluster_profiles.csv | Selected delivery-model group sizes, means, and medians |
| output/tables/representative_songs.csv | Up to ten close-to-center songs per group, with artist diversity |
| output/tables/boundary_songs.csv | Up to three boundary examples per group |
| output/tables/client_listening_review.csv | Blank client naming and fit-review template |
| data/processed/songs_with_clusters.csv | All unique tracks, selected delivery groups, splits, and links |
| data/processed/split_manifest.csv | Track-level train/validation/test membership |
| output/summary.json | Counts, nine-candidate scope, elbow scope, chosen configuration, test results, mappings, versions |
| output/models/playlist_model.joblib | Selected full-data pipeline, feature order, specification, label mapping |
| output/examples/sample_predictions.csv | Predictions for the supplied six-song demonstration |
| reports/ | English report, workflow summary, HTML page, and verification |

The example songs are sampled from the source catalog. Their predictions verify
functionality, not independent performance. New-song input needs all nine audio
feature columns; individual missing measurements use saved medians. Completely
unusable rows and invalid bounded values are rejected. No audio-file feature
extraction or external data collection is implemented.

## Documentation, sources, and GitHub upload

See [analysis report](reports/analysis_report.md),
[workflow summary](reports/workflow_summary.txt),
[HTML results page](reports/index.html), and
[verification](reports/verification.md).

The [client brief](docs/client_brief.pdf) and unmodified
[cleaned CSV](data/raw/tracks_clean.csv) are included.
The brief asks for natural audio-based groupings and listening examples; exact
preprocessing, split ratios, algorithm settings, and thresholds are project
choices. No complete course grading rubric is assumed.

Song titles, artist names, and playlist names preserve their original source
language. English documentation does not alter source metadata.

Extract the ZIP and upload the contents of spotify-kmeans-nine-comparisons/
with README.md at the repository root. Data and analysis outputs are
intentionally included. Temporary runs, caches, and virtual environments are
ignored. No repository is automatically published.
"""
    (root/"README.md").write_text(readme,encoding="utf8")
    relative_comparison=comparison_table
    report=f"""# Analysis Report: Nine K-Means Comparisons

## Scope and selection

The formal comparison consists of three representations and k=5, 6, 7:
nine K-means configurations. Training-only elbow curves provide contextual
diagnostics. They do not automatically or uniquely identify the candidate range.
This is a focused follow-up on the same previously analyzed data.

{relative_comparison}

The selected candidate is {NAMES[selection['representation']]}, k={int(selection['k'])}.
The selection rule is highest validation silhouette after at least 1% group
membership in training and validation and mean subsampling ARI at least 0.8.
All nine pass those requirements. The five-group models are slightly more
stable, but six groups improve silhouette while remaining stable. Supporting
DB and CH metrics have different preferences.

## Elbow interpretation

![Training elbow curves](../output/figures/elbow_diagnostic.png)

Training inertia decreases gradually rather than showing one uniquely sharp
knee. The project uses the compact 5–7 range as a practical focused comparison.
Thirty diagnostic training fits span k=1–10 across the three representations;
only the flagged nine k=5,6,7 configurations enter validation and selection.
Normalized curve heights are not cross-representation quality scores.

## PCA

{pca_table}

{component_table}

Six principal components retain 84.39% of training variance; seven retain 91.34%.
The first two explain only 40.97%. Component count and cluster count are
independent design choices. PCA maximizes explained variance, not silhouette.

![Training PCA variance](../output/figures/pca_variance.png)

At k=6, PCA does not improve validation silhouette over original standardized
features in this dataset. The score differences are small and are not supported
by a formal significance test. Loadings are provided in
[pca_axes.csv](../output/tables/pca_axes.csv).

## Data, split, and evaluation

The supplied tracks_clean.csv contains 28,356 tracks and 19 columns.
All rows are used directly, without repeated deduplication. Playlist fields
are not supplied and no playlist-membership file is generated. Different IDs
with the same title and artist remain. The nine features are {", ".join(summary['features'])}.
One nonpositive tempo becomes missing, other valid zeros remain, and four
metadata-missing tracks remain available for audio modeling.

Training, validation, and test sizes are 17,013, 5,671, and 5,672, using seed 42.
Imputation, scaling, PCA, and candidate clustering fit training songs only.
All validation metrics use the same 2,500 validation songs in the original
standardized nine-dimensional space.

Each candidate receives five 80% training subsamples without replacement.
Every repeat refits its own preprocessing and model, with seeds 100–104.
ARI compares assignments on the same full validation set with the reference
training model. These 45 runs include variation in both samples and
initialization. This is subsampling, not bootstrap sampling.

After freezing the chosen configuration, the model is refitted on training plus
validation, evaluated on the test set, and finally fitted on all songs for delivery.

{test_table}

The first three test metrics use 2,500 test songs; assignment ARI uses all 5,672.
This reproduces the earlier held-out evaluation rather than introducing a new
independent test sample. High stability does not establish mood accuracy,
and silhouette around 0.15 shows group overlap.

## Delivery groups and listening

{group_table}

![Candidate comparison](../output/figures/model_comparison.png)

![Delivery group profiles](../output/figures/group_profiles.png)

Representative songs are close to their group's center; boundary examples have
small relative gaps between their nearest two centers. They are geometric
examples, not calibrated emotion-confidence scores. Client review fields
remain blank. High instrumentalness or liveness alone does not prove calmness,
a pure instrumental track, or a live recording.

![Descriptive group projection](../output/figures/pca_group_projection.png)

The two-dimensional projection is a separate full-data descriptive PCA fit,
not the training PCA in the variance table. It displays 4,000 sampled songs
and was not used to fit the selected clusters.

## Reproduction and limitations

Read [README.md](../README.md) for code structure, the complete directory tree,
commands, and artifact descriptions. The selected full-data model supports
new rows with the same nine measurements. The example input is sampled from
the existing catalog and is not a new test set.

No true mood or activity labels, completed listening feedback, online feature
collection, or supervised classification are claimed. Duplicate releases and
artist overlap may cross the song-ID split.

Sources: [cleaned CSV](../data/raw/tracks_clean.csv),
[client brief](../docs/client_brief.pdf), and the included executable source.
"""
    (reports/"analysis_report.md").write_text(report,encoding="utf8")
    (reports/"workflow_summary.txt").write_text(
        "Nine K-means comparisons\n\n"
        "Three representations: original standardized 9D, PCA80, PCA90.\n"
        "Candidate cluster counts: 5, 6, 7.\n"
        "Training-only elbow diagnostic: k=1 through 10; no unique sharp elbow.\n"
        "PCA80: six components, 84.39% variance. PCA90: seven, 91.34%.\n"
        "Five stability subsamples per candidate: 45 repeats.\n"
        f"Selected: {NAMES[selection['representation']]}, k={int(selection['k'])}.\n"
        f"Validation silhouette: {selection['silhouette']:.4f}.\n"
        f"Test silhouette: {test['silhouette']:.4f}.\n"
        f"Mean stability ARI: {selection['stability_ARI_mean']:.4f}.\n"
        "These are audio groups, not verified mood labels.\n",encoding="utf8")
    html_table=comparison[["representation","k","dimensions","silhouette","db","ch","stability_ARI_mean","selected"]].to_html(index=False,float_format=lambda x:f"{x:.4f}")
    figure_names=["elbow_diagnostic","model_comparison","pca_variance","group_profiles","pca_group_projection"]
    figures="".join(f'<figure><img src="../output/figures/{name}.png" alt="{name.replace("_"," ")}"></figure>' for name in figure_names)
    page=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Nine K-Means Comparisons</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#203047}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:14px}}th,td{{padding:6px 10px;border-bottom:1px solid #ccd5de;text-align:right}}figure{{margin:24px 0}}</style></head><body>
<h1>Nine K-Means Comparisons</h1>
<p>Selected: {NAMES[selection['representation']]}, k={int(selection['k'])}. PCA80 uses six components; PCA90 uses seven.</p>
<p><a href="../README.md">README</a> | <a href="analysis_report.md">Analysis report</a> | <a href="../output/tables/model_comparison.csv">Full comparison CSV</a></p>
<div style="overflow-x:auto">{html_table}</div>{figures}</body></html>"""
    (reports/"index.html").write_text(page,encoding="utf8")
    print(f"English README and reports saved to {root}")

