# Verification record: tracks_clean.csv input

## Input audit

- The supplied tracks_clean.csv contains 28,356 rows and 19 columns.
- Track IDs are nonmissing and unique; there are no fully duplicate rows.
- All nine audio feature columns are numeric and finite.
- No audio feature is missing in the supplied file.
- One track has tempo=0; this field is treated as unavailable and uses the
  fitted median while its row is retained.
- Four tracks have missing track name, artist, and album name metadata.
- The input is already deduplicated and remains in its supplied order.
- It has no playlist columns and is not standardized.
- The bundled input file matches the supplied file byte-for-byte.
  Its SHA256 is recorded in output/summary.json.

## Implementation changes

The default source in run_all.py and scripts/01_fit_clusters.py is now
data/raw/tracks_clean.csv. The data/raw directory stores the immutable input
as supplied, even though it is already cleaned.

The input-stage drop_duplicates operation and duplicate-feature conflict
aggregation were removed. Input contract checks raise errors for missing
required columns, missing or repeated IDs, and malformed/nonfinite audio
values instead of silently changing the supplied table.

Playlist-membership extraction was removed because its source fields are not
present. The obsolete original CSV and membership output are not included.

Median treatment of nonpositive tempo and StandardScaler are retained because
they are still needed for the supplied numeric measurements. Split proportions,
seed, nine feature columns, K-means settings, elbow diagnostic, stability
protocol, final fitting, and prediction logic are otherwise unchanged.

README and generated reports now describe the clean-input workflow and no
longer claim that this run removed records or reconstructed playlists.

## Executed checks

- Compiled all Python modules and numbered scripts.
- Ran python run_all.py --include-demo successfully.
- Regenerated the fitted model, metrics, profiles, example songs, and four figures.
- Regenerated the English analysis report, TXT summary, and HTML results page.
- Ran the prediction entry point from outside the repository working directory.
- Verified all 28,356 supplied IDs appear once in the delivery assignments.
- Verified validation, test, and stability values exactly match the attached
  six-group baseline.
- Verified splits, all assignments, profiles, elbow data, representative songs,
  boundary songs, and example predictions match the attached baseline on the
  shared columns. The assignment file omits the four absent playlist columns.
- Verified the six demonstration inputs agree with tracks_clean.csv on every
  audio feature.
- Visually checked the regenerated elbow plot.
- Checked local README, Markdown, HTML, and image links.
- Checked ZIP integrity and excluded temporary files and Python caches.

## Result

The supplied tracks_clean.csv is identical to the earlier track-ID-deduplicated
table on its shared columns and row order. Therefore replacing the source and
removing redundant deduplication preserves the six-group results.

The package retains one formal specification: original standardized nine
features and K-means with k=6, n_init=20, seed 42. Other k values are used only
for the training elbow diagnostic, not model selection or alternative delivery
models. No new independent test data or human-confirmed mood labels are claimed.

