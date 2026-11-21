# Week 5 - Labeling and Supervised Learning
## Overview
Our objective this week was to 
- obtain lyrics from the list of failures from week 4
- apply sentiment analysis on the song lyrics to generate mood labels
- perform supervised learning with the generated mood labels
- draft a report outline 

## Data
### Newly Generated Datasets
- `geniusLyrics.jsonl` - Contains the `track_id`, `track_name`, `track_artist`, and lyrics of songs from `failed_lyrics.jsonl` that were successfully queried from the Genius API.
- `failedGenius.jsonl` - Contains the `track_id`, `track_name`, `track_artist`, error type, and error message for songs that could not be retrieved through the Genius API.

### Modifications to Existing Datasets
- `spotify_songs.csv` - added mood labels from sentiment analysis to prepare for supervised learning.

### Discarded Datasets
- `notWordy.csv` - songs from `failed_lyrics.jsonl` that had `speechiness` below the .33 threshold. Discarded as songs at .04 in `speechiness` had lyrics, indicating that the threshold was not suitable for identifying songs with lyrics.
- `extraSongs.csv`- songs that had `speechiness` above the .33 threshold. Discarded because lyrics existing for songs with speechiness values below the threshold indicated that speechiness alone was insufficient for determining whether a song contained lyrics.

## Methodology
- 
## Results

## Limitations and Obstacles

## Deliverables

## Next Steps