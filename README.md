# Project Overview
## Objective
To take a dataset of songs and their features and create a model that will cluster them into groups based on mood and activities instead of genre

## Scope


# Methodology
## Data Collection
The data utilized was the spotify dataset from tidytuesday linked [here](https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-01-21/readme.md). Please refer to the readme for the data collection process and explanation of the features.

## Data Preparation
### Data Cleaning
- **Removing Duplicates by Unique ID:**
`track_id` uniquely identifies a song, thus any rows with a duplicate `track_id` were removed.
- **Composite ID Duplicate Removal:** Identified songs sharing the same name, artist, and audio features but having different `track_id` values. To prevent duplicate entries from introducing a downward bias in popularity, the record with the lower popularity value was removed.
- 

## Exploratory Data Analysis

## Modeling

## Evaluation

# Weekly Progress

# Results