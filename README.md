# Project Overview
## Objective
To take a dataset of songs and their features and create a model that will cluster them into groups based on mood and activities instead of genre

## Scope


# Methodology
## Data Collection
The data utilized was the spotify dataset from tidytuesday linked [here](https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-01-21/readme.md). Please refer to the readme for the data collection process and explanation of the features.

Additionally, we queried the LRCLIB and Genius API for song lyrics.[^1]
[^1]: refer to week 3 code for the methodology on obtaining song lyrics

## Data Preparation
### Data Cleaning
- **Removing Duplicates by Unique ID:**
`track_id` uniquely identifies a song, thus any rows with a duplicate `track_id` were removed.
- **Composite ID Duplicate Removal:** Identified songs sharing the same name, artist, and audio features but having different `track_id` values. To prevent duplicate entries from introducing a downward bias in popularity, the record with the lower popularity value was removed.
- **Composite ID Duplicate Variations:** Some songs share the same name and artist, but have different audio features. We calculated thresholds for each audio feature and used those to determine whether songs were similar enough to be classified as the same song.[^2]
[^2]:refer to readme in week 4 for a detailed explanation on how we decided what to drop
- **Missing Values:[^3]** There existed 5 rows of data with missing track names and artist names. However, they all contained information about the audio features so they were retained after removing any duplicates.
[^3]: refer to week 4 readme

### Data Transformation
- **Data Standardization:** Numerical variables were standardized in order to perform principal component analysis. 
- **Data Normalization:** Four audio features are not normalized. These four are loudness, key, tempo, and duration. Key is a categorical variable so it will not be subject to normalization. Loudness is logarithmic so it will only be normalized during modeling.

### Data Annotation
- **Playlist Name Label Generation:** The `playlist_name` associated with each `track_id` is created by a user. We filtered through these to find playlist names that had clearly associated activities and used these to generate labels for songs for supervised learning.
- **Sentiment Analysis Label Generation:** We used sentiment analysis on the song lyrics we queried to generate an alternative set of labels for supervised learning.

## Exploratory Data Analysis
- **Principal Component Analysis (PCA):** We performed PCA after standardizing the numerical variables. PCA with 7 principal components retains roughly 91% of the training variance.

## Modeling
- **Data Splitting:** We divided the data randomly into three sets with 60% of the data being used for training, 20% for selecting a model, and the last 20% for evaluation as a test set.
- **Initial Model Selection:** We compared the results from four models:  K-means, GMM with full covariance, GMM with diagonal covariance, and BIRCH.
- **Unsupervised Learning:** We began by using K-means clustering in order to create our initial clusters. This approach yielded six distinct groups 
## Evaluation

# Weekly Progress

# Results
