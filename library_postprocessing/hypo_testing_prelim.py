import pandas as pd
from scipy.stats import pearsonr, spearmanr


video_csv_file = "vids_with_freqs.csv"

vid_df = pd.read_csv(video_csv_file)

vid_df["concept_frequency_ratio"] = vid_df["mean_supporting_concept_frequency"] / vid_df["mean_primary_concept_frequency"]
vid_df["concept_advancedness_ratio"] = vid_df["mean_supporting_concept_advancedness"] / vid_df["mean_primary_concept_advancedness"]


average_view_count_per_playlist = vid_df.groupby('playlist_id')['view_count'].mean()
vid_df = vid_df.merge(average_view_count_per_playlist, on='playlist_id', suffixes=('', '_average'))
vid_df['normalized_view_count'] = vid_df['view_count'] / vid_df['view_count_average']
vid_df.drop('view_count_average', axis=1, inplace=True)

# Columns to test against score value (e.g. view count, comment sentiment)
columns_to_test = [
  "mean_supporting_concept_frequency", 
  "mean_primary_concept_frequency", 
  "concept_frequency_ratio",
  "mean_supporting_concept_advancedness", 
  "mean_primary_concept_advancedness", 
  "concept_advancedness_ratio",
]

for score in ["view_count", "normalized_view_count"]:
  print("SCORE METRIC: " + score)

  for column in columns_to_test:
      
      temp_df = vid_df[[score, column]].dropna()

      # Pearson correlation
      pearson_corr, pearson_p_value = pearsonr(temp_df[score], temp_df[column])
      print(f"Pearson correlation between {score} and {column}: {pearson_corr}, p-value: {pearson_p_value}")

      # Spearman correlation
      spearman_corr, spearman_p_value = spearmanr(temp_df[score], temp_df[column])
      print(f"Spearman correlation between {score} and {column}: {spearman_corr}, p-value: {spearman_p_value}")