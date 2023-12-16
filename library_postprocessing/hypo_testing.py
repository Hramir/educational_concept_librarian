import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine


# BERT similarity-remapped version
# video_csv_file = "video_transcripts_with_hierarchy_mapped_with_freqs_1702443584.csv"

# BERT rarity-remapped version
# video_csv_file = "video_transcripts_with_hierarchy_mapped_bert_rarity_with_freqs_1702443584.csv"

# <UNK> truncated version
# video_csv_file = "video_transcripts_with_hierarchy_mapped_truncated_with_freqs_1702443584.csv"
video_csv_file = "video_transcripts_with_hierarchy_mapped_truncated_conceptual_lda_1702443584.csv"

conceptual_lda = True
num_topics = 10

vid_df = pd.read_csv(video_csv_file)

if conceptual_lda: 

  for index, row in vid_df.iterrows():
    primary_concept_topic_weights = row[["primary_concept_topic_" + str(t) for t in range(num_topics)]].to_numpy()
    supporting_concept_topic_weights = row[["supporting_concept_topic_" + str(t) for t in range(num_topics)]].to_numpy()

    conceptual_support_similarity = 1 - cosine(primary_concept_topic_weights, supporting_concept_topic_weights)
    vid_df.at[index, "conceptual_support_similarity"] = conceptual_support_similarity

# Calculate normalized metrics
for metric in ["view_count", "like_to_view_ratio", "average_sentiment"]:
  average_metric_per_playlist = vid_df.groupby("playlist_id")[metric].mean()
  vid_df = vid_df.merge(average_metric_per_playlist, on="playlist_id", suffixes=("", "_average"))
  vid_df["normalized_" + metric] = vid_df[metric] / vid_df[metric + "_average"]
  vid_df.drop(metric + "_average", axis=1, inplace=True)

# Columns to test against score value (e.g. view count, comment sentiment)
columns_to_test = [
  "view_count",
  "like_count",
  "like_to_view_ratio",
  "average_sentiment",
  "playlist_position",
  "mean_supporting_concept_frequency", 
  "mean_primary_concept_frequency", 
  "simple_concept_frequency_ratio",
  "mean_supporting_concept_advancedness", 
  "mean_primary_concept_advancedness", 
  "simple_concept_advancedness_ratio",
  "concept_advancedness_ratio",
]

if conceptual_lda:
  columns_to_test.append("conceptual_support_similarity")

# for score in ["view_count", "normalized_view_count", "like_to_view_ratio", "normalized_like_to_view_ratio", "average_sentiment", "normalized_average_sentiment"]:
for score in ["view_count", "like_to_view_ratio", "average_sentiment"]:
  print("SCORE METRIC: " + score)

  for column in columns_to_test:
      
    if not column == score:
      temp_df = vid_df[[score, column]].dropna()

      # Pearson correlation (Only applies if vars are normally distributed)
      # pearson_corr, pearson_p_value = pearsonr(temp_df[score], temp_df[column])
      # print(f"Pearson correlation between {score} and {column}: {pearson_corr}, p-value: {pearson_p_value}")

      # Spearman correlation
      spearman_corr, spearman_p_value = spearmanr(temp_df[score], temp_df[column])
      print(f"Spearman correlation between {score} and {column}: {spearman_corr}, p-value: {spearman_p_value}")