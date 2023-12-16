import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine


video_csv_file = "video_transcripts_with_hierarchy_mapped_truncated_1702443584.csv"
concepts_csv_file = "concept_library_mapped_truncated_with_stats_1702443584.csv"

concept_df = pd.read_csv(concepts_csv_file)
vid_df = pd.read_csv(video_csv_file)

total_concept_count = concept_df["total_count"].sum()
concept_df["frequency"] = concept_df["total_count"] / total_concept_count

temp_df = concept_df[["frequency", "all_first_playlist_position"]].dropna()
spearman_corr, spearman_p_value = spearmanr(temp_df["frequency"], temp_df["all_first_playlist_position"])
print(f"Spearman correlation between frequency and advancedness: {spearman_corr}, p-value: {spearman_p_value}")
