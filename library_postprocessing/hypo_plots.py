import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# BERT similarity-remapped version
# video_csv_file = "video_transcripts_with_hierarchy_mapped_1702443584.csv"
# concepts_csv_file = "concept_library_mapped_with_stats_1702443584.csv"

# BERT rarity-remapped version
# video_csv_file = "video_transcripts_with_hierarchy_mapped_bert_rarity_1702443584.csv"
# concepts_csv_file = "concept_library_mapped_bert_rarity_with_stats_1702443584.csv"

# <UNK> truncated version
video_csv_file = "video_transcripts_with_hierarchy_mapped_truncated_with_freqs_1702443584.csv"
concepts_csv_file = "concept_library_mapped_truncated_with_stats_1702443584.csv"

vid_df = pd.read_csv(video_csv_file)

concept_df = pd.read_csv(concepts_csv_file)

plt.figure()
plt.hist(vid_df["concept_advancedness_ratio"])
plt.title("Concept advancedness ratio")
plt.savefig("concept_advancedness_ratio.png")

plt.figure()
plt.hist(vid_df["average_sentiment"])
plt.title("concept_advancedness_ratio")
plt.savefig("average_sentiment.png")