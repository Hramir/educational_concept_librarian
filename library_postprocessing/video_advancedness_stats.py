import pandas as pd
import numpy as np
import json
from helpers import count_concepts, merge_concept_counts, func_or_nan
from statistics import mean


video_csv_file = "video_transcripts_with_hierarchy_1701550479_ordered.csv"
concepts_csv_file = "concept_library_1701550479.csv"

vid_df = pd.read_csv(video_csv_file)
concept_df = pd.read_csv(concepts_csv_file)

total_concept_count = concept_df["total_count"].sum()

concept_df["frequency"] = concept_df["total_count"] / total_concept_count

vid_df["mean_primary_concept_frequency"] = np.nan
vid_df["mean_supporting_concept_frequency"] = np.nan
for index, row in vid_df.iterrows():
  if isinstance(row['activity_concept_hierarchy'], str):
    concept_hierarchy = json.loads(row['activity_concept_hierarchy'].replace("'", "\""))

    # Run the count_concepts function
    primary_concepts, supporting_concepts = count_concepts(concept_hierarchy)

    freqs = []
    advs = []
    for concept in primary_concepts.keys():
      try:
        freqs.extend([concept_df[concept_df["concept"] == concept]["frequency"].iloc[0]]*primary_concepts[concept])
        advs.extend([concept_df[concept_df["concept"] == concept]["primary_first_playlist_position"].iloc[0]]*primary_concepts[concept])
      except: 
        print("Warning: primary concept error for: " + concept)
    vid_df.at[index, "mean_primary_concept_frequency"] = mean(freqs)
    vid_df.at[index, "mean_primary_concept_advancedness"] = mean(advs)

    freqs = []
    advs = []
    for concept in supporting_concepts.keys():
      try:
        freqs.extend([concept_df[concept_df["concept"] == concept]["frequency"].iloc[0]]*supporting_concepts[concept])
        advs.extend([concept_df[concept_df["concept"] == concept]["supporting_first_playlist_position"].iloc[0]]*supporting_concepts[concept])
      except: 
        print("Warning: supporting concept error for: " + concept)
    vid_df.at[index, "mean_supporting_concept_frequency"] = mean(freqs)
    vid_df.at[index, "mean_supporting_concept_advancedness"] = mean(advs)

vid_df.to_csv("vids_with_freqs.csv", index=False)

