import pandas as pd
import numpy as np
import json
from helpers import count_concepts
from statistics import mean

# BERT similarity-remapped version
# video_csv_file = "video_transcripts_with_hierarchy_mapped_1702443584.csv"
# concepts_csv_file = "concept_library_mapped_with_stats_1702443584.csv"
# vids_with_freqs_csv_file = "video_transcripts_with_hierarchy_mapped_with_freqs_1702443584.csv"

# BERT rarity-remapped version
# video_csv_file = "video_transcripts_with_hierarchy_mapped_bert_rarity_1702443584.csv"
# concepts_csv_file = "concept_library_mapped_bert_rarity_with_stats_1702443584.csv"
# vids_with_freqs_csv_file = "video_transcripts_with_hierarchy_mapped_bert_rarity_with_freqs_1702443584.csv"

# <UNK> truncated version
video_csv_file = "video_transcripts_with_hierarchy_mapped_truncated_1702443584.csv"
concepts_csv_file = "concept_library_mapped_truncated_with_stats_1702443584.csv"
vids_with_freqs_csv_file = "video_transcripts_with_hierarchy_mapped_truncated_with_freqs_1702443584.csv"

def get_concept_advancedness_ratio(json_data, concept_advancedness_pri, concept_advancedness_supp, ratios=None):
  """
  Function to calculate the concept advancedness ratio

  :param json_data: List of nested JSON objects.
  :return: mean concept advancedness ratio
  """

  if ratios is None:
    ratios = []
  for item in json_data["lesson"]:
    if not item["primary_concept"] == "<UNK>":
      pri_adv = concept_advancedness_pri[item["primary_concept"]]
      supp_advs = []
      for supp_concept in item["supporting_concepts"]:
        if not supp_concept == "<UNK>":
          supp_advs.append(concept_advancedness_supp[supp_concept])
      if len(supp_advs) > 0:
        ratios.append(mean(supp_advs) / pri_adv)
      
    # If there are nested activities, recurse into them
    if "activities" in item and len(item["activities"]) > 0:
      _ = get_concept_advancedness_ratio({"lesson": item["activities"]}, concept_advancedness_pri, concept_advancedness_supp, ratios)
  
  if len(ratios) > 0:
    return mean(ratios)
  else:
    return np.nan


vid_df = pd.read_csv(video_csv_file)
concept_df = pd.read_csv(concepts_csv_file)

total_concept_count = concept_df["total_count"].sum()

concept_df["frequency"] = concept_df["total_count"] / total_concept_count

vid_df["mean_primary_concept_frequency"] = np.nan
vid_df["mean_supporting_concept_frequency"] = np.nan
for index, row in vid_df.iterrows():
  if isinstance(row['activity_concept_hierarchy'], str):
    concept_hierarchy = json.loads(row['activity_concept_hierarchy'])

    # Run the count_concepts function
    primary_concepts, supporting_concepts = count_concepts(concept_hierarchy)

    freqs = []
    advs = []
    for concept in primary_concepts.keys():
      if not concept == "<UNK>":
        try:
          freqs.extend([concept_df[concept_df["concept"] == concept]["frequency"].iloc[0]]*primary_concepts[concept])
          # advs.extend([concept_df[concept_df["concept"] == concept]["primary_first_playlist_position"].iloc[0]]*primary_concepts[concept])
          advs.extend([concept_df[concept_df["concept"] == concept]["all_first_playlist_position"].iloc[0]]*primary_concepts[concept])
        except: 
          print("Warning: primary concept error for: " + concept)
    if len(freqs) > 0:
      vid_df.at[index, "mean_primary_concept_frequency"] = mean(freqs)
      vid_df.at[index, "mean_primary_concept_advancedness"] = mean(advs)
    else:
      vid_df.at[index, "mean_primary_concept_frequency"] = np.nan
      vid_df.at[index, "mean_primary_concept_advancedness"] = np.nan

    freqs = []
    advs = []
    for concept in supporting_concepts.keys():
      if not concept == "<UNK>":
        try:
          freqs.extend([concept_df[concept_df["concept"] == concept]["frequency"].iloc[0]]*supporting_concepts[concept])
          # advs.extend([concept_df[concept_df["concept"] == concept]["supporting_first_playlist_position"].iloc[0]]*supporting_concepts[concept])
          advs.extend([concept_df[concept_df["concept"] == concept]["all_first_playlist_position"].iloc[0]]*supporting_concepts[concept])
        except: 
          print("Warning: supporting concept error for: " + concept)
    if len(freqs) > 0:
      vid_df.at[index, "mean_supporting_concept_frequency"] = mean(freqs)
      vid_df.at[index, "mean_supporting_concept_advancedness"] = mean(advs)
    else:
      vid_df.at[index, "mean_supporting_concept_frequency"] = np.nan
      vid_df.at[index, "mean_supporting_concept_advancedness"] = np.nan

vid_df["simple_concept_frequency_ratio"] = vid_df["mean_supporting_concept_frequency"] / vid_df["mean_primary_concept_frequency"]
vid_df["simple_concept_advancedness_ratio"] = vid_df["mean_supporting_concept_advancedness"] / vid_df["mean_primary_concept_advancedness"]


# Get complex version of  concept advancedness ratio
concept_advancedness_primary = dict(zip(concept_df['concept'], concept_df['primary_first_playlist_position']))
concept_advancedness_supporting = dict(zip(concept_df['concept'], concept_df['supporting_first_playlist_position']))

for index, row in vid_df.iterrows():
  if isinstance(row['activity_concept_hierarchy'], str):
    concept_hierarchy = json.loads(row['activity_concept_hierarchy'])

    concept_advancedness_ratio = get_concept_advancedness_ratio(concept_hierarchy, concept_advancedness_primary, concept_advancedness_supporting)

    vid_df.at[index, "concept_advancedness_ratio"] = concept_advancedness_ratio

vid_df.to_csv(vids_with_freqs_csv_file, index=False)

