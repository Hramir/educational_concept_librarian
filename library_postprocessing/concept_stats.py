from helpers import count_concepts, merge_concept_counts, func_or_nan
import json
import pandas as pd
import pickle
from statistics import mean
from tqdm import tqdm

# First run in pipeline (raw library after fix_jsons.py)
# concept_library_pickle_file = "concept_library_1702443584.pkl"
# csv_file = "video_transcripts_with_hierarchy_fixed_1702443584.csv"
# csv_file_out = "concept_library_with_stats_1702443584.csv"

# Second run in pipeline (after similarity-based mapping with BERT)
# concept_library_pickle_file = "concept_library_mapped_1702443584.pkl"
# csv_file = "video_transcripts_with_hierarchy_mapped_1702443584.csv"
# csv_file_out = "concept_library_mapped_with_stats_1702443584.csv"

# Third run in pipeline (after rarity-based mapping to null concept "<UNK>")
# concept_library_pickle_file = "concept_library_mapped_truncated_1702443584.pkl"
# csv_file = "video_transcripts_with_hierarchy_mapped_truncated_1702443584.csv"
# csv_file_out = "concept_library_mapped_truncated_with_stats_1702443584.csv"

# BERT rarity version
concept_library_pickle_file = "concept_library_mapped_bert_rarity_1702443584.pkl"
csv_file = "video_transcripts_with_hierarchy_mapped_bert_rarity_1702443584.csv"
csv_file_out = "concept_library_mapped_bert_rarity_with_stats_1702443584.csv"

# Load the list from the pickle file
if concept_library_pickle_file is not None:
  with open(concept_library_pickle_file, 'rb') as file:
    concept_library_string_list = list(pickle.load(file))
# elif concept_library_csv_file is not None: 
#   concept_library_string_list = pd.read_csv(concept_library_csv_file)["concept"].tolist()

vid_df = pd.read_csv(csv_file)

vid_df['primary_concepts'] = vid_df.apply(lambda row: {}, axis=1)
vid_df['supporting_concepts'] = vid_df.apply(lambda row: {}, axis=1)

grouped = vid_df.groupby('playlist_id')
playlist_vid_dfs = [group for _, group in grouped]

print(len(playlist_vid_dfs), "playlists")

total_concept_uses_count = 0

concept_library_dfs = []
for playlist_vid_df in tqdm(playlist_vid_dfs): 

  concept_library_dict = {item: {
    "concept": item,
    "primary_count": 0, 
    "supporting_count": 0, 
    "total_count": 0,
    "video_count": 0,
    "primary_playlist_positions": [], 
    "supporting_playlist_positions": [],
    "all_playlist_positions": []
  } for item in concept_library_string_list}

  for index, row in playlist_vid_df.iterrows():
    # Convert the activity_concept_hierarchy string to a dictionary

    if isinstance(row['activity_concept_hierarchy'], str):
      json_str = row['activity_concept_hierarchy']
      json_str = json_str.replace("{'", "{\"").replace(", '", ", \"").replace("['", "[\"").replace(": '", ": \"")
      json_str = json_str.replace("',", "\",").replace("']", "\"]").replace("':", "\":")
      try:
        concept_hierarchy = json.loads(json_str)

        # Run the count_concepts function
        primary_concepts, supporting_concepts = count_concepts(concept_hierarchy)

        total_concept_uses_count += sum(primary_concepts.values()) + sum(supporting_concepts.values())
        
      except Exception as e:
        print(row["video_id"])
        print(json_str)
        raise e
      
      video_concept_set = set()

      for concept in primary_concepts.keys():
        video_concept_set.add(concept)
        concept_library_dict[concept]["primary_count"] = concept_library_dict[concept]["primary_count"] + primary_concepts[concept]
        concept_library_dict[concept]["total_count"] = concept_library_dict[concept]["total_count"] + primary_concepts[concept]
        concept_library_dict[concept]["primary_playlist_positions"].extend([row["playlist_position"]]*primary_concepts[concept])
        concept_library_dict[concept]["all_playlist_positions"].extend([row["playlist_position"]]*primary_concepts[concept])

      for concept in supporting_concepts.keys():
        video_concept_set.add(concept)
        concept_library_dict[concept]["supporting_count"] = concept_library_dict[concept]["supporting_count"] + supporting_concepts[concept]
        concept_library_dict[concept]["total_count"] = concept_library_dict[concept]["total_count"] + supporting_concepts[concept]
        concept_library_dict[concept]["supporting_playlist_positions"].extend([row["playlist_position"]]*supporting_concepts[concept])
        concept_library_dict[concept]["all_playlist_positions"].extend([row["playlist_position"]]*supporting_concepts[concept])

      for concept in video_concept_set:
        concept_library_dict[concept]["video_count"] = concept_library_dict[concept]["video_count"] + 1

  concept_library_df = pd.DataFrame(concept_library_dict.values())

  # print(concept_library_df.head())

  concept_library_df["primary_playlist_position"] = 0.
  concept_library_df["supporting_playlist_position"] = 0.
  concept_library_df["all_playlist_position"] = 0.
  concept_library_df["primary_first_playlist_position"] = 0.
  concept_library_df["supporting_first_playlist_position"] = 0.
  concept_library_df["all_first_playlist_position"] = 0.
  for index, row in concept_library_df.iterrows():
    concept_library_df.at[index, "primary_playlist_position"] = func_or_nan(mean, row["primary_playlist_positions"])
    concept_library_df.at[index, "supporting_playlist_position"] = func_or_nan(mean, row["supporting_playlist_positions"])
    concept_library_df.at[index, "all_playlist_position"] = func_or_nan(mean, row["all_playlist_positions"])
    concept_library_df.at[index, "primary_first_playlist_position"] = func_or_nan(min, row["primary_playlist_positions"])
    concept_library_df.at[index, "supporting_first_playlist_position"] = func_or_nan(min, row["supporting_playlist_positions"])
    concept_library_df.at[index, "all_first_playlist_position"] = func_or_nan(min, row["all_playlist_positions"])

  columns_to_drop = ["primary_playlist_positions", "supporting_playlist_positions", "all_playlist_positions"]
  concept_library_df = concept_library_df.drop(columns=columns_to_drop)

  # print(concept_library_df.head())

  concept_library_dfs.append(concept_library_df)


aggregation_rules = {
    'primary_count': 'sum',
    'supporting_count': 'sum',
    'total_count': 'sum',
    'video_count': 'sum',
    'primary_playlist_position': 'mean',
    'supporting_playlist_position': 'mean',
    'all_playlist_position': 'mean',
    'primary_first_playlist_position': 'mean',
    'supporting_first_playlist_position': 'mean',
    'all_first_playlist_position': 'mean'
}
concept_library_df = pd.concat(concept_library_dfs).groupby('concept', as_index=False).agg(aggregation_rules)

concept_library_df = concept_library_df.sort_values(by="video_count", ascending=False)

print(concept_library_df.head())

concept_library_df.to_csv(csv_file_out, index=False)

print(total_concept_uses_count)
