from helpers import count_concepts, merge_concept_counts, func_or_nan
import json
import pandas as pd
import pickle
from statistics import mean
import math


concept_library_pickle_file = "concept_library_1701550479.pkl"
csv_file = "video_transcripts_with_hierarchy_1701550479_ordered.csv"

# Load the list from the pickle file
with open(concept_library_pickle_file, 'rb') as file:
    concept_library_string_list = pickle.load(file)

# Convert the list to a dictionary with each item as key and 0 as value
# concept_library_dict = {item: {
#   "concept": item,
#   "primary_count": 0, 
#   "supporting_count": 0, 
#   "total_count": 0,
#   "primary_playlist_positions": [], 
#   "supporting_playlist_positions": [],
#   "all_playlist_positions": []
# } for item in concept_library_string_list}


vid_df = pd.read_csv(csv_file)

vid_df['primary_concepts'] = vid_df.apply(lambda row: {}, axis=1)
vid_df['supporting_concepts'] = vid_df.apply(lambda row: {}, axis=1)

grouped = vid_df.groupby('playlist_id')
playlist_vid_dfs = [group for _, group in grouped]

print(len(playlist_vid_dfs))

concept_library_dfs = []
for playlist_vid_df in playlist_vid_dfs: 

  concept_library_dict = {item: {
    "concept": item,
    "primary_count": 0, 
    "supporting_count": 0, 
    "total_count": 0,
    "primary_playlist_positions": [], 
    "supporting_playlist_positions": [],
    "all_playlist_positions": []
  } for item in concept_library_string_list}

  for index, row in playlist_vid_df.iterrows():
    # Convert the activity_concept_hierarchy string to a dictionary

    if isinstance(row['activity_concept_hierarchy'], str):
      concept_hierarchy = json.loads(row['activity_concept_hierarchy'].replace("'", "\""))

      # Run the count_concepts function
      primary_concepts, supporting_concepts = count_concepts(concept_hierarchy)

      for concept in primary_concepts.keys():
        concept_library_dict[concept]["primary_count"] = concept_library_dict[concept]["primary_count"] + primary_concepts[concept]
        concept_library_dict[concept]["total_count"] = concept_library_dict[concept]["total_count"] + primary_concepts[concept]
        concept_library_dict[concept]["primary_playlist_positions"].extend([row["playlist_position"]]*primary_concepts[concept])
        concept_library_dict[concept]["all_playlist_positions"].extend([row["playlist_position"]]*primary_concepts[concept])

      for concept in supporting_concepts.keys():
        concept_library_dict[concept]["supporting_count"] = concept_library_dict[concept]["supporting_count"] + supporting_concepts[concept]
        concept_library_dict[concept]["total_count"] = concept_library_dict[concept]["total_count"] + supporting_concepts[concept]
        concept_library_dict[concept]["supporting_playlist_positions"].extend([row["playlist_position"]]*supporting_concepts[concept])
        concept_library_dict[concept]["all_playlist_positions"].extend([row["playlist_position"]]*supporting_concepts[concept])

  concept_library_df = pd.DataFrame(concept_library_dict.values())

  # print(concept_library_df.head())

  concept_library_df["primary_playlist_position"] = 0
  concept_library_df["supporting_playlist_position"] = 0
  concept_library_df["all_playlist_position"] = 0
  concept_library_df["primary_first_playlist_position"] = 0
  concept_library_df["supporting_first_playlist_position"] = 0
  concept_library_df["all_first_playlist_position"] = 0
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
    'primary_playlist_position': 'mean',
    'supporting_playlist_position': 'mean',
    'all_playlist_position': 'mean',
    'primary_first_playlist_position': 'mean',
    'supporting_first_playlist_position': 'mean',
    'all_first_playlist_position': 'mean'
}
concept_library_df = pd.concat(concept_library_dfs).groupby('concept', as_index=False).agg(aggregation_rules)

print(concept_library_df.head())

concept_library_df.to_csv("concept_library_1701550479.csv", index=False)




#     # Save the outputs to new columns in the DataFrame
#     vid_df.at[index, 'primary_concepts'] = primary_concepts
#     vid_df.at[index, 'supporting_concepts'] = supporting_concepts

# vid_df.to_csv("video_transcripts_with_hierarchy_1701550479_ordered_counted.csv", index=False)






# Load the list from the pickle file
# with open(concept_library_pickle_file, 'rb') as file:
#     concept_library_string_list = pickle.load(file)

# # Convert the list to a dictionary with each item as key and 0 as value
# concept_library_dict = {item: {"count": 0, "positions": []} for item in concept_library_string_list}

# concept_library = pd.DataFrame(concept_library_string_list, columns=['concept_name'])

# concept_library["primary_count"] = 0
# concept_library["supporting_count"] = 0
# concept_library["total_count"] = 0 # TODO set this to sum of primary and supporting at the end
# concept_library["playlist_positions"] = vid_df.apply(lambda row: [], axis=1)
