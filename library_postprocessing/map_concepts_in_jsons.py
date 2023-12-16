import pandas as pd
import json
import pickle

# First run in pipeline (similarity-based mappings)
# csv_file = "video_transcripts_with_hierarchy_fixed_1702443584.csv"
# concept_mapping_dict_file = "concept_similarity_mappings_87_1702443584.pkl"
# csv_file_out = "video_transcripts_with_hierarchy_mapped_1702443584.csv"

# Second run in pipeline (rarity-based mappings to null concept "<UNK>")
# csv_file = "video_transcripts_with_hierarchy_mapped_1702443584.csv"
# concept_mapping_dict_file = "concept_rarity_mappings_1702443584.pkl"
# csv_file_out = "video_transcripts_with_hierarchy_mapped_truncated_1702443584.csv"

# BERT rarity version
csv_file = "video_transcripts_with_hierarchy_mapped_1702443584.csv"
concept_mapping_dict_file = "concept_bert_rarity_mappings_1702443584.pkl"
csv_file_out = "video_transcripts_with_hierarchy_mapped_bert_rarity_1702443584.csv"


def map_concept(concept, mapping_dict):
  if concept in mapping_dict:
    return map_concept(mapping_dict[concept], mapping_dict)
  else:
    return concept

def map_concepts_in_activity(activity, mapping_dict):
  # Map the primary concept
  activity["primary_concept"] = map_concept(activity["primary_concept"], mapping_dict)

  # Map each of the supporting concepts
  activity["supporting_concepts"] = [map_concept(concept, mapping_dict) for concept in activity["supporting_concepts"]]

  # Recursively update nested activities
  for nested_activity in activity.get("activities", []):
    map_concepts_in_activity(nested_activity, mapping_dict)

def map_concepts_in_json(json_data, mapping_dict):
  for activity in json_data["lesson"]:
    map_concepts_in_activity(activity, mapping_dict)
  return json_data


vid_df = pd.read_csv(csv_file)

with open(concept_mapping_dict_file, 'rb') as file:
  mapping_dict = pickle.load(file)

for index, row in vid_df.iterrows():
  concept_hierarchy = json.loads(row['activity_concept_hierarchy'])
  concept_hierarchy = map_concepts_in_json(concept_hierarchy, mapping_dict)
  vid_df.at[index, "activity_concept_hierarchy"] = json.dumps(concept_hierarchy)


vid_df.to_csv(csv_file_out, index=False)
  

