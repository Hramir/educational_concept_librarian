from helpers import standardize
import pandas as pd
import json

forgot_json_dumps = False
standardize_concept_names = True
csv_file = "video_transcripts_with_hierarchy_1702443584.csv"
csv_file_out = "video_transcripts_with_hierarchy_fixed_1702443584.csv"


def update_concepts_in_activity(activity):
  # Standardize the primary concept
  activity["primary_concept"] = standardize(activity["primary_concept"])

  # Standardize each of the supporting concepts
  activity["supporting_concepts"] = [standardize(concept) for concept in activity["supporting_concepts"]]

  # Recursively update nested activities
  for nested_activity in activity.get("activities", []):
    update_concepts_in_activity(nested_activity)

def update_concepts_in_json(json_data):
  for activity in json_data["lesson"]:
    update_concepts_in_activity(activity)
  return json_data


vid_df = pd.read_csv(csv_file)

if forgot_json_dumps:
  for index, row in vid_df.iterrows():
    json_str = row['activity_concept_hierarchy']
    json_str = json_str.replace("{'", "{\"").replace(", '", ", \"").replace("['", "[\"").replace(": '", ": \"")
    json_str = json_str.replace("',", "\",").replace("']", "\"]").replace("':", "\":")

    concept_hierarchy = json.loads(json_str)

    vid_df.at[index, "activity_concept_hierarchy"] = json.dumps(concept_hierarchy)

if standardize_concept_names:
  for index, row in vid_df.iterrows():
    try: 
      concept_hierarchy = json.loads(row['activity_concept_hierarchy'])
      concept_hierarchy = update_concepts_in_json(concept_hierarchy)
      vid_df.at[index, "activity_concept_hierarchy"] = json.dumps(concept_hierarchy)
    except Exception as e:
      print(json.dumps(concept_hierarchy, indent=2))
      raise e
      print("=============================")


vid_df.to_csv(csv_file_out, index=False)
  

