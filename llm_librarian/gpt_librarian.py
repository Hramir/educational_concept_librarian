import pandas as pd
import openai
import pickle
import sys
import json
import time
from logger import Logger


def print_dialog(prompt_1, response_1, prompt_2, response_2, video_json, concept_library):
  print("\n\n========== Prompt 1 ==================")
  print(prompt_1_filled)

  print("\n\n=== Prompt 2 (includes response 1) ===")
  print(response_1_text + "\n\n" + prompt_2_filled)

  print("\n\n========== Response 2 ================")
  print(response_2_text)

  print("\n\n========== Video JSON ================")
  print(video_json)

  print("\n\n========== Concept library ===========")
  print(concept_library)


def expand_concepts(json_data, concepts_set):
    """
    Function to expand the concepts set with unique concepts found in a nested JSON structure.

    :param json_data: List of nested JSON objects.
    :param concepts_set: Set of existing concepts.
    :return: Expanded set of concepts.
    """
    for item in json_data["lesson"]:
        # Add concepts from the current level to the set
        concepts_set.add(item["primary_concept"])
        for concept in item["supporting_concepts"]:
            concepts_set.add(concept)
        
        # If there are nested activities, recurse into them
        if "activities" in item and item["activities"]:
            expand_concepts({"lesson": item["activities"]}, concepts_set)
    
    return concepts_set


def check_and_repair_json(json_obj):
  valid_activities = ["definition", "example", "visualization", "application", "analogy", "additional resources"]
  anomalies = []

  def create_empty_activity(original_activity_name="", upper_primary_concept=""):
    return {
      "activity": original_activity_name if isinstance(original_activity_name, str) else "",
      "primary_concept": upper_primary_concept if isinstance(upper_primary_concept, str) else "",
      "supporting_concepts": [],
      "activities": []
    }

  def check_activity(activity, path):
    # Reinitialize fields if they are not strings or lists as required
    if not isinstance(activity.get("activity", None), str):
      activity["activity"] = ""
      anomalies.append(f"Non-string 'activity' at {path}, reinitialized to empty string")

    if not isinstance(activity.get("primary_concept", None), str):
      activity["primary_concept"] = ""
      anomalies.append(f"Non-string 'primary_concept' at {path}, reinitialized to empty string")

    if not isinstance(activity.get("supporting_concepts", None), list):
      activity["supporting_concepts"] = []
      anomalies.append(f"Non-list 'supporting_concepts' at {path}, reinitialized to empty list")

    # Check and fix 'activities' field
    if "activities" not in activity or not isinstance(activity["activities"], list):
      activity["activities"] = []
      anomalies.append(f"Missing or invalid 'activities' at {path}, initialized to empty list")
    else:
      for i, nested_activity in enumerate(activity["activities"]):
        if isinstance(nested_activity, str):
          # Replace string with an empty activity object
          activity["activities"][i] = create_empty_activity(nested_activity, activity["primary_concept"])
          anomalies.append(f"String 'activities' item replaced with empty activity at {path}[{i}]")
        elif not isinstance(nested_activity, dict):
          anomalies.append(f"Invalid 'activities' item at {path}[{i}], replaced with empty activity")
          activity["activities"][i] = create_empty_activity()
        else:
          check_activity(nested_activity, f"{path}[{i}]")

  # Clone the object to avoid modifying the original
  json_obj_clone = json.loads(json.dumps(json_obj))

  # Check the structure
  for i, lesson in enumerate(json_obj_clone["lesson"]):
    check_activity(lesson, f"lesson[{i}]")

  return json_obj_clone, anomalies


verbose = True
debugging = True
logging = True
model_id = "gpt-4-1106-preview"
# model_id = "gpt-3.5-turbo-1106"
randomly_sample_subset = 5

# Load the video transcripts
df = pd.read_csv("video_transcripts.csv")

if randomly_sample_subset and randomly_sample_subset is not None: 
  df = df.sample(n=randomly_sample_subset, replace=False, random_state=randomly_sample_subset)

# New column for hierarchy
df["activity_concept_hierarchy"] = ""

# Initialize the concept library set
concept_library = set()

# Load prompt templates
with open("prompt_1.txt", "r") as file:
    prompt_1 = file.read()
with open("prompt_2.txt", "r") as file:
    prompt_2 = file.read()

start_timestamp = time.time()
if logging:
   sys.stdout = Logger("gpt_librarian_" + str(int(start_timestamp)) + ".log")

client = openai.OpenAI()

major_error_count = 0
json_structure_error_count = 0
prompt_token_count = 0
completion_token_count = 0

# Process each transcript in the dataframe
for index, row in df.iterrows():
    
  try:
    response_1_text = ""
    response_2_text = None
    prompt_1_filled = None
    prompt_2_filled = None
    video_json = None
    print("====================================================")
    print("============= Video ID: " + str(row["video_id"]) + " ================")

    # Replace <transcript> in prompt_1 with the actual transcript
    transcript = row["transcript"]
    prompt_1_filled = prompt_1.replace("<transcript>", transcript)

    # Send prompt_1 to OpenAI and get the response
    response_1 = client.chat.completions.create(
        model=model_id,  # Adjust the model as necessary
        messages=[
          {"role": "system", "content": "You are are an expert data analyst. You are part of a research team studying the role of concept hierarchies in determining the teaching quality of educational videos."},
          {"role": "user", "content": prompt_1_filled},
        ]
        # max_tokens=500  # Adjust max tokens as necessary
    )

    # Count tokens
    prompt_token_count = prompt_token_count + response_1.usage.prompt_tokens
    completion_token_count = completion_token_count + response_1.usage.completion_tokens

    response_1_text = response_1.choices[0].message.content.strip()

    # Replace <concept_library> in prompt_2 with the string representation of concept_library
    prompt_2_filled = prompt_2.replace("<concept_library>", str(list(concept_library)))

    # Send prompt_2 to OpenAI and get the JSON string
    response_2 = client.chat.completions.create(
        model=model_id,  # Adjust the model as necessary
        response_format={"type": "json_object"},
        messages=[
          {"role": "system", "content": "You are are an expert data analyst. You are part of a research team studying the role of concept hierarchies in determining the teaching quality of educational videos."},
          {"role": "user", "content": response_1_text + "\n\n" + prompt_2_filled,},
        ]
        # max_tokens=500  # Adjust max tokens as necessary
    )

    # Count tokens to track usage: 
    prompt_token_count = prompt_token_count + response_1.usage.prompt_tokens
    completion_token_count = completion_token_count + response_1.usage.completion_tokens

    response_2_text = response_2.choices[0].message.content.strip()

    if response_2_text[0:7] == "```json": response_2_text = response_2_text[7:]
    if response_2_text[-3:] == "```": response_2_text = response_2_text[0:-3]
    video_json = json.loads(response_2_text)

    # Check and repair errors in json structure
    repaired_json, anomalies = check_and_repair_json(video_json)
    if len(anomalies) > 0:
      json_structure_error_count = json_structure_error_count + len(anomalies)
      print("WARNING: JSON structure errors encountered. Here is a summary of the errors:")
      print(anomalies)
      video_json = repaired_json

    # Save the JSON string to the dataframe
    df.at[index, "activity_concept_hierarchy"] = video_json

    # Augment the concept library
    concept_library = expand_concepts(video_json, concept_library)

    if verbose: 
      print_dialog(prompt_1, response_1, prompt_2, response_2, video_json, concept_library)
    
    # Save the updated dataframe
    df.to_csv("video_transcripts_with_hierarchy_" + str(int(start_timestamp)) + ".csv", index=False)

    # Save the concept library using pickle
    with open("concept_library_" + str(int(start_timestamp)) + ".pkl", "wb") as file:
      pickle.dump(concept_library, file)

  except Exception as e:
    major_error_count = major_error_count + 1
    print("ERROR. Printing dialog:")
    print_dialog(prompt_1, response_1, prompt_2, response_2, video_json, concept_library)
    if debugging: 
      raise e

print("Total number of transcripts attempted:", len(df))
print("Major error count:", major_error_count)
print("JSON structure error count", json_structure_error_count)

print("FINAL CONCEPT LIBRARY:")
print(concept_library)

print("RUNTIME:", time.time() - start_timestamp, "seconds")

print("Prompt token count:", prompt_token_count)
print("Completion token count:", completion_token_count)

print("PRICING:")
if model_id == "gpt-4-1106-preview":
  print("Prompt tokens: $" + str(prompt_token_count * ((0.01/1000))))
  print("Completion tokens: $" + str(completion_token_count * ((0.03/1000))))
elif model_id == "gpt-3.5-turbo-1106":
  print("Prompt tokens: $" + str(prompt_token_count * ((0.001/1000))))
  print("Completion tokens: $" + str(completion_token_count * ((0.002/1000))))
else: 
  print("Unspecified model type, unable to calculate cost.")