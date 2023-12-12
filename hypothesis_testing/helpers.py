import math
import statistics

def count_concepts(json_data, primary_concepts={}, supporting_concepts={}):
  """
  Function to count the primary and supporting concepts found in a nested JSON structure.

  :param json_data: List of nested JSON objects.
  :return: dicts of primary and supporting concepts
  """

  for item in json_data["lesson"]:
    # Add concepts from the current level to the set
    if item["primary_concept"] in primary_concepts:
      primary_concepts[item["primary_concept"]] = primary_concepts[item["primary_concept"]] + 1
    else:
      primary_concepts[item["primary_concept"]] = 1
    for concept in item["supporting_concepts"]:
      if concept in supporting_concepts:
        supporting_concepts[concept] = supporting_concepts[concept] + 1
      else:
        supporting_concepts[concept] = 1
      
    # If there are nested activities, recurse into them
    if "activities" in item and item["activities"]:
      primary_concepts, supporting_concepts = count_concepts({"lesson": item["activities"]}, primary_concepts, supporting_concepts)
  
  return primary_concepts, supporting_concepts


def merge_concept_counts(dict1, dict2):
  """
  Merges two dictionaries containing counts. If a key is present in both dictionaries,
  their counts are summed. Otherwise, the key is added with its count.

  :param dict1: First dictionary with counts.
  :param dict2: Second dictionary with counts.
  :return: Merged dictionary with summed counts.
  """
  merged_dict = dict1.copy()  # Start with a copy of the first dictionary

  for key, value in dict2.items():
    if key in merged_dict:
      merged_dict[key] += value  # Sum the counts if key is already present
    else:
      merged_dict[key] = value  # Add the new key with its count

  return merged_dict


def func_or_nan(func, values):
    # Check if the list is empty
    if not values:
        # Return NaN if the list is empty
        return math.nan
    else:
        # Calculate and return the func (e.g. mean) otherwise
        return func(values)