import math
import statistics
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def get_embedding(model, tokenizer, text):
  inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
  outputs = model(**inputs)
  return outputs.last_hidden_state.mean(dim=1).squeeze()

def standardize(concept):
  return concept.lower().replace("_", " ").replace(" = ", "=").replace(" x ", "x").replace(" + ", "+").replace("^", "")

def lda_preprocess(text):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([word for word in text.lower().split() if word not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Function to get the topic distribution for a document
def get_lda_topic_distribution(lda_model, bow):
  topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0)
  return dict(topic_distribution)

def count_concepts(json_data, pri_concepts=None, supp_concepts=None):
  """
  Function to count the primary and supporting concepts found in a nested JSON structure.

  :param json_data: List of nested JSON objects.
  :return: dicts of primary and supporting concepts
  """

  if pri_concepts is None:
    pri_concepts = {}
  if supp_concepts is None:
    supp_concepts = {}

  for item in json_data["lesson"]:
    # Add concepts from the current level to the set
    if item["primary_concept"] in pri_concepts:
      pri_concepts[item["primary_concept"]] = pri_concepts[item["primary_concept"]] + 1
    else:
      pri_concepts[item["primary_concept"]] = 1
    for concept in item["supporting_concepts"]:
      if concept in supp_concepts:
        supp_concepts[concept] = supp_concepts[concept] + 1
      else:
        supp_concepts[concept] = 1
      
    # If there are nested activities, recurse into them
    if "activities" in item and len(item["activities"]) > 0:
      pri_concepts, supp_concepts = count_concepts({"lesson": item["activities"]}, pri_concepts, supp_concepts)
  
  return pri_concepts, supp_concepts


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