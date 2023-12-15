import pandas as pd
import pickle

# First run in pipeline (after similarity-based mapping with BERT)
# concept_library_pkl = "concept_library_1702443584.pkl"
# concept_mapping_dict_file = "concept_similarity_mappings_87_1702443584.pkl"
# concept_library_pkl_out = "concept_library_mapped_1702443584.pkl"
# add_unk = False
# print_library = False

# Second run in pipeline (after rarity-based mapping to "<UNK>")
# concept_library_pkl = "concept_library_mapped_1702443584.pkl"
# concept_mapping_dict_file = "concept_rarity_mappings_1702443584.pkl"
# concept_library_pkl_out = "concept_library_mapped_truncated_1702443584.pkl"
# add_unk = True
# print_library = True

# BERT rarity version
concept_library_pkl = "concept_library_mapped_1702443584.pkl"
concept_mapping_dict_file = "concept_bert_rarity_mappings_1702443584.pkl"
concept_library_pkl_out = "concept_library_mapped_bert_rarity_1702443584.pkl"
add_unk = False
print_library = True

with open(concept_library_pkl, 'rb') as file:
  concept_library = pickle.load(file)

with open(concept_mapping_dict_file, 'rb') as file:
  mapping_dict = pickle.load(file)

concept_library = {c for c in concept_library if c not in mapping_dict}

if add_unk:
  concept_library.add("<UNK>")

if print_library:
  print(list(concept_library))

with open(concept_library_pkl_out, "wb") as file:
  pickle.dump(concept_library, file)